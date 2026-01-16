import argparse
import json
import logging
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import datasets as hfds
import numpy as np
import pandas as pd
from cloudpathlib import AnyPath, CloudPath, S3Path

import fmri_fm_eval.nisc as nisc
import fmri_fm_eval.readers as readers
import fmri_fm_eval.utils as ut

logging.basicConfig(
    format="[%(levelname)s %(asctime)s]: %(message)s",
    level=logging.INFO,
    datefmt="%y-%m-%d %H:%M:%S",
)
logging.getLogger("nibabel").setLevel(logging.ERROR)
logging.getLogger("botocore").setLevel(logging.ERROR)  # quiet aws credential log msg

_logger = logging.getLogger(__name__)

ROOT = Path(__file__).parents[1]

SPLITS = ["train", "validation", "test"]

# Resample all time series to fixed TR
TARGET_TR = 2.0
# Keep first 150 TRs (5 mins) from each run
MAX_NUM_TRS = 150


def main(args):
    out_root = AnyPath(args.out_root or (ROOT / "data/processed"))
    outdir = out_root / f"abide.{args.space}.arrow"
    _logger.info("Generating dataset: %s", outdir)
    if outdir.exists():
        _logger.info("Output %s exists; exiting.", outdir)
        return

    # load table of curated subs and targets
    curated_df = pd.read_csv(ROOT / "metadata/ABIDE_curated.csv", dtype={"sub": str})
    # list of included bold paths (one per sub)
    curated_paths = np.loadtxt(ROOT / "metadata/ABIDE_curated_paths.txt", dtype=str)

    if args.space in readers.VOLUME_SPACES:
        suffix = "_space-MNI152NLin6Asym_res-2_desc-preproc_bold.nii.gz"
    else:
        suffix = "_space-fsLR_den-91k_bold.dtseries.nii"

    # preprocessed data paths
    curated_paths = [p.replace("_bold.nii.gz", suffix) for p in curated_paths]

    # mapping of subs to assigned splits
    sub_split_map = {sub: split for sub, split in zip(curated_df["sub"], curated_df["split"])}
    # data paths for each split
    path_splits = {split: [] for split in SPLITS}
    for path in curated_paths:
        sub = path.split("/")[1].split("-")[1]
        split = sub_split_map[sub]
        path_splits[split].append(path)

    # load the data reader for the target space and look up the data dimension.
    # all readers return a bold data array of shape (n_samples, dim).
    reader = readers.READER_DICT[args.space]()
    dim = readers.DATA_DIMS[args.space]

    # root can be local or remote.
    root = AnyPath(args.root or ROOT / "data/fmriprep")

    # the bold data are scaled to mean 0, stdev 1 and then truncated to float16 to save
    # space. but we keep the mean and std to reverse this since some models need this.
    features = hfds.Features(
        {
            "sub": hfds.Value("string"),
            "site": hfds.Value("string"),
            "dataset": hfds.Value("string"),
            "path": hfds.Value("string"),
            "n_frames": hfds.Value("int32"),
            "tr": hfds.Value("float32"),
            "confounds": hfds.List(
                hfds.List(hfds.Value("float32"))
            ),  # List for variable column counts
            "confounds_columns": hfds.List(hfds.Value("string")),
            "bold": hfds.Array2D(shape=(None, dim), dtype="float16"),
            "mean": hfds.Array2D(shape=(1, dim), dtype="float32"),
            "std": hfds.Array2D(shape=(1, dim), dtype="float32"),
        }
    )

    # generate the datasets with huggingface. cache to a temp dir to save space.
    with tempfile.TemporaryDirectory(prefix="huggingface-") as tmpdir:
        dataset_dict = {}
        for split, paths in path_splits.items():
            dataset_dict[split] = hfds.Dataset.from_generator(
                generate_samples,
                features=features,
                gen_kwargs={"paths": paths, "root": root, "reader": reader},
                num_proc=args.num_proc,
                split=hfds.NamedSplit(split),
                cache_dir=tmpdir,
                # otherwise fingerprint crashes on mni space
                fingerprint=f"abide-{args.space}-{split}",
            )
        dataset = hfds.DatasetDict(dataset_dict)

        if isinstance(outdir, S3Path):
            _logger.info("Saving to s3: %s", outdir)
            tmp_outdir = Path(tmpdir) / outdir.name
            # in theory save_to_disk should support s3, but idk why it wasn't working
            dataset.save_to_disk(tmp_outdir, max_shard_size="300MB")
            ut.rsync(tmp_outdir, outdir)
        else:
            _logger.info("Saving locally: %s", outdir)
            dataset.save_to_disk(outdir, max_shard_size="300MB")


def generate_samples(paths: list[Path], *, root: AnyPath, reader: readers.Reader):
    for path, fullpath in prefetch(root, paths):
        sidecar_path = fullpath.parent / (fullpath.name.split(".")[0] + ".json")
        with sidecar_path.open() as f:
            sidecar_data = json.load(f)
        tr = float(sidecar_data["RepetitionTime"])
        meta = parse_abide_metadata(path)

        series = reader(fullpath)
        series = nisc.resample_timeseries(series, tr=tr, new_tr=TARGET_TR, kind="pchip")

        T, D = series.shape
        assert T >= MAX_NUM_TRS, f"Path {path} has too few TRs ({T}<{MAX_NUM_TRS})"

        series = series[:MAX_NUM_TRS]
        series, mean, std = nisc.scale(series)

        # Load and resample confounds
        confounds, confounds_columns = nisc.load_fmriprep_confounds(fullpath)
        confounds = nisc.resample_timeseries(confounds, tr=tr, new_tr=TARGET_TR, kind="pchip")
        confounds = confounds[:MAX_NUM_TRS]

        sample = {
            "sub": meta["sub"],
            "site": meta["site"],
            "dataset": meta["dataset"],
            "path": str(path),
            "n_frames": MAX_NUM_TRS,
            "tr": TARGET_TR,
            "bold": series.astype(np.float16),
            "mean": mean.astype(np.float32),
            "std": std.astype(np.float32),
            "confounds": confounds.tolist(),  # Convert to list for Sequence(Sequence())
            "confounds_columns": confounds_columns,
        }
        yield sample


def prefetch(root: AnyPath, paths: list[str], *, max_workers: int = 1):
    """Prefetch files from remote storage."""

    with tempfile.TemporaryDirectory(prefix="prefetch-") as tmpdir:

        def fn(path: str):
            fullpath = root / path
            if isinstance(fullpath, CloudPath):
                tmppath = Path(tmpdir) / path
                tmppath.parent.mkdir(parents=True, exist_ok=True)

                # get sidecar too (hack)
                stem = fullpath.name.split(".")[0]
                sidecar = fullpath.parent / f"{stem}.json"
                tmpsidecar = tmppath.parent / f"{stem}.json"
                sidecar.download_to(tmpsidecar)

                # get confounds TSV file
                confounds_prefix = stem.split("_space-")[0]
                confounds_file = (
                    fullpath.parent / f"{confounds_prefix}_desc-confounds_timeseries.tsv"
                )
                tmp_confounds = tmppath.parent / f"{confounds_prefix}_desc-confounds_timeseries.tsv"
                confounds_file.download_to(tmp_confounds)

                fullpath = fullpath.download_to(tmppath)

            return path, fullpath

        with ThreadPoolExecutor(max_workers) as executor:
            futures = [executor.submit(fn, p) for p in paths]

            for future in futures:
                path, fullpath = future.result()
                yield path, fullpath

                if str(fullpath).startswith(tmpdir):
                    fullpath.unlink()


def parse_abide_metadata(path: Path):
    # CMU_a/sub-0050642/anat/sub-0050642_T1w.nii.gz
    path = Path(path)
    dataset = path.parts[-4]
    site = dataset.split("_")[0]  # CMU_a -> CMU
    stem, ext = path.name.split(".", 1)
    stem, suffix = stem.rsplit("_", 1)
    meta = dict(item.split("-") for item in stem.split("_") if "-" in item)
    meta = {"sub": meta["sub"], "site": site, "dataset": dataset}
    return meta


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=str, default=None)
    parser.add_argument("--out-root", type=str, default=None)
    parser.add_argument(
        "--space", type=str, default="schaefer400", choices=list(readers.READER_DICT)
    )
    parser.add_argument("--num_proc", "-j", type=int, default=32)
    args = parser.parse_args()
    sys.exit(main(args))
