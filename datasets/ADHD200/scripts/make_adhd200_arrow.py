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
    outdir = out_root / f"adhd200.{args.space}.arrow"
    _logger.info("Generating dataset: %s", outdir)
    if outdir.exists():
        _logger.warning("Output %s exists; exiting.", outdir)
        return 1

    # load table of curated subs and targets
    curated_df = pd.read_csv(ROOT / "metadata/ADHD200_curated.csv", dtype={"sub": str})
    curated_paths = curated_df["path"].values

    if args.space in {"a424", "mni", "mni_cortex", "schaefer400_tians3_buckner7"}:
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
            "gender": hfds.Value("string"),
            "dx": hfds.Value("string"),
            "path": hfds.Value("string"),
            "n_frames": hfds.Value("int32"),
            "tr": hfds.Value("float32"),
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
                gen_kwargs={
                    "paths": paths,
                    "root": root,
                    "curated_df": curated_df,
                    "reader": reader,
                },
                num_proc=args.num_proc,
                split=hfds.NamedSplit(split),
                cache_dir=tmpdir,
                # otherwise fingerprint crashes on mni space, ig bc of hashing the reader
                fingerprint=f"adhd200-{args.space}-{split}",
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


def generate_samples(
    paths: list[Path], *, root: AnyPath, curated_df: pd.DataFrame, reader: readers.Reader
):
    for path, fullpath in prefetch(root, paths):
        sidecar_path = fullpath.parent / (fullpath.name.split(".")[0] + ".json")
        with sidecar_path.open() as f:
            sidecar_data = json.load(f)
        tr = float(sidecar_data["RepetitionTime"])
        meta = parse_adhd200_metadata(path)

        sub = meta["sub"]
        row = curated_df.query(f"sub == '{sub}'").iloc[0]

        series = reader(fullpath)
        series = nisc.resample_timeseries(series, tr=tr, new_tr=TARGET_TR, kind="pchip")

        T, D = series.shape
        assert T >= MAX_NUM_TRS, f"Path {path} has too few TRs ({T}<{MAX_NUM_TRS})"

        series = series[:MAX_NUM_TRS]
        series, mean, std = nisc.scale(series)

        sample = {
            "sub": sub,
            "site": meta["site"],
            "gender": row["gender"],
            "dx": row["dx"],
            "path": str(path),
            "n_frames": MAX_NUM_TRS,
            "tr": TARGET_TR,
            "bold": series.astype(np.float16),
            "mean": mean.astype(np.float32),
            "std": std.astype(np.float32),
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

                fullpath = fullpath.download_to(tmppath)

            return path, fullpath

        with ThreadPoolExecutor(max_workers) as executor:
            futures = [executor.submit(fn, p) for p in paths]

            for future in futures:
                path, fullpath = future.result()
                yield path, fullpath

                if str(fullpath).startswith(tmpdir):
                    fullpath.unlink()


def parse_adhd200_metadata(path: Path):
    # Brown/sub-0026001/ses-1/func/sub-0026001_ses-1_task-rest_run-1_bold.nii.gz
    path = Path(path)
    dataset = path.parts[0]
    site = dataset.split("_")[0]  # Peking_1 -> Peking
    stem, ext = path.name.split(".", 1)
    stem, suffix = stem.rsplit("_", 1)
    meta = dict(item.split("-") for item in stem.split("_") if "-" in item)
    # set sometimes missing keys
    for k in ["task", "run", "acq"]:
        meta[k] = meta.get(k)
    meta = {"site": site, "dataset": dataset, **meta, "suffix": suffix}
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
