import argparse
import json
import logging
import sys
import tempfile
from pathlib import Path

import datasets as hfds
import numpy as np
import pandas as pd
from cloudpathlib import AnyPath

import fmri_fm_eval.nisc as nisc
import fmri_fm_eval.readers as readers

logging.basicConfig(
    format="[%(levelname)s %(asctime)s]: %(message)s",
    level=logging.INFO,
    datefmt="%y-%m-%d %H:%M:%S",
)
logging.getLogger("nibabel").setLevel(logging.ERROR)

_logger = logging.getLogger(__name__)

ROOT = Path(__file__).parents[1]

SPLITS = ["train", "validation", "test"]

# Resample all time series to fixed TR
TARGET_TR = 2.0
# Keep first 150 TRs (5 mins) from each run
MAX_NUM_TRS = 150


def main(args):
    outdir = ROOT / f"data/processed/abide.{args.space}.arrow"
    _logger.info("Generating dataset: %s", outdir)
    if outdir.exists():
        _logger.warning("Output %s exists; exiting.", outdir)
        return 1

    # load table of curated subs and targets
    curated_df = pd.read_csv(ROOT / "metadata/ABIDE_curated.csv", dtype={"sub": str})
    # list of included bold paths (one per sub)
    curated_paths = np.loadtxt(ROOT / "metadata/ABIDE_curated_paths.txt", dtype=str)

    if args.space in {"a424", "mni", "mni_cortex", "schaefer400_tians3_buckner7"}:
        suffix = "_space-MNI152NLin6Asym_res-2_desc-preproc_bold.nii.gz"
    else:
        suffix = "_space-fsLR_den-91k_bold.dtseries.nii"

    # preprocessed data paths
    fmriprep_root = ROOT / "data/fmriprep"
    curated_paths = [fmriprep_root / p.replace("_bold.nii.gz", suffix) for p in curated_paths]
    assert all(p.exists() for p in curated_paths)

    # mapping of subs to assigned splits
    sub_split_map = {sub: split for sub, split in zip(curated_df["sub"], curated_df["split"])}
    # data paths for each split
    path_splits = {split: [] for split in SPLITS}
    for path in curated_paths:
        sub = path.parts[-3].split("-")[1]
        split = sub_split_map[sub]
        path_splits[split].append(path)

    # load the data reader for the target space and look up the data dimension.
    # all readers return a bold data array of shape (n_samples, dim).
    reader = readers.READER_DICT[args.space]()
    dim = readers.DATA_DIMS[args.space]

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
                gen_kwargs={"paths": paths, "root": fmriprep_root, "reader": reader, "dim": dim},
                num_proc=args.num_proc,
                split=hfds.NamedSplit(split),
                cache_dir=tmpdir,
            )
        dataset = hfds.DatasetDict(dataset_dict)

        outdir.parent.mkdir(exist_ok=True, parents=True)
        dataset.save_to_disk(outdir, max_shard_size="300MB")


def generate_samples(paths: list[Path], *, root: AnyPath, reader: readers.Reader, dim: int):
    for path in paths:
        sidecar_path = path.parent / (path.name.split(".")[0] + ".json")
        with sidecar_path.open() as f:
            sidecar_data = json.load(f)
        tr = float(sidecar_data["RepetitionTime"])
        meta = parse_abide_metadata(path)

        series = reader(path)
        series = nisc.resample_timeseries(series, tr=tr, new_tr=TARGET_TR, kind="pchip")

        T, D = series.shape
        assert D == dim
        if T < MAX_NUM_TRS:
            _logger.warning(f"Path {path} does not have enough data ({T}<{MAX_NUM_TRS}); skipping.")
            continue

        series = series[:MAX_NUM_TRS]
        series, mean, std = nisc.scale(series)

        sample = {
            **meta,
            "path": str(path.relative_to(root)),
            "n_frames": MAX_NUM_TRS,
            "tr": TARGET_TR,
            "bold": series.astype(np.float16),
            "mean": mean.astype(np.float32),
            "std": std.astype(np.float32),
        }
        yield sample


def parse_abide_metadata(path: Path):
    # CMU_a/sub-0050642/anat/sub-0050642_T1w.nii.gz
    dataset = path.parts[-4]
    site = dataset.split("_")[0]  # CMU_a -> CMU
    stem, ext = path.name.split(".", 1)
    stem, suffix = stem.rsplit("_", 1)
    meta = dict(item.split("-") for item in stem.split("_") if "-" in item)
    meta = {"sub": meta["sub"], "site": site, "dataset": dataset}
    return meta


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--space", type=str, default="fslr64k", choices=list(readers.READER_DICT))
    parser.add_argument("--num_proc", "-j", type=int, default=32)
    args = parser.parse_args()
    sys.exit(main(args))
