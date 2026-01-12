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
# PPMI TR is 1.0 (for AP/PA) or 2.5 (for LR/RL).
TARGET_TR = 2.5
# Keep first 120 TRs (5 mins) from each run
# All PPMI runs are 10 min long. We could also consider keeping more data.
MAX_NUM_TRS = 120


def main(args):
    out_root = AnyPath(args.out_root or (ROOT / "data/processed"))
    outdir = out_root / f"ppmi.{args.space}.arrow"
    _logger.info("Generating dataset: %s", outdir)
    if outdir.exists():
        _logger.warning("Output %s exists; exiting.", outdir)
        return 1

    # load curated subjects
    curated_df = pd.read_csv(ROOT / "metadata/PPMI_curated.csv", dtype={"Subject": str})

    # load curated paths and get first run per subject
    all_curated_paths = open(ROOT / "metadata/PPMI_curated_paths.txt").read().splitlines()
    curated_paths = {}
    for path in all_curated_paths:
        sub_with_prefix = path.split("/")[0]
        sub = sub_with_prefix.split("-")[1]
        if sub not in curated_paths:
            curated_paths[sub] = path
    curated_paths = list(curated_paths.values())

    if args.space in {"a424", "mni", "mni_cortex", "schaefer400_tians3_buckner7"}:
        suffix = "_space-MNI152NLin6Asym_res-2_desc-preproc_bold.nii.gz"
    else:
        suffix = "_space-fsLR_den-91k_bold.dtseries.nii"

    # preprocessed data paths
    curated_paths = [p.replace("_bold.nii.gz", suffix) for p in curated_paths]

    # mapping of subs to assigned splits
    sub_split_map = {sub: split for sub, split in zip(curated_df["Subject"], curated_df["split"])}
    # data paths for each split
    path_splits = {split: [] for split in SPLITS}
    for path in curated_paths:
        sub_with_prefix = path.split("/")[0]
        sub = sub_with_prefix.split("-")[1]
        split = sub_split_map[sub]
        path_splits[split].append(path)

    # load the data reader for the target space and look up the data dimension
    reader = readers.READER_DICT[args.space]()
    dim = readers.DATA_DIMS[args.space]

    # root can be local or remote.
    root = AnyPath(args.root or ROOT / "data/fmriprep")

    # define features
    features = hfds.Features(
        {
            "sub": hfds.Value("string"),
            "ses": hfds.Value("string"),
            "dir": hfds.Value("string"),
            "sex": hfds.Value("string"),
            "age": hfds.Value("float32"),
            "age_bin": hfds.Value("string"),
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
                fingerprint=f"ppmi-{args.space}-{split}",
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


def generate_samples(paths, *, root, curated_df, reader):
    for path, fullpath in prefetch(root, paths):
        sidecar_path = fullpath.parent / (fullpath.name.split(".")[0] + ".json")
        with sidecar_path.open() as f:
            sidecar_data = json.load(f)
        tr = float(sidecar_data["RepetitionTime"])

        # Extract subject and session from path
        stem = fullpath.name.split(".")[0]
        meta = dict(item.split("-") for item in stem.split("_") if "-" in item)
        sub = meta["sub"]

        # Get demographics from curated_df
        row = curated_df[curated_df["Subject"] == sub].iloc[0]
        meta = {
            "sub": sub,
            "ses": meta["ses"],
            "dir": meta["dir"],
            "sex": row["Sex"],
            "age": float(row["Age"]),
            "age_bin": row["age_bin"],
            "dx": row["Group"],
        }

        series = reader(fullpath)
        series = nisc.resample_timeseries(series, tr=tr, new_tr=TARGET_TR, kind="pchip")

        T, D = series.shape
        assert T >= MAX_NUM_TRS, f"Path {path} has too few TRs ({T}<{MAX_NUM_TRS})"

        series = series[:MAX_NUM_TRS]
        series, mean, std = nisc.scale(series)

        sample = {
            **meta,
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
