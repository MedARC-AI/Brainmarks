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
from cloudpathlib import AnyPath, CloudPath

import fmri_fm_eval.nisc as nisc
import fmri_fm_eval.readers as readers

# use smaller writer batch size to avoid OverflowError on very large mni data
# https://github.com/huggingface/datasets/issues/6422
hfds.config.DEFAULT_MAX_BATCH_SIZE = 256

logging.basicConfig(
    format="[%(levelname)s %(asctime)s]: %(message)s",
    level=logging.INFO,
    datefmt="%y-%m-%d %H:%M:%S",
)
logging.getLogger("nibabel").setLevel(logging.ERROR)
logging.getLogger("botocore").setLevel(logging.ERROR)  # quiet aws credential log msg

_logger = logging.getLogger(__name__)

ROOT = Path(__file__).parents[1]
HCP_ROOT = ROOT / "data/sourcedata/HCP_1200"
META_PATH = ROOT / "metadata/hcpya_metadata.parquet"

# ~600 subs total, 4:1:1 ratio
SUB_BATCH_SPLITS = {
    "train": [0, 1, 2, 3, 4, 5, 6, 7],
    "validation": [16, 17],
    "test": [18, 19],
}
# number of runs per batch
NUM_RUNS_PER_BATCH = 200
# clip length
NUM_FRAMES = 16
# clip sampling stride
STRIDE = 64

# Resample all time series to 1s tr.
TARGET_TR = 1.0
INTERPOLATION = "pchip"

# https://www.humanconnectome.org/hcp-protocols-ya-3t-imaging
# https://www.humanconnectome.org/hcp-protocols-ya-7t-imaging
HCP_TR = {"3T": 0.72, "7T": 1.0}

# Fixed seed for sampling runs.
SEED = 8581


def main(args):
    outdir = ROOT / f"data/processed/hcpya-clips.{args.space}.arrow"
    _logger.info("Generating dataset: %s", outdir)
    if outdir.exists():
        _logger.warning("Output %s exists; exiting.", outdir)
        return 1

    # construct train/val/test splits by combining subject batches.
    # nb, across batches subjects are unrelated. we use the batches to dial how much
    # data to include.
    with (ROOT / "metadata/hcpya_subject_batch_splits.json").open() as f:
        sub_batch_splits = json.load(f)

    rng = np.random.default_rng(SEED)
    meta_df = pd.read_parquet(META_PATH)

    # pick a random sample of paths after restricting to the given batches of subjects.
    path_splits = {}
    for split, batch_ids in SUB_BATCH_SPLITS.items():
        split_subjects = [
            sub for batch_id in batch_ids for sub in sub_batch_splits[f"batch-{batch_id:02d}"]
        ]
        sub_mask = meta_df["sub"].isin(split_subjects)
        split_paths = sorted(meta_df.loc[sub_mask, "path"].values)

        num_runs = len(batch_ids) * NUM_RUNS_PER_BATCH
        split_paths = rng.choice(split_paths, num_runs, replace=False).tolist()
        path_splits[split] = split_paths
        _logger.info(f"Split ({split}): N={len(split_paths)}\n{split_paths[:5]}")

    # volume space for a424 and mni, otherwise cifti space
    # TODO: hacky, the reader should know what input space it needs. we shouldn't need
    # to remember this in every script.
    # TDOO: indeed, this sucks
    if args.space in {"a424", "mni", "mni_cortex", "schaefer400_tians3_buckner7"}:
        for split, split_paths in path_splits.items():
            path_splits[split] = [
                p.replace("_Atlas_MSMAll.dtseries.nii", ".nii.gz") for p in split_paths
            ]

    # load the data reader for the target space and look up the data dimension.
    # all readers return a bold data array of shape (n_samples, dim).
    reader = readers.READER_DICT[args.space]()
    dim = readers.DATA_DIMS[args.space]

    # root can be local or remote.
    root = AnyPath(args.root or HCP_ROOT)

    # the bold data are scaled to mean 0, stdev 1 and then truncated to float16 to save
    # space. but we keep the mean and std to reverse this since some models need this.
    features = hfds.Features(
        {
            "sub": hfds.Value("string"),
            "mod": hfds.Value("string"),
            "task": hfds.Value("string"),
            "mag": hfds.Value("string"),
            "dir": hfds.Value("string"),
            "path": hfds.Value("string"),
            "start": hfds.Value("int32"),
            "end": hfds.Value("int32"),
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
                gen_kwargs={"paths": paths, "root": root, "reader": reader},
                num_proc=args.num_proc,
                split=hfds.NamedSplit(split),
                cache_dir=tmpdir,
                # otherwise fingerprint crashes on mni space, ig bc of hashing the reader
                fingerprint=f"hcpya-clips-{args.space}-{split}",
            )
        dataset = hfds.DatasetDict(dataset_dict)

        outdir.parent.mkdir(exist_ok=True, parents=True)
        dataset.save_to_disk(outdir, max_shard_size="300MB")


def generate_samples(paths: list[str], *, root: AnyPath, reader: readers.Reader):
    for path, fullpath in prefetch(root, paths):
        meta = parse_hcp_metadata(fullpath)
        tr = HCP_TR[meta["mag"]]

        series = reader(fullpath)
        series, mean, std = nisc.scale(series)

        series = nisc.resample_timeseries(series, tr=tr, new_tr=TARGET_TR, kind=INTERPOLATION)
        tr = TARGET_TR

        for start in range(0, len(series) - STRIDE + 1, STRIDE):
            end = start + NUM_FRAMES
            clip = series[start:end]
            assert len(clip) == NUM_FRAMES

            sample = {
                **meta,
                "path": str(path),
                "start": start,
                "end": end,
                "n_frames": len(clip),
                "tr": tr,
                "bold": clip.astype(np.float16),
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
                fullpath = fullpath.download_to(tmppath)
            return path, fullpath

        with ThreadPoolExecutor(max_workers) as executor:
            futures = [executor.submit(fn, p) for p in paths]

            for future in futures:
                path, fullpath = future.result()
                yield path, fullpath

                if str(fullpath).startswith(tmpdir):
                    fullpath.unlink()


def parse_hcp_metadata(path: Path) -> dict[str, str]:
    sub = path.parents[3].name
    acq = path.parent.name
    if "7T" in acq:
        mod, task, mag, dir = acq.split("_")
    else:
        mod, task, dir = acq.split("_")
        mag = "3T"
    metadata = {"sub": sub, "mod": mod, "task": task, "mag": mag, "dir": dir}
    return metadata


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=str, default=None)
    parser.add_argument(
        "--space", type=str, default="schaefer400", choices=list(readers.READER_DICT)
    )
    parser.add_argument("--num_proc", "-j", type=int, default=32)
    args = parser.parse_args()
    sys.exit(main(args))
