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

# clip length
NUM_FRAMES = 16

# Resample all time series to 1s tr.
TARGET_TR = 1.0
INTERPOLATION = "pchip"

# https://www.humanconnectome.org/hcp-protocols-ya-3t-imaging
# https://www.humanconnectome.org/hcp-protocols-ya-7t-imaging
HCP_TR = {"3T": 0.72, "7T": 1.0}

# 21 task conditions for 6 tasks
# (excludes gambling due to fast event related design)
HCP_TASK21_TASKS = ["EMOTION", "LANGUAGE", "MOTOR", "RELATIONAL", "SOCIAL", "WM"]
HCP_TASK21_TRIAL_TYPES = {
    "fear": 0,
    "neut": 1,
    "math": 2,
    "story": 3,
    "lf": 4,
    "lh": 5,
    "rf": 6,
    "rh": 7,
    "t": 8,
    "match": 9,
    "relation": 10,
    "mental": 11,
    "rnd": 12,
    "0bk_body": 13,
    "2bk_body": 14,
    "0bk_faces": 15,
    "2bk_faces": 16,
    "0bk_places": 17,
    "2bk_places": 18,
    "0bk_tools": 19,
    "2bk_tools": 20,
}


def main(args):
    outdir = ROOT / f"data/processed/hcpya-task21.{args.space}.arrow"
    _logger.info("Generating dataset: %s", outdir)
    if outdir.exists():
        _logger.warning("Output %s exists; exiting.", outdir)
        return 1

    # construct train/val/test splits by combining subject batches.
    # nb, across batches subjects are unrelated. we use the batches to dial how much
    # data to include.
    with (ROOT / "metadata/hcpya_subject_batch_splits.json").open() as f:
        sub_batch_splits = json.load(f)

    meta_df = pd.read_parquet(META_PATH)

    # get all task fmri paths for each batch of subjects
    path_splits = {}
    # map from paths to events
    all_events = {}
    for split, batch_ids in SUB_BATCH_SPLITS.items():
        split_subjects = [
            sub for batch_id in batch_ids for sub in sub_batch_splits[f"batch-{batch_id:02d}"]
        ]
        sub_mask = (
            meta_df["sub"].isin(split_subjects)
            & meta_df["task"].isin(HCP_TASK21_TASKS)
            & (meta_df["mag"] == "3T")  # tbf, these are all in 3T, but just to be explicit
            & (meta_df["dir"] == "LR")  # restrict to LR phase direction only
        )

        sub_df = meta_df.loc[sub_mask]

        # only keep subjects with complete task data
        counts = sub_df.groupby("sub").agg({"task": "nunique"})
        split_subjects = counts.index.values[counts["task"] == len(HCP_TASK21_TASKS)]
        sub_df = sub_df.loc[sub_df["sub"].isin(split_subjects)]

        for path, events in zip(sub_df["path"], sub_df["events"]):
            # "100307/MNINonLinear/Results/tfMRI_EMOTION_LR"
            # this way it works for both surface and mni (bit hacky)
            key = str(Path(path).parent)
            all_events[key] = events

        path_splits[split] = sorted(sub_df["path"].tolist())

    # volume space for a424 and mni, otherwise cifti space
    # TODO: hacky, the reader should know what input space it needs. we shouldn't need
    # to remember this in every script.
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
    # note, the mean and std are computed over the entire run and are redundant across
    # clips from the same run.
    features = hfds.Features(
        {
            "sub": hfds.Value("string"),
            "task": hfds.Value("string"),
            "cond": hfds.Value("string"),
            "cond_id": hfds.Value("int32"),
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
                gen_kwargs={
                    "paths": paths,
                    "root": root,
                    "all_events": all_events,
                    "reader": reader,
                },
                num_proc=args.num_proc,
                split=hfds.NamedSplit(split),
                cache_dir=tmpdir,
                # otherwise fingerprint crashes on mni space, ig bc of hashing the reader
                fingerprint=f"hcpya-task21-{args.space}-{split}",
            )
        dataset = hfds.DatasetDict(dataset_dict)

        outdir.parent.mkdir(exist_ok=True, parents=True)
        dataset.save_to_disk(outdir, max_shard_size="300MB")


def generate_samples(
    paths: list[str],
    *,
    root: AnyPath,
    all_events: dict[list[dict[str, float]]],
    reader: readers.Reader,
):
    for path, fullpath in prefetch(root, paths):
        meta = parse_hcp_metadata(fullpath)
        assert meta["mag"] == "3T"
        tr = HCP_TR[meta["mag"]]

        series = reader(fullpath)
        series, mean, std = nisc.scale(series)

        series = nisc.resample_timeseries(series, tr=tr, new_tr=TARGET_TR, kind=INTERPOLATION)
        tr = TARGET_TR

        key = str(Path(path).parent)
        events = all_events[key]
        for event in events:
            cond = event["trial_type"]
            if cond not in HCP_TASK21_TRIAL_TYPES:
                continue
            cond_id = HCP_TASK21_TRIAL_TYPES[cond]

            start = int(event["onset"] / tr)
            end = start + NUM_FRAMES
            if end > len(series):
                continue
            clip = series[start:end]

            sample = {
                "sub": meta["sub"],
                "task": meta["task"],
                "cond": cond,
                "cond_id": cond_id,
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
