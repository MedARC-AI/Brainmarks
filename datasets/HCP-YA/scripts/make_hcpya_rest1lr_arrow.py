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
TARGET_PATH = ROOT / "metadata/hcpya_pheno_targets.csv"

# ~600 subs total, 4:1:1 ratio
SUB_BATCH_SPLITS = {
    "train": [0, 1, 2, 3, 4, 5, 6, 7],
    "validation": [16, 17],
    "test": [18, 19],
}

NUM_SUBS = {
    "train": 406,
    "validation": 88,
    "test": 104,
}

# https://www.humanconnectome.org/hcp-protocols-ya-3t-imaging
# https://www.humanconnectome.org/hcp-protocols-ya-7t-imaging
HCP_TR = {"3T": 0.72, "7T": 1.0}

# Only keep the first 500 TRs = 6 mins for each run
MAX_NUM_TRS = 500

# Fixed seed for sampling subjects.
SEED = 8581


def main(args):
    outdir = ROOT / f"data/processed/hcpya-rest1lr.{args.space}.arrow"
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

    target_df = pd.read_csv(TARGET_PATH, dtype={"Subject": str})
    # drop subjects with missing phenotypic data
    target_df = target_df.dropna()
    target_df = target_df.set_index("Subject", drop=True)
    target_df.columns = [col.lower() for col in target_df.columns]

    path_splits = {}
    for split, batch_ids in SUB_BATCH_SPLITS.items():
        split_subjects = [
            sub for batch_id in batch_ids for sub in sub_batch_splits[f"batch-{batch_id:02d}"]
        ]
        # filter for selected runs (one per subject)
        sub_mask = (
            meta_df["sub"].isin(split_subjects)
            & meta_df["sub"].isin(target_df.index)
            & (meta_df["task"] == "REST1")
            & (meta_df["mag"] == "3T")  # tbf, these are all in 3T, but just to be explicit
            & (meta_df["dir"] == "LR")  # restrict to LR phase direction only
        )

        sub_df = meta_df.loc[sub_mask]
        assert (sub_df["sub"].value_counts() == 1).all(), "expected exactly one run per subject"
        split_subjects = sub_df["sub"].values

        # stratify by gender
        # want balanced gender in each split
        strat_split_indices = stratify_subsample(
            target_df.loc[split_subjects, "gender"],
            {"F": 0.5, "M": 0.5},
            random_state=SEED,
        )
        split_subjects = split_subjects[strat_split_indices]
        sub_df = sub_df.loc[sub_df["sub"].isin(split_subjects)]

        counts = target_df.loc[split_subjects, "gender"].value_counts().to_dict()
        _logger.info(f"Split ({split}): N={len(sub_df)} {counts}\n{split_subjects[:5].tolist()}")

        path_splits[split] = paths = sorted(sub_df["path"].tolist())
        assert len(paths) == NUM_SUBS[split], "unexpected number of paths"

    # volume space for a424 and mni, otherwise cifti space
    # TODO: hacky, the reader should know what input space it needs. we shouldn't need
    # to remember this in every script.
    # TODO: this sucks
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

    target_features = {
        "gender": hfds.Value("string"),
        "age_q": hfds.Value("int32"),
        "neofac_n_q": hfds.Value("int32"),
        "flanker_unadj_q": hfds.Value("int32"),
        "pmat24_a_cr_q": hfds.Value("int32"),
    }
    features = hfds.Features(
        {
            "sub": hfds.Value("string"),
            **target_features,
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
                    "target_df": target_df,
                    "target_cols": tuple(target_features),
                    "reader": reader,
                },
                num_proc=args.num_proc,
                split=hfds.NamedSplit(split),
                cache_dir=tmpdir,
                # otherwise fingerprint crashes on mni space, ig bc of hashing the reader
                fingerprint=f"hcpya-rest1lr-{args.space}-{split}",
            )
        dataset = hfds.DatasetDict(dataset_dict)

        outdir.parent.mkdir(exist_ok=True, parents=True)
        dataset.save_to_disk(outdir, max_shard_size="300MB")


def generate_samples(
    paths: list[str],
    *,
    root: AnyPath,
    target_df: pd.DataFrame,
    target_cols: list[str],
    reader: readers.Reader,
):
    for path, fullpath in prefetch(root, paths):
        meta = parse_hcp_metadata(fullpath)
        assert meta["mag"] == "3T"
        tr = HCP_TR[meta["mag"]]

        sub = meta["sub"]
        row = target_df.loc[sub].to_dict()
        targets = {col: row[col] for col in target_cols}

        series = reader(fullpath)

        assert len(series) >= MAX_NUM_TRS
        start, end = 0, MAX_NUM_TRS
        series = series[start:end]
        series, mean, std = nisc.scale(series)

        sample = {
            "sub": sub,
            **targets,
            "path": str(path),
            "start": start,
            "end": end,
            "n_frames": len(series),
            "tr": tr,
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


def stratify_subsample(
    y: np.ndarray,
    freqs: dict[int, int],
    n_samples: int | None = None,
    random_state: int | np.random.Generator | None = None,
):
    rng = np.random.default_rng(random_state)

    labels, counts = np.unique(y, return_counts=True)
    probs = counts / counts.sum()
    target_probs = np.array([freqs[label] for label in labels])
    target_probs = target_probs / target_probs.sum()

    # choose number of samples so that we have enough of every class
    if n_samples is None:
        fraction = np.min(probs / target_probs)
        n_samples = int(fraction * len(y))

    indices = []
    for label, prob in zip(labels, target_probs):
        cls_ids = np.flatnonzero(y == label)
        n_cls = int(round(prob * n_samples))
        assert n_cls <= len(cls_ids), f"not enough samples of class {label}"
        cls_ids = rng.choice(cls_ids, size=n_cls, replace=False)
        indices.append(cls_ids)

    indices = np.sort(np.concatenate(indices))
    return indices


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=str, default=None)
    parser.add_argument(
        "--space", type=str, default="schaefer400", choices=list(readers.READER_DICT)
    )
    parser.add_argument("--num_proc", "-j", type=int, default=32)
    args = parser.parse_args()
    sys.exit(main(args))
