"""Create ADNI evaluation dataset (Arrow format).

This script creates HuggingFace Arrow datasets for ADNI resting-state fMRI:
- All BOLD data has TR=3.0s (filtered at curation stage)
- Takes first 100 TRs (5 mins) from each session
- Uses existing Train/Val/Test split from metadata JSON

Supports parcellations: schaefer400, schaefer400_tians3, flat, a424, mni, mni_cortex

Data paths:
- Preprocessed data: data/fmriprep/output/
- Curation JSON: metadata/ADNI_curation.json
- Output datasets: data/processed/
"""

import argparse
import json
import logging
import sys
import tempfile
from pathlib import Path

import datasets as hfds
import numpy as np

import brainmarks.nisc as nisc
import brainmarks.readers as readers

# use smaller writer batch size to avoid OverflowError on very large mni data
# https://github.com/huggingface/datasets/issues/6422
hfds.config.DEFAULT_MAX_BATCH_SIZE = 256

logging.basicConfig(
    format="[%(levelname)s %(asctime)s]: %(message)s",
    level=logging.INFO,
    datefmt="%y-%m-%d %H:%M:%S",
)
logging.getLogger("nibabel").setLevel(logging.ERROR)

_logger = logging.getLogger(__name__)

ROOT = Path(__file__).parents[1]

# Data paths relative to dataset root
ADNI_FMRIPREP_ROOT = ROOT / "data/fmriprep"
ADNI_OUTPUT = ADNI_FMRIPREP_ROOT / "output"
ADNI_CURATION_JSON = ROOT / "metadata" / "ADNI_curation.json"

# All data has TR=3.0s (filtered at curation stage)
TARGET_TR = 3.0
# Keep first 100 TRs (5 mins) from each run
MAX_TRS = 100

# Split mapping from CSV
SPLIT_MAP = {
    "Train": "train",
    "Val": "validation",
    "Test": "test",
}


def ptid_to_sub(ptid: str) -> str:
    """Convert PTID format to sub directory format.

    Example: "168_S_6049" -> "168S6049"
    """
    return ptid.replace("_", "")


def scandate_to_ses(scandate: str) -> str:
    """Convert scandate format to session directory format.

    Example: "2020-10-12" -> "20201012"
    """
    return scandate.replace("-", "")


def main(args):
    outdir = ROOT / f"data/processed/adni.{args.space}.arrow"
    _logger.info("Generating dataset: %s", outdir)
    if outdir.exists() and not args.overwrite:
        _logger.warning("Output %s exists; exiting. Use --overwrite to replace.", outdir)
        return 1

    # Load metadata JSON
    with ADNI_CURATION_JSON.open() as f:
        curation_data = json.load(f)
    _logger.info("Loaded %d samples from curation JSON", len(curation_data))

    # Build samples by split
    samples_by_split = {"train": [], "validation": [], "test": []}

    for entry in curation_data:
        ptid = entry["PTID"]
        scandate = entry["SCANDATE"]
        split = SPLIT_MAP[entry["Partition"]]
        original_tr = float(entry["TR"])

        sub = ptid_to_sub(ptid)
        ses = scandate_to_ses(scandate)

        # Build file path based on space
        if args.space in readers.VOLUME_SPACES:
            # Volumetric MNI path
            file_path = (
                ADNI_OUTPUT
                / f"sub-{sub}"
                / f"ses-{ses}"
                / "func"
                / f"sub-{sub}_ses-{ses}_task-rest_space-MNI152NLin6Asym_res-2_desc-preproc_bold.nii.gz"
            )
        else:
            # CIFTI path (schaefer400, schaefer400_tians3, flat, a424)
            file_path = (
                ADNI_OUTPUT
                / f"sub-{sub}"
                / f"ses-{ses}"
                / "func"
                / f"sub-{sub}_ses-{ses}_task-rest_space-fsLR_den-91k_bold.dtseries.nii"
            )

        sample = {
            "sub": sub,
            "visit": ses,
            "mod": "MR",
            "task": "rest",
            "path": str(file_path.relative_to(ADNI_OUTPUT)),
            "fullpath": str(file_path),
            "original_tr": original_tr,
            "ptid": ptid,
            "scandate": scandate,
        }
        samples_by_split[split].append(sample)

    for split, samples in samples_by_split.items():
        _logger.info("Num samples (%s): %d", split, len(samples))

    # Load reader for target space
    reader = readers.READER_DICT[args.space]()
    dim = readers.DATA_DIMS[args.space]
    _logger.info("Using reader for space '%s' with dimension: %d", args.space, dim)

    # Features
    features = hfds.Features(
        {
            "sub": hfds.Value("string"),
            "visit": hfds.Value("string"),
            "mod": hfds.Value("string"),
            "task": hfds.Value("string"),
            "path": hfds.Value("string"),
            "start": hfds.Value("int32"),
            "end": hfds.Value("int32"),
            "tr": hfds.Value("float32"),
            "bold": hfds.Array2D(shape=(None, dim), dtype="float16"),
            "mean": hfds.Array2D(shape=(1, dim), dtype="float32"),
            "std": hfds.Array2D(shape=(1, dim), dtype="float32"),
        }
    )

    writer_batch_size = args.writer_batch_size
    if writer_batch_size is None:
        if args.space == "flat":
            writer_batch_size = 16
        elif args.space in {"mni", "mni_cortex"}:
            writer_batch_size = 8

    # Generate datasets
    with tempfile.TemporaryDirectory(prefix="huggingface-") as tmpdir:
        dataset_dict = {}
        for split, samples in samples_by_split.items():
            dataset_dict[split] = hfds.Dataset.from_generator(
                generate_samples,
                features=features,
                gen_kwargs={"samples": samples, "reader": reader, "dim": dim},
                num_proc=args.num_proc,
                split=hfds.NamedSplit(split),
                cache_dir=tmpdir,
                writer_batch_size=writer_batch_size,
                # fingerprint needed for mni/mni_cortex to avoid hashing the reader
                fingerprint=f"adni-{args.space}-{split}",
            )
        dataset = hfds.DatasetDict(dataset_dict)

        outdir.parent.mkdir(exist_ok=True, parents=True)
        dataset.save_to_disk(outdir, max_shard_size="300MB")

    _logger.info("Dataset saved to: %s", outdir)
    return 0


def generate_samples(samples: list[dict], *, reader, dim: int):
    """Generate samples for evaluation dataset."""
    for sample_info in samples:
        fullpath = sample_info["fullpath"]

        series = reader(fullpath)

        T, D = series.shape
        assert D == dim, f"Path {fullpath} has wrong dimension ({D} != {dim})"
        assert T >= MAX_TRS, f"Path {fullpath} does not have enough data ({T} < {MAX_TRS})"

        series = series[:MAX_TRS]
        series, mean, std = nisc.scale(series)

        yield {
            "sub": sample_info["sub"],
            "visit": sample_info["visit"],
            "mod": sample_info["mod"],
            "task": sample_info["task"],
            "path": sample_info["path"],
            "start": 0,
            "end": MAX_TRS,
            "tr": TARGET_TR,
            "bold": series.astype(np.float16),
            "mean": mean.astype(np.float32),
            "std": std.astype(np.float32),
        }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create ADNI evaluation dataset")
    parser.add_argument(
        "--space",
        type=str,
        default="schaefer400",
        choices=list(readers.READER_DICT),
        help="Target anatomical space for processing (default: schaefer400)",
    )
    parser.add_argument(
        "--num_proc", "-j", type=int, default=32, help="Number of parallel processes"
    )
    parser.add_argument(
        "--writer_batch_size",
        type=int,
        default=None,
        help="Arrow writer batch size (default: 16 for flat, otherwise datasets default)",
    )
    parser.add_argument(
        "--overwrite", "-x", action="store_true", help="Overwrite existing output directory"
    )
    args = parser.parse_args()
    sys.exit(main(args))
