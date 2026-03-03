"""Create AABC pretraining dataset with REST only (WebDataset TAR format).

This script creates WebDataset TARs containing REST fMRI only:
- REST: 1912 TRs -> windowed to 3 x 500 TR segments (3 independent samples per subject)

Using REST-only ensures consistent sample sizes for pretraining.
Follows the same pattern as HCP-YA pretraining datasets.
"""

import argparse
import json
import logging
import os
from pathlib import Path

import numpy as np
import pandas as pd
import webdataset as wds

import brainmarks.nisc as nisc
import brainmarks.readers as readers

logging.basicConfig(
    format="[%(levelname)s %(asctime)s]: %(message)s",
    level=logging.INFO,
    datefmt="%y-%m-%d %H:%M:%S",
)
logging.getLogger("nibabel").setLevel(logging.ERROR)

_logger = logging.getLogger(__name__)

ROOT = Path(__file__).parents[1]
AABC_ROOT = ROOT / "data" / "raw"
# Pretraining uses batches 0-9 (10 batches)
# Batches 10-19 are reserved for evaluation
PRETRAIN_BATCHES = list(range(0, 10))

# AABC TR (constant across all tasks)
AABC_TR = 0.72

# Task configurations: (directory_name, file_suffix, window_size, max_windows)
# REST only: window into 500-TR segments (3 independent samples per subject)
TASK_CONFIG = {
    "REST": ("rfMRI_REST", "rfMRI_REST_Atlas_MSMAll_hp0_clean_rclean_tclean.dtseries.nii", 500, 3),
}

DEFAULT_SHARD_SIZE_MB = 700


def main(args):
    outdir = Path(args.outdir) / f"aabc-{args.space}"
    outdir.mkdir(parents=True, exist_ok=True)

    _logger.info("Generating WebDataset TARs for AABC pretraining (all tasks)")
    _logger.info("Output directory: %s", outdir)
    _logger.info("Space: %s", args.space)
    _logger.info("Shard target size: %d MB", args.shard_size_mb)

    # Load metadata
    metadata_df = pd.read_parquet(ROOT / "metadata/aabc_metadata.parquet")
    _logger.info("Total scans in metadata: %d", len(metadata_df))

    # Load subject batch splits
    with (ROOT / "scripts/aabc_subject_batch_splits.json").open() as f:
        sub_batch_splits = json.load(f)

    # Get all subjects in pretraining batches (0-13)
    pretrain_subjects = []
    for batch_id in PRETRAIN_BATCHES:
        pretrain_subjects.extend(sub_batch_splits[f"batch-{batch_id:02d}"])

    _logger.info("Pretraining subjects: %d", len(pretrain_subjects))

    # Filter metadata for pretraining subjects only
    metadata_df = metadata_df[metadata_df["sub"].isin(pretrain_subjects)]
    _logger.info("Pretraining scans (after subject filtering): %d", len(metadata_df))

    # Log task breakdown
    task_counts = metadata_df["task"].value_counts()
    _logger.info("Task breakdown: %s", task_counts.to_dict())

    # Initialize reader for target space
    reader = readers.READER_DICT[args.space]()
    dim = readers.DATA_DIMS[args.space]
    _logger.info("Using %s reader with dimension: %d", args.space, dim)

    # Sort by subject-visit-task for consistent sharding
    metadata_df = metadata_df.sort_values(["sub", "visit", "task"])

    # Generate all samples (with windowing for REST)
    all_samples = []
    for _, row in metadata_df.iterrows():
        task = row["task"]
        if task not in TASK_CONFIG:
            _logger.warning("Unknown task %s, skipping", task)
            continue

        task_dir, suffix, window_size, max_windows = TASK_CONFIG[task]

        # For REST, create multiple windows; for tasks, single window
        for segment in range(max_windows):
            all_samples.append(
                {
                    "sub": row["sub"],
                    "visit": row["visit"],
                    "task": task,
                    "path": row["path"],
                    "segment": segment,
                    "window_size": window_size,
                }
            )

    _logger.info("Total samples (with windowing): %d", len(all_samples))

    if not args.overwrite:
        existing = list(outdir.glob("aabc-*.tar"))
        if existing:
            _logger.error(
                "Found %d existing shards in %s. Use --overwrite to replace.", len(existing), outdir
            )
            return

    total_written = 0
    total_skipped = 0
    shard_size_bytes = int(args.shard_size_mb * 1024 * 1024)

    pattern = str(outdir / "aabc-%06d.tar")
    with wds.ShardWriter(pattern, maxsize=shard_size_bytes) as sink:
        for sample in all_samples:
            # Load CIFTI file
            path = AABC_ROOT / sample["path"]
            series = reader(path)

            T, D = series.shape
            assert D == dim, f"Expected dim {dim}, got {D}"

            window_size = sample["window_size"]
            segment = sample["segment"]
            start = segment * window_size
            end = start + window_size

            # Check if we have enough data for this segment
            if end > T:
                if segment == 0:
                    # For first segment, use what we have if it's close enough
                    if T >= window_size * 0.9:
                        end = T
                        window_size = T
                    else:
                        _logger.warning(
                            "Subject %s visit %s task %s has only %d TRs (< %d), skipping",
                            sample["sub"],
                            sample["visit"],
                            sample["task"],
                            T,
                            window_size,
                        )
                        total_skipped += 1
                        continue
                else:
                    # Skip additional segments that don't have enough data
                    continue

            # Extract window
            series_window = series[start:end]

            # Z-score normalization
            series_window, mean, std = nisc.scale(series_window)

            # Convert to float16 to save space
            series_f16 = series_window.astype(np.float16)

            # Create sample key: {subject}_{visit}_{task}_{segment}
            key = f"{sample['sub']}_{sample['visit']}_{sample['task']}_{segment}"

            # Write to TAR
            sink.write(
                {
                    "__key__": key,
                    "npy": series_f16.tobytes(),
                    "json": json.dumps(
                        {
                            "sub": sample["sub"],
                            "visit": sample["visit"],
                            "task": sample["task"],
                            "tr": AABC_TR,
                            "n_frames": series_f16.shape[0],
                            "dim": dim,
                            "dtype": "float16",
                            "shape": list(series_f16.shape),
                            "mean": mean.squeeze().astype(np.float32).tolist(),
                            "std": std.squeeze().astype(np.float32).tolist(),
                            "segment": segment,
                            "start": start,
                            "end": end,
                        }
                    ),
                }
            )
            total_written += 1

    _logger.info("WebDataset TAR creation complete!")
    _logger.info("Total samples written: %d", total_written)
    _logger.info("Total samples skipped: %d", total_skipped)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create AABC pretraining dataset")
    parser.add_argument(
        "--space",
        type=str,
        default="flat",
        choices=list(readers.READER_DICT),
        help="Target anatomical space for processing (default: flat)",
    )
    parser.add_argument(
        "--outdir",
        type=str,
        default=str(AABC_ROOT),
        help="Output directory for WebDataset TARs (default: AABC_data)",
    )
    parser.add_argument(
        "--shard_size_mb",
        type=int,
        default=DEFAULT_SHARD_SIZE_MB,
        help="Target shard size in MB (default: 700)",
    )
    parser.add_argument(
        "--overwrite", action="store_true", help="Overwrite existing shards in the output directory"
    )
    args = parser.parse_args()
    main(args)
