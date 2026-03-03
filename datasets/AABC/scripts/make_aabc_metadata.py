import logging
from pathlib import Path

import datasets as hfds
import nibabel as nib

logging.getLogger("nibabel").setLevel(logging.ERROR)

# AABC uses consistent 0.72s TR across all scans
AABC_TR = 0.72

# Expected number of runs: 2,214 subjects * 4 scans = 8,856 total
# But some subjects may be missing scans, so we'll check after loading
AABC_EXPECTED_SUBJECTS = 2214  # subject-visits

NUM_PROC = 16

ROOT = Path(__file__).parents[1]

AABC_ROOT = ROOT / "data" / "raw"


def main():
    """Generate metadata for AABC dataset."""
    # Find all CIFTI files in AABC dataset
    series_paths = sorted(
        AABC_ROOT.rglob("*_Atlas_MSMAll_hp0_clean_rclean_tclean.dtseries.nii")
    )

    print(f"Found {len(series_paths)} scan files")

    # Expected: ~8,856 scans (2,214 subjects * 4 scans, minus some missing data)
    # Actual coverage based on exploration: ~96.2% have all 4 scans

    features = hfds.Features(
        {
            "sub": hfds.Value("string"),
            "visit": hfds.Value("string"),
            "mod": hfds.Value("string"),
            "task": hfds.Value("string"),
            "tr": hfds.Value("float32"),
            "n_frames": hfds.Value("int32"),
            "path": hfds.Value("string"),
        }
    )

    dataset = hfds.Dataset.from_generator(
        generate_metadata,
        features=features,
        gen_kwargs={"paths": series_paths},
        num_proc=NUM_PROC,
    )

    # Create metadata directory if it doesn't exist
    metadata_dir = ROOT / "metadata"
    metadata_dir.mkdir(parents=True, exist_ok=True)

    dataset.to_parquet(metadata_dir / "aabc_metadata.parquet")

    print(f"Saved metadata for {len(dataset)} scans")

    # Print statistics
    task_counts = {}
    for sample in dataset:
        task = sample["task"]
        task_counts[task] = task_counts.get(task, 0) + 1

    print("\nScan counts by task:")
    for task, count in sorted(task_counts.items()):
        print(f"  {task}: {count}")


def generate_metadata(paths: list[str]):
    """Generate metadata for each CIFTI file."""
    for path in paths:
        path = Path(path)
        meta = parse_aabc_metadata(path)

        # Load CIFTI header to get number of frames
        img = nib.load(path)
        n_frames = img.shape[0]

        sample = {
            **meta,
            "tr": AABC_TR,
            "n_frames": n_frames,
            "path": str(path.relative_to(AABC_ROOT)),
        }
        yield sample


def parse_aabc_metadata(path: Path) -> dict[str, str]:
    """Parse AABC metadata from file path.

    Path structure: HCA{subject_id}_V{visit}_MR/MNINonLinear/Results/{task}/{scan_name}.dtseries.nii

    Example:
        HCA6000030_V1_MR/MNINonLinear/Results/rfMRI_REST/rfMRI_REST_Atlas_MSMAll_hp0_clean_rclean_tclean.dtseries.nii

    Returns:
        dict with keys: sub (e.g., "HCA6000030"), visit (e.g., "V1"), mod (always "MR"), task (e.g., "REST")
    """
    # Parse subject-visit directory name: HCA6000030_V1_MR
    subject_visit_dir = path.parents[3].name
    parts = subject_visit_dir.split("_")

    # Extract: HCA6000030, V1, MR
    sub = parts[0]  # e.g., "HCA6000030"
    visit = parts[1]  # e.g., "V1"
    mod = parts[2]  # always "MR"

    # Extract task from scan directory name
    # rfMRI_REST → REST
    # tfMRI_CARIT_PA → CARIT
    # tfMRI_FACENAME_PA → FACENAME
    # tfMRI_VISMOTOR_PA → VISMOTOR
    scan_dir = path.parent.name
    if scan_dir.startswith("rfMRI_"):
        task = scan_dir.split("_")[1]  # rfMRI_REST → REST
    elif scan_dir.startswith("tfMRI_"):
        task = scan_dir.split("_")[1]  # tfMRI_CARIT_PA → CARIT
    else:
        task = scan_dir

    metadata = {
        "sub": sub,
        "visit": visit,
        "mod": mod,
        "task": task,
    }
    return metadata


if __name__ == "__main__":
    main()
