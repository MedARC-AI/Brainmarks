import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(
    format="[%(levelname)s %(asctime)s]: %(message)s",
    level=logging.INFO,
    datefmt="%y-%m-%d %H:%M:%S",
)

_logger = logging.getLogger(__name__)

ROOT = Path(__file__).parents[1]
AABC_ROOT = ROOT / "data" / "raw"
AABC_CSV_PATH = AABC_ROOT / "AABC_subjects_2026_01_03_14_21_56.csv"

# Gender mapping (classification target)
GENDER_MAP = {
    "F": 0,
    "M": 1,
}
GENDER_CLASSES = ["F", "M"]

# quantile binning with balance check
PRIMARY_BINS = 4
FALLBACK_BINS = 3
MIN_BIN_FRACTION = 0.20

# Per-target bin overrides (use exact bin count, skip balance rule)
BIN_OVERRIDES = {
    "age_open": 4,
    "Memory_Tr35_60y": 4,
    "FluidIQ_Tr35_60y": 4,
    "CrystIQ_Tr35_60y": 4,
    "neo_o": 3,
    "neo_a": 3,
    "neo_c": 3,
}

# Phenotypic/Cognitive targets
#
# Demographics:
# - sex: Binary gender (M/F)
# - age_open: Age in years
#
# Cognitive Composites:
# - Memory_Tr35_60y: Memory composite score
# - FluidIQ_Tr35_60y: Fluid intelligence composite
# - CrystIQ_Tr35_60y: Crystallized intelligence composite
#
# Personality (NEO-FFI Big Five):
# - neo_n: Neuroticism
# - neo_e: Extraversion
# - neo_o: Openness
# - neo_a: Agreeableness
# - neo_c: Conscientiousness

TARGETS = [
    # Demographics
    "sex",
    "age_open",
    # Cognitive composites
    "Memory_Tr35_60y",
    "FluidIQ_Tr35_60y",
    "CrystIQ_Tr35_60y",
    # NEO-FFI Personality
    "neo_n",
    "neo_e",
    "neo_o",
    "neo_a",
    "neo_c",
]


def get_rest_subjects(aabc_root: Path) -> set[str]:
    subjects = set()
    for path in aabc_root.rglob("rfMRI_REST_*_tclean*"):
        if not path.is_file():
            continue
        subject_visit_dir = path.parents[3].name
        parts = subject_visit_dir.split("_")
        if parts:
            subjects.add(parts[0])
    return subjects


def quantize(series: pd.Series, num_bins: int):
    values = series.values

    qs = np.arange(1, num_bins) / num_bins
    bins = np.nanquantile(values, qs)
    bins = np.round(bins, 3).tolist()

    # right=True produces more balanced splits, and is consistent with pandas qcut
    targets = np.digitize(values, bins, right=True)
    targets = pd.Series(targets, index=series.index)
    counts = np.bincount(targets, minlength=num_bins)
    return targets, bins, counts


def quantize_with_balance(series: pd.Series):
    targets, bins, counts = quantize(series, num_bins=PRIMARY_BINS)
    total = counts.sum()
    if total == 0 or (counts / total).min() < MIN_BIN_FRACTION:
        targets, bins, counts = quantize(series, num_bins=FALLBACK_BINS)
        num_bins = FALLBACK_BINS
    else:
        num_bins = PRIMARY_BINS
    return targets, bins, counts, num_bins


def build_bin_stats(values: pd.Series, labels: pd.Series, num_bins: int):
    stats = []
    total = int(values.shape[0])
    for bin_idx in range(num_bins):
        bin_vals = values[labels == bin_idx]
        count = int(bin_vals.shape[0])
        stats.append(
            {
                "bin": bin_idx,
                "count": count,
                "fraction": round(count / total, 4) if total else 0.0,
                "min": float(bin_vals.min()) if count else None,
                "max": float(bin_vals.max()) if count else None,
            }
        )
    return stats


def main():
    # Load AABC phenotypic data
    df = pd.read_csv(AABC_CSV_PATH, usecols=["id_event"] + TARGETS)

    # Convert all columns except 'sex' to numeric, coercing errors to NaN
    for col in TARGETS:
        if col != "sex":
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Extract subject ID from id_event (HCA6000030_V1 -> HCA6000030)
    df["sub"] = df["id_event"].str.rsplit("_", n=1).str[0]

    # For subjects with multiple visits, use data from first visit
    # (longitudinal changes are minimal for most phenotypes)
    df = df.drop_duplicates(subset="sub", keep="first")
    df = df.set_index("sub")

    rest_subjects = get_rest_subjects(AABC_ROOT)
    if not rest_subjects:
        _logger.warning("No REST subjects found under %s; targets will be empty.", AABC_ROOT)
    df = df.loc[df.index.intersection(rest_subjects)]
    _logger.info("REST subjects in CSV: %d", len(df))

    outdir = ROOT / "metadata/targets"
    outdir.mkdir(exist_ok=True, parents=True)

    for target in TARGETS:
        outpath = outdir / f"aabc_target_map_{target}.json"
        infopath = outdir / f"aabc_target_info_{target}.json"
        series = df[target]
        na_mask = series.isna()
        series = series.loc[~na_mask]
        na_count = int(na_mask.sum())

        if target == "sex":
            targets = series.map(GENDER_MAP)
            counts = targets.value_counts().sort_index().tolist()
            info = {
                "target": target,
                "na_count": na_count,
                "subjects_total": int(df.shape[0]),
                "classes": GENDER_CLASSES,
                "label_counts": counts,
            }
        else:
            numeric = series.astype(float)
            override_bins = BIN_OVERRIDES.get(target)
            if override_bins is not None:
                targets, bins, counts = quantize(numeric, num_bins=override_bins)
                num_bins = override_bins
            else:
                targets, bins, counts, num_bins = quantize_with_balance(numeric)
            bin_stats = build_bin_stats(numeric, targets, num_bins)
            info = {
                "target": target,
                "na_count": na_count,
                "subjects_total": int(df.shape[0]),
                "bins": bins,
                "label_counts": counts.tolist(),
                "num_bins": num_bins,
                "bin_stats": bin_stats,
            }

        targets = targets.to_dict()
        _logger.info(json.dumps(info))

        with outpath.open("w") as f:
            print(json.dumps(targets, indent=4), file=f)

        with infopath.open("w") as f:
            print(json.dumps(info, indent=4), file=f)

    _logger.info(f"Created {len(TARGETS)} target files in {outdir}")


if __name__ == "__main__":
    main()
