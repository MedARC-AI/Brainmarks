import json
import os

import datasets as hfds
import fsspec

from fmri_fm_eval.datasets.base import HFDataset, load_arrow_dataset
from fmri_fm_eval.datasets.registry import register_dataset

ADNI_ROOT = os.getenv("ADNI_ROOT", "s3://medarc/fmri-datasets/eval")

ADNI_TARGET_MAP_DICT = {
    # Demographics
    "sex": "adni_target_map_sex.json",
    "age": "adni_target_map_age.json",
    # Cognitive scores
    "mmse": "adni_target_map_mmse.json",
    "cdrsb": "adni_target_map_cdrsb.json",
    # Clinical classification
    "diagnosis": "adni_target_map_diagnosis.json",
    "ad_vs_cn": "adni_target_map_ad_vs_cn.json",
    "pmci_vs_smci": "adni_target_map_pmci_vs_smci.json",
}


def _create_adni(space: str, target: str, **kwargs):
    """Create ADNI dataset with target labels.

    Args:
        space: Anatomical space (e.g., "schaefer400", "flat")
        target: Target variable name
        **kwargs: Additional arguments for dataset loading

    Returns:
        Dictionary of train/validation/test datasets
    """
    # Load target map
    target_map_path = ADNI_TARGET_MAP_DICT[target]
    target_map_path = f"{ADNI_ROOT}/targets/{target_map_path}"

    with fsspec.open(target_map_path, "r") as f:
        target_map = json.load(f)

    dataset_dict = {}
    splits = ["train", "validation", "test"]
    for split in splits:
        url = f"{ADNI_ROOT}/adni.{space}.arrow/{split}"
        dataset = load_arrow_dataset(url, **kwargs)

        # For ADNI, we need custom target key mapping since targets are keyed by PTID_SCANDATE
        # We'll use a custom wrapper that builds the key from sub and visit
        dataset = ADNIDataset(dataset, target_map=target_map)
        dataset_dict[split] = dataset

    return dataset_dict


def build_sample_key(sub: str, visit: str) -> str:
    """Build PTID_SCANDATE key from sub and visit fields.

    Args:
        sub: Subject ID (e.g., "168S6049")
        visit: Session/visit ID (e.g., "20201012")

    Returns:
        Sample key in format "PTID_SCANDATE" (e.g., "168_S_6049_2020-10-12")
    """
    # Reconstruct PTID from sub
    # sub format: "168S6049" -> PTID format: "168_S_6049"
    # Find the 'S' position and insert underscores
    s_pos = sub.find("S")
    if s_pos > 0:
        ptid = f"{sub[:s_pos]}_S_{sub[s_pos + 1 :]}"
    else:
        ptid = sub  # Fallback

    # Format visit as date: "20201012" -> "2020-10-12"
    if len(visit) == 8:
        scandate = f"{visit[:4]}-{visit[4:6]}-{visit[6:]}"
    else:
        scandate = visit  # Fallback

    return f"{ptid}_{scandate}"


# TODO: fix this duplication


class ADNIDataset(HFDataset):
    """ADNI dataset with custom sample key construction.

    ADNI targets are keyed by PTID_SCANDATE (e.g., "168_S_6049_2020-10-12"),
    but the dataset contains sub and visit separately. This class builds the
    target key from sub and visit.
    """

    def __init__(self, dataset: hfds.Dataset, target_map: dict):
        # Build target keys for all samples
        import numpy as np
        import pandas as pd

        self.dataset = dataset
        self.target_map = target_map
        self.dataset.set_format("torch")

        # Build sample keys from sub and visit
        subs = dataset["sub"]
        visits = dataset["visit"]
        sample_keys = [build_sample_key(sub, visit) for sub, visit in zip(subs, visits)]

        # Filter to samples with valid targets
        indices = np.array(
            [
                ii
                for ii, key in enumerate(sample_keys)
                if key in target_map and not pd.isna(target_map[key])
            ]
        )
        targets = np.array([target_map[sample_keys[idx]] for idx in indices])

        # Compute label statistics
        labels, target_ids, label_counts = np.unique(
            targets, return_inverse=True, return_counts=True
        )

        self.indices = indices
        self.labels = labels
        self.label_counts = label_counts
        self.targets = targets
        self.target_ids = target_ids
        self.num_classes = len(labels)
        self._transforms = []

    def __getitem__(self, index: int):
        sample = self.dataset[self.indices[index]]
        sample["target"] = self.target_ids[index]
        for transform in self._transforms:
            sample = transform(sample)
        return sample

    def reset_transform(self) -> "ADNIDataset":
        self._transforms = []
        return self

    def set_transform(self, transform) -> "ADNIDataset":
        self._transforms = [transform]
        return self

    def compose(self, transform) -> "ADNIDataset":
        self._transforms.append(transform)
        return self

    def __len__(self):
        return len(self.indices)

    def __repr__(self):
        s = (
            f"    dataset={self.dataset},\n"
            f"    labels={self.labels},\n"
            f"    counts={self.label_counts}"
        )
        return f"ADNIDataset(\n{s}\n)"


# Demographics
@register_dataset
def adni_sex(space: str, **kwargs):
    """ADNI sex (binary: Female=0, Male=1)."""
    return _create_adni(space, target="sex", **kwargs)


@register_dataset
def adni_age(space: str, **kwargs):
    """ADNI age (binned into 3 quantile bins)."""
    return _create_adni(space, target="age", **kwargs)


# Cognitive scores
@register_dataset
def adni_mmse(space: str, **kwargs):
    """ADNI Mini-Mental State Exam score (binned: 0-23=Dementia, 24-27=MCI, 28-30=Normal)."""
    return _create_adni(space, target="mmse", **kwargs)


@register_dataset
def adni_cdrsb(space: str, **kwargs):
    """ADNI Clinical Dementia Rating Sum of Boxes (binned: 0-0.5=Normal, 1-4.5=Mild, 5+=Moderate/Severe)."""
    return _create_adni(space, target="cdrsb", **kwargs)


# Clinical classification
@register_dataset
def adni_diagnosis(space: str, **kwargs):
    """ADNI diagnosis (3-class: CN=0, MCI=1, Dementia=2)."""
    return _create_adni(space, target="diagnosis", **kwargs)


@register_dataset
def adni_ad_vs_cn(space: str, **kwargs):
    """ADNI AD vs CN (binary: CN=0, Dementia=1, excludes MCI)."""
    return _create_adni(space, target="ad_vs_cn", **kwargs)


@register_dataset
def adni_pmci_vs_smci(space: str, **kwargs):
    """ADNI progressive vs stable MCI (binary: sMCI=0, pMCI=1, MCI subset only)."""
    return _create_adni(space, target="pmci_vs_smci", **kwargs)
