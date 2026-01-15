import json
import os

import fsspec

from fmri_fm_eval.datasets.base import HFDataset, load_arrow_dataset
from fmri_fm_eval.datasets.registry import register_dataset

ABIDE_ROOT = os.getenv("ABIDE_ROOT", "s3://medarc/fmri-datasets/eval")

ABIDE_TARGET_MAP_DICT = {
    "dx": "abide_target_map_dx.json",
    "age": "abide_target_map_age_bin.json",
    "sex": "abide_target_map_sex.json",
}


def _create_abide(space: str, target: str, **kwargs):
    target_key = "sub"
    target_map_path = ABIDE_TARGET_MAP_DICT[target]
    target_map_path = f"{ABIDE_ROOT}/targets/{target_map_path}"

    with fsspec.open(target_map_path, "r") as f:
        target_map = json.load(f)

    dataset_dict = {}
    splits = ["train", "validation", "test"]
    for split in splits:
        url = f"{ABIDE_ROOT}/abide.{space}.arrow/{split}"
        dataset = load_arrow_dataset(url, **kwargs)
        dataset = HFDataset(dataset, target_key=target_key, target_map=target_map)
        dataset_dict[split] = dataset

    return dataset_dict


@register_dataset
def abide_dx(space: str, **kwargs):
    return _create_abide(space, target="dx", **kwargs)


@register_dataset
def abide_age(space: str, **kwargs):
    return _create_abide(space, target="age", **kwargs)


@register_dataset
def abide_sex(space: str, **kwargs):
    return _create_abide(space, target="sex", **kwargs)
