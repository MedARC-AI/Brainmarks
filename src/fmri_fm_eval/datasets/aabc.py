import os
import json

import fsspec

from fmri_fm_eval.datasets.base import HFDataset, load_arrow_dataset
from fmri_fm_eval.datasets.registry import register_dataset

AABC_ROOT = os.getenv("AABC_ROOT", "s3://medarc/fmri-datasets/eval")

AABC_TARGET_MAP_DICT = {
    # Demographics
    "sex": "aabc_target_map_sex.json",
    "age": "aabc_target_map_age_open.json",
    # NEO-FFI Personality
    "neo_n": "aabc_target_map_neo_n.json",
    "neo_e": "aabc_target_map_neo_e.json",
    "neo_o": "aabc_target_map_neo_o.json",
    "neo_a": "aabc_target_map_neo_a.json",
    "neo_c": "aabc_target_map_neo_c.json",
    # Cognitive Composites
    "fluid_iq": "aabc_target_map_FluidIQ_Tr35_60y.json",
    "cryst_iq": "aabc_target_map_CrystIQ_Tr35_60y.json",
    "memory": "aabc_target_map_Memory_Tr35_60y.json",
}


def _create_aabc(space: str, target: str, **kwargs):
    target_key = "sub"
    target_map_path = AABC_TARGET_MAP_DICT[target]
    target_map_path = f"{AABC_ROOT}/targets/{target_map_path}"

    with fsspec.open(target_map_path, "r") as f:
        target_map = json.load(f)

    dataset_dict = {}
    splits = ["train", "validation", "test"]
    for split in splits:
        url = f"{AABC_ROOT}/aabc.{space}.arrow/{split}"
        dataset = load_arrow_dataset(url, **kwargs)
        dataset = HFDataset(dataset, target_key=target_key, target_map=target_map)
        dataset_dict[split] = dataset

    return dataset_dict


# Demographics
@register_dataset
def aabc_sex(space: str, **kwargs):
    return _create_aabc(space, target="sex", **kwargs)


@register_dataset
def aabc_age(space: str, **kwargs):
    return _create_aabc(space, target="age", **kwargs)


# NEO-FFI Personality
@register_dataset
def aabc_neo_n(space: str, **kwargs):
    return _create_aabc(space, target="neo_n", **kwargs)


@register_dataset
def aabc_neo_e(space: str, **kwargs):
    return _create_aabc(space, target="neo_e", **kwargs)


@register_dataset
def aabc_neo_o(space: str, **kwargs):
    return _create_aabc(space, target="neo_o", **kwargs)


@register_dataset
def aabc_neo_a(space: str, **kwargs):
    return _create_aabc(space, target="neo_a", **kwargs)


@register_dataset
def aabc_neo_c(space: str, **kwargs):
    return _create_aabc(space, target="neo_c", **kwargs)


# Cognitive Composites
@register_dataset
def aabc_fluid_iq(space: str, **kwargs):
    return _create_aabc(space, target="fluid_iq", **kwargs)


@register_dataset
def aabc_cryst_iq(space: str, **kwargs):
    return _create_aabc(space, target="cryst_iq", **kwargs)


@register_dataset
def aabc_memory(space: str, **kwargs):
    return _create_aabc(space, target="memory", **kwargs)
