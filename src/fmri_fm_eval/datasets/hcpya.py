import os


from fmri_fm_eval.datasets.base import HFDataset, load_arrow_dataset
from fmri_fm_eval.datasets.registry import register_dataset

HCPYA_ROOT = os.getenv("HCPYA_ROOT", "s3://medarc/fmri-datasets/eval")

HCPYA_TASK21_TRIAL_TYPES = {
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


def _create_hcpya(
    name: str,
    space: str,
    target_key: str | None = None,
    **kwargs,
):
    dataset_dict = {}
    splits = ["train", "validation", "test"]
    for split in splits:
        url = f"{HCPYA_ROOT}/hcpya-{name}.{space}.arrow/{split}"
        dataset = load_arrow_dataset(url)
        dataset = HFDataset(dataset, target_key=target_key)
        dataset_dict[split] = dataset

    return dataset_dict


@register_dataset
def hcpya_rest1lr_age(space: str, **kwargs):
    return _create_hcpya("rest1lr", space, target_key="age_q", **kwargs)


@register_dataset
def hcpya_rest1lr_gender(space: str, **kwargs):
    return _create_hcpya("rest1lr", space, target_key="gender", **kwargs)


@register_dataset
def hcpya_rest1lr_flanker(space: str, **kwargs):
    return _create_hcpya("rest1lr", space, target_key="flanker_unadj_q", **kwargs)


@register_dataset
def hcpya_rest1lr_neofacn(space: str, **kwargs):
    return _create_hcpya("rest1lr", space, target_key="neofac_n_q", **kwargs)


@register_dataset
def hcpya_rest1lr_pmat24(space: str, **kwargs):
    return _create_hcpya("rest1lr", space, target_key="pmat24_a_cr_q", **kwargs)


@register_dataset
def hcpya_task21(space: str, **kwargs):
    return _create_hcpya("task21", space, target_key="cond_id", **kwargs)
