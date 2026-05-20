from brainmarks.datasets.base import HFDataset, get_dataset_root, load_arrow_dataset
from brainmarks.datasets.registry import register_dataset

PPMI_ROOT = get_dataset_root("ppmi")


def _create_ppmi(space: str, target: str, **kwargs):
    dataset_dict = {}
    splits = ["train", "validation", "test"]
    for split in splits:
        url = f"{PPMI_ROOT}/ppmi.{space}.arrow/{split}"
        dataset = load_arrow_dataset(url, **kwargs)
        dataset = HFDataset(dataset, target_key=target)
        dataset_dict[split] = dataset
    return dataset_dict


@register_dataset
def ppmi_dx(space: str, **kwargs):
    return _create_ppmi(space, target="dx", **kwargs)


@register_dataset
def ppmi_age(space: str, **kwargs):
    return _create_ppmi(space, target="age_bin", **kwargs)


@register_dataset
def ppmi_sex(space: str, **kwargs):
    return _create_ppmi(space, target="sex", **kwargs)
