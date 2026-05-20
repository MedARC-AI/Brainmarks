from brainmarks.datasets.base import HFDataset, get_dataset_root, load_arrow_dataset
from brainmarks.datasets.registry import register_dataset

NSD_ROOT = get_dataset_root("nsd")


@register_dataset
def nsd_cococlip(space: str, **kwargs):
    dataset_dict = {}
    splits = ["train", "validation", "test", "testid"]
    for split in splits:
        url = f"{NSD_ROOT}/nsd-cococlip.{space}.arrow/{split}"
        dataset = load_arrow_dataset(url, **kwargs)
        dataset = HFDataset(dataset, target_key="category_id")
        dataset_dict[split] = dataset

    return dataset_dict


@register_dataset
def nsd_cococlip_subj01(space: str, **kwargs):
    dataset_dict = {}
    splits = {"train": "train", "testid": "validation", "shared1000": "test"}

    for split, name in splits.items():
        url = f"{NSD_ROOT}/nsd-cococlip.{space}.arrow/{split}"
        dataset = load_arrow_dataset(url, **kwargs)
        dataset = dataset.filter(lambda sub: sub == "subj01", input_columns="sub")
        dataset = HFDataset(dataset, target_key="category_id")
        dataset_dict[name] = dataset

    return dataset_dict
