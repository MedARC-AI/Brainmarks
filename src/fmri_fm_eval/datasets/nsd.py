import os


from fmri_fm_eval.datasets.base import HFDataset, load_arrow_dataset
from fmri_fm_eval.datasets.registry import register_dataset

NSD_ROOT = os.getenv("NSD_ROOT", "s3://medarc/fmri-datasets/eval")


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
