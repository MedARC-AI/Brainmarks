import os


from fmri_fm_eval.datasets.base import HFDataset, load_arrow_dataset
from fmri_fm_eval.datasets.registry import register_dataset

ADHD200_ROOT = os.getenv("ADHD200_ROOT", "s3://medarc/fmri-datasets/eval")


def _create_adhd200(space: str, target: str, **kwargs):
    dataset_dict = {}
    splits = ["train", "validation", "test"]
    for split in splits:
        url = f"{ADHD200_ROOT}/adhd200.{space}.arrow/{split}"
        dataset = load_arrow_dataset(url, **kwargs)
        dataset = HFDataset(dataset, target_key=target)
        dataset_dict[split] = dataset
    return dataset_dict


@register_dataset
def adhd200_dx(space: str, **kwargs):
    return _create_adhd200(space, target="dx", **kwargs)


@register_dataset
def adhd200_sex(space: str, **kwargs):
    return _create_adhd200(space, target="gender", **kwargs)
