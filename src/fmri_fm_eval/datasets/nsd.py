import os

import datasets as hfds

from fmri_fm_eval.datasets.base import HFDataset, HF_DOWNLOAD_CONFIG
from fmri_fm_eval.datasets.registry import register_dataset

NSD_ROOT = os.getenv("NSD_ROOT", "s3://medarc/fmri-datasets/eval")


@register_dataset
def nsd_cococlip(space: str, **kwargs):
    dataset_dict = {}
    splits = ["train", "validation", "test", "testid"]
    for split in splits:
        url = f"{NSD_ROOT}/nsd-cococlip.{space}.arrow/{split}"
        dataset = hfds.load_dataset(
            "arrow",
            data_files=f"{url}/*.arrow",
            split="train",
            download_config=HF_DOWNLOAD_CONFIG,
            **kwargs,
        )
        dataset = HFDataset(dataset, target_key="category_id")
        dataset_dict[split] = dataset

    return dataset_dict
