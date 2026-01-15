import os
from pathlib import Path
from typing import Any, Callable

import datasets as hfds
import numpy as np
import pandas as pd
import torch
from cloudpathlib import AnyPath
from filelock import FileLock
from datasets.config import HF_DATASETS_CACHE

hfds.disable_progress_bars()

# parallelize download
HF_NUM_PROC = min(int(os.getenv("OMP_NUM_THREADS", "8")), 8)
HF_DOWNLOAD_CONFIG = hfds.DownloadConfig(num_proc=HF_NUM_PROC)


class HFDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        dataset: hfds.Dataset,
        target_key: str | None = None,
        target_map: str | Path | None = None,
        targets: np.ndarray | None = None,
        transform: Callable[[dict[str, Any]], dict[str, Any]] | None = None,
    ):
        self.dataset = dataset
        self.target_key = target_key

        self.dataset.set_format("torch")

        if target_map is not None:
            keys = self.dataset[target_key]
            indices = np.array(
                [
                    ii
                    for ii, key in enumerate(keys)
                    if key in target_map and not pd.isna(target_map[key])
                ]
            )
            targets = np.array([target_map[keys[idx]] for idx in indices])
        elif target_key is not None:
            targets = self.dataset[target_key]
            indices = np.array([ii for ii, target in enumerate(targets) if not pd.isna(target)])
            targets = np.asarray(targets[indices])
        else:
            indices = np.arange(len(dataset))

        if targets is not None:
            assert len(targets) == len(indices), "invalid targets"
            labels, target_ids, label_counts = np.unique(
                targets, return_inverse=True, return_counts=True
            )
        else:
            labels = label_counts = target_ids = None

        self.indices = indices
        self.labels = labels
        self.label_counts = label_counts
        self.targets = targets
        self.target_ids = target_ids
        self.num_classes = len(labels) if labels is not None else None

        self._transforms = [transform] if transform is not None else []

    def __getitem__(self, index: int):
        sample = self.dataset[self.indices[index]]
        if self.target_ids is not None:
            sample["target"] = self.target_ids[index]
        for transform in self._transforms:
            sample = transform(sample)
        return sample

    def reset_transform(self) -> "HFDataset":
        self._transforms = []
        return self

    def set_transform(self, transform: Callable[[dict[str, Any]], dict[str, Any]]) -> "HFDataset":
        self._transforms = [transform]
        return self

    def compose(self, transform: Callable[[dict[str, Any]], dict[str, Any]]) -> "HFDataset":
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
        s = f"HFDataset(\n{s}\n)"
        return s


def load_arrow_dataset(path: str | Path, cache_dir: str | Path | None = None, **kwargs):
    # try to give a more informative error message if a dataset doesn't exist
    # hf error message is pretty useless
    path = AnyPath(path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset {path} does not exist")

    cache_dir = Path(cache_dir or HF_DATASETS_CACHE)
    cache_dir.mkdir(exist_ok=True, parents=True)

    # try handle race condition when multiple jobs attempt to download the same dataset
    # TODO: think more about this
    dataset_name = str(path.relative_to(path.parents[1])).replace("/", "__")
    with FileLock(f"{cache_dir}/.{dataset_name}.lock"):
        dataset = hfds.load_dataset(
            "arrow",
            data_files=f"{path}/*.arrow",
            split="train",
            download_config=HF_DOWNLOAD_CONFIG,
            cache_dir=cache_dir,
            **kwargs,
        )
    return dataset
