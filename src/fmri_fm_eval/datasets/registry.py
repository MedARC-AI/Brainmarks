import importlib
import logging
import pkgutil
from typing import Callable

import fmri_fm_eval.datasets
from torch.utils.data import Dataset

_logger = logging.getLogger(__package__)

DatasetDict = dict[str, Dataset]
DatasetFn = Callable[..., DatasetDict]

_DATASET_REGISTRY: dict[str, DatasetFn] = {}


def register_dataset(name_or_func: str | DatasetFn | None = None):
    def _decorator(func: Callable):
        name = name_or_func if isinstance(name_or_func, str) else func.__name__
        if name in _DATASET_REGISTRY:
            _logger.warning(f"Dataset {name} already registered; overwriting.")
        _DATASET_REGISTRY[name] = func
        return func

    if isinstance(name_or_func, Callable):
        return _decorator(name_or_func)
    return _decorator


def create_dataset(name: str, space: str, **kwargs) -> DatasetDict:
    if name not in _DATASET_REGISTRY:
        raise ValueError(f"Dataset {name} not registered")
    dataset_dict = _DATASET_REGISTRY[name](space=space, **kwargs)
    return dataset_dict


def list_datasets() -> list[str]:
    return list(_DATASET_REGISTRY)


def import_dataset_plugins():
    """Finds and imports all plugins registering new models."""
    # https://packaging.python.org/en/latest/guides/creating-and-discovering-plugins/#using-namespace-packages
    plugins = {}
    for finder, name, ispkg in pkgutil.iter_modules(fmri_fm_eval.datasets.__path__):
        if not (name in {"base", "registry", "template"} or name.startswith("test_")):
            try:
                plugins[name] = importlib.import_module(f"fmri_fm_eval.datasets.{name}")
            except Exception as exc:
                _logger.warning(f"Import dataset plugin {name} failed: {exc}")
    return plugins


# import all discovered plugins to register
_DATASET_PLUGINS = import_dataset_plugins()
