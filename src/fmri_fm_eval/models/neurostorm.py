"""

NeuroSTORM: Towards a general-purpose foundation model for fMRI analysis

"""

import torch.nn as nn
from torch import Tensor
import torch.nn.functional as F

from fmri_fm_eval.models.base import Embeddings
from fmri_fm_eval.models.registry import register_model
from pathlib import Path
import numpy as np
import torch
from fmri_fm_eval import nisc
from einops import rearrange
import templateflow.api as tflow
import types
from types import SimpleNamespace
from huggingface_hub import hf_hub_download
import shutil


try:
    from neurostorm.models.lightning_model import LightningModel
    from neurostorm.datasets.preprocessing_volume import select_middle_96, spatial_resampling, temporal_resampling # original functions from: https://github.com/CUHK-AIM-Group/NeuroSTORM/blob/5bb4f7c844ed7544f95cd934eece69b390a55ea4/datasets/preprocessing_volume.py#L12C1-L69C26
    from neurostorm.datasets.fmri_datasets import pad_to_96, resize_volume

except ImportError as exc:
    raise ImportError(
        "neurostorm not installed. Please install the optional neurostorm extra with `uv sync --extra neurostorm`"
    ) from exc


# Cache directory for downloaded files
NEUROSTORM_CACHE_DIR = Path.home() / ".cache" / "fmri-fm-eval" / "neurostorm"


NEUROSTORM_VARIANTS = {
    "0.8": "fmrifound/pt_fmrifound_mae_ratio0.8.ckpt",
    "0.5": "fmrifound/pt_fmrifound_mae_ratio0.5.ckpt",
}

def fetch_neurostorm_checkpoint(variant: str) -> Path:

    repo_id = "zxcvb20001/NeuroSTORM"
    filename = NEUROSTORM_VARIANTS[variant]

    cached_path = hf_hub_download(repo_id=repo_id, filename=filename)

    # 2) copy into ./ with no containing folder
    dst = NEUROSTORM_CACHE_DIR / filename
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(cached_path, dst)

    return dst


# Dummy datamodule to initialize LitClassifier
class _DummyTrainDataset:
    target_values = np.zeros((32, 1), dtype=np.float32)


class _DummyDataModule:
    def __init__(self):
        self.train_dataset = _DummyTrainDataset()


class NeuroStormWrapper(nn.Module):
    __space__: str = "mni"

    def __init__(self, variant: str) -> None:
        super().__init__()

        self.ckpt_path = fetch_neurostorm_checkpoint(variant)

        dm = _DummyDataModule()

        ckpt = torch.load(self.ckpt_path, map_location="cpu")

        # patch hyperparameters
        hparams = ckpt["hyper_parameters"]
        hparams["print_flops"] = False # missing required key
        hparams["model"] = "neurostorm" # overwrite model name, current checkpoint has "swin4d_mae" which is not a valid model name here: https://github.com/CUHK-AIM-Group/NeuroSTORM/blob/main/models/load_model.py but the keys match the 'neurostorm' model
        model = LightningModel(**hparams, data_module=_DummyDataModule())

        # load weights
        state_dict = ckpt["state_dict"]
        model.load_state_dict(state_dict, strict=True)

        self.backbone = model.model
        self.monkey_patch_forward_encoder()
        self.expected_seq_len = 20

    def monkey_patch_forward_encoder(self):

        def forward_encoder(self, x, apply_mask: bool = True): # patch method since original code always applies mask: https://github.com/CUHK-AIM-Group/NeuroSTORM/blob/5bb4f7c844ed7544f95cd934eece69b390a55ea4/models/neurostorm.py#L1191C1-L1205C1
            x = self.patch_embed(x)
            mask = None

            if apply_mask:
                x, mask = self.random_masking(x)

            for i in range(self.num_layers):
                x = self.pos_embeds[i](x)
                x = self.layers[i](x.contiguous())

            return x, mask

        self.backbone.forward_encoder = types.MethodType(forward_encoder, self.backbone)

    def forward(self, batch: dict[str, Tensor]) -> Embeddings:
        x = batch["bold"]
        B, C, H, W, D, T = x.shape

        # handle sliding windows
        num_windows = T // self.expected_seq_len
        T = num_windows * self.expected_seq_len
        x = rearrange(x[..., :T], "b c x y z (w t) -> (b w) c x y z t", w=num_windows)

        feats, mask = self.backbone.forward_encoder(x, apply_mask=False)  # feats have shape (B, channels, H, W, D, T) (B, 288, 2, 2, 2, 20)

        if feats.isnan().sum() > 0:
            print("NaNs in feats")

        feats = rearrange(
            feats, "(b w) c x y z t -> b (w x y z t) c", w=num_windows
        )  # convert to (B, patches, channels)

        return Embeddings(
            cls_embeds=None,
            reg_embeds=None,
            patch_embeds=feats,
        )

class NeuroStormTransform:
    """
    0. unnormalize voxelwize z-scored data
    1. convert to 4D volume
    2. spatial resampling (to 2mm), temporal resampling to (0.8) and select middle 96 voxels
    3. build mask as every zero value and drop every negative value
    4. global normalization
    5. fill background with global min value
    6. pad/crop to expected sequence length t=20
    7. pad to (96, 96, 96) for axes that are smaller
    8. reshape to expected shape (C, H, W, D, T)
    """

    def __init__(self):
        # Mask calculation from fmri_fm_eval.readers
        roi_path = tflow.get(
            "MNI152NLin6Asym", desc="brain", resolution=2, suffix="mask", extension="nii.gz"
        )
        mask = nisc.read_mni152_2mm_data(roi_path) > 0  # (Z, Y, X)

        self.mask = torch.from_numpy(mask)
        self.mask_shape = mask.shape

        # NeuroSTORM input size is (H, W, D, T) = (X, Y, Z, T) = (96, 96, 96, 20), https://github.com/CUHK-AIM-Group/NeuroSTORM/blob/49dd063e48a635d66653e3b02e752256f6813621/README.md?plain=1#L297
        self.expected_seq_len = 20
        self.spatial_target = 96

        # target temporal resampling: https://github.com/CUHK-AIM-Group/NeuroSTORM/blob/5bb4f7c844ed7544f95cd934eece69b390a55ea4/datasets/preprocessing_volume.py#L54C1-L69C26
        self.target_tr = 0.8

    def __call__(self, sample: dict[str, Tensor]) -> dict[str, Tensor]:
        """
        Transform bold volumes to model input format.

        sample dicts requires keys:
            - bold: (T,V) normalized bold signal,
            - mean: (1,V) mean of bold signal,
            - std: (1,V) standard deviation of bold signal

        sample dict is modified in place:
            - bold: (C, H, W, D, T)

        """
        # unnormalize
        bold = sample["bold"] * sample["std"] + sample["mean"]

        Z, Y, X = self.mask.shape
        tr = sample["tr"]

        # unflatten bold to (X, Y, Z, T)
        T, V = bold.shape
        mask = self.mask.to(device=bold.device)
        vol = torch.full((T, Z, Y, X), 0, device=bold.device, dtype=bold.dtype)
        vol[:, mask] = bold
        vol = rearrange(vol, "t z y x -> x y z t")

        header_handler = SimpleNamespace(get_zooms=lambda: (2.0, 2.0, 2.0, tr))
        vol = np.array(vol) # the functions expect numpy arrays
        vol = spatial_resampling(vol, header_handler) # is all the data already on 2mm, if so this line is a not needed, if not we'll need to pass that info to the samples
        vol = temporal_resampling(vol, header_handler)
        vol = select_middle_96(vol)
        vol = torch.from_numpy(vol) # convert back to torch

        background = vol == 0 # get background https://github.com/CUHK-AIM-Group/NeuroSTORM/blob/5bb4f7c844ed7544f95cd934eece69b390a55ea4/datasets/preprocessing_volume.py#L94
        vol[vol<0] = 0 # every negative value is set to 0 https://github.com/CUHK-AIM-Group/NeuroSTORM/blob/5bb4f7c844ed7544f95cd934eece69b390a55ea4/datasets/preprocessing_volume.py#L119

        # global z-score https://github.com/CUHK-AIM-Group/NeuroSTORM/blob/5bb4f7c844ed7544f95cd934eece69b390a55ea4/datasets/preprocessing_volume.py#L123C1-L126C54
        vol = (vol - vol[~background].mean()) / vol[~background].std()

        vol_g = torch.empty(vol.shape)
        vol_g[background] = vol[~background].min().item() #https://github.com/CUHK-AIM-Group/NeuroSTORM/blob/5bb4f7c844ed7544f95cd934eece69b390a55ea4/datasets/preprocessing_volume.py#L130C5-L132C54
        vol_g[~background] = vol[~background]

        T = vol.shape[3] # get T after resampling
        # Pad if too short - repeat mean (consistent with other models)
        if T < self.expected_seq_len:
            mean = vol.mean(dim=3, keepdim=True).repeat(1, 1, 1, self.expected_seq_len - T)
            vol = torch.cat([vol, mean], dim=3)

        # Crop to fixed number of non-overlapping windows
        num_windows = vol.shape[3] // self.expected_seq_len
        T_cropped = num_windows * self.expected_seq_len
        vol = vol[..., :T_cropped]

        volume = rearrange(vol, "x y z t -> 1 x y z t") # include the channel dimension
        volume = pad_to_96(volume) # pad to 96x96x96 https://github.com/CUHK-AIM-Group/NeuroSTORM/blob/5bb4f7c844ed7544f95cd934eece69b390a55ea4/datasets/fmri_datasets.py#L12C1-L21C13
        # volume = resize_volume(volume, (self.spatial_target, self.spatial_target, self.spatial_target, self.expected_seq_len)) # this is not needed I think, we're already on the dimension required: https://github.com/CUHK-AIM-Group/NeuroSTORM/blob/5bb4f7c844ed7544f95cd934eece69b390a55ea4/datasets/fmri_datasets.py#L26C1-L56C17
        # with_voxel_norm is False on the checkpoint: https://github.com/CUHK-AIM-Group/NeuroSTORM/blob/5bb4f7c844ed7544f95cd934eece69b390a55ea4/datasets/fmri_datasets.py#L117

        if volume.isnan().sum() > 0:
            print("NaNs in volume")

        sample["bold"] = volume
        return sample


@register_model
def neurostorm_mae_0p5() -> tuple[NeuroStormTransform, NeuroStormWrapper]:
    return NeuroStormTransform(), NeuroStormWrapper(variant="0.5")

@register_model
def neurostorm_mae_0p8() -> tuple[NeuroStormTransform, NeuroStormWrapper]:
    return NeuroStormTransform(), NeuroStormWrapper(variant="0.8")

