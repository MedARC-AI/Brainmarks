"""

Omni-fMRI: A Universal Atlas-Free fMRI Foundation Model

https://github.com/OneMore1/Omni-fMRI

"""

from functools import partial
from pathlib import Path

import numpy as np
import templateflow.api as tflow
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from huggingface_hub import hf_hub_download
from torch import Tensor

from brainmarks import nisc
from brainmarks.models.base import Embeddings
from brainmarks.models.registry import register_model

try:
    from omnifmri.models.mae_model import AdaptiveMAE
    from omnifmri.models.patch_embed_3d import TokenizedZeroConvPatchAttn3D

except ImportError as exc:
    raise ImportError(
        "omnifmri not installed. Please install the optional omnifmri extra with `uv sync --extra omnifmri`"
    ) from exc


def fetch_omnifmri_checkpoint() -> Path:
    ckpt_path = hf_hub_download(
        repo_id="OneMore1/Omni-fMRI",
        filename="checkpoint.pth",
        revision="0bd3c34ba191e6ed9ac2a453b314f79178b7dfb2",
    )
    return Path(ckpt_path)


# The following model-construction utils are copied verbatim from Omni-fMRI
# extract_feat.py: https://github.com/OneMore1/Omni-fMRI/blob/main/extract_feat.py


def as_3tuple(value: object) -> tuple[int, int, int]:
    if isinstance(value, (list, tuple)):
        if len(value) != 3:
            raise ValueError(f"Expected a 3-item spatial tuple, got {value}")
        return tuple(int(x) for x in value)
    return (int(value), int(value), int(value))


def create_model(config: dict) -> nn.Module:
    model_config = config["model"]

    return AdaptiveMAE(
        img_size=as_3tuple(model_config["img_size"]),
        patch_size=as_3tuple(model_config["patch_size"]),
        in_chans=int(model_config["in_chans"]),
        embed_dim=int(model_config["embed_dim"]),
        depth=int(model_config["depth"]),
        qkv_bias=bool(model_config["qkv_bias"]),
        qk_norm=bool(model_config["qk_norm"]),
        num_heads=int(model_config["num_heads"]),
        decoder_embed_dim=int(model_config["decoder_embed_dim"]),
        drop_path_rate=float(model_config["drop_path_rate"]),
        decoder_depth=int(model_config["decoder_depth"]),
        decoder_num_heads=int(model_config["decoder_num_heads"]),
        mlp_ratio=float(model_config["mlp_ratio"]),
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        mask_ratio=float(model_config["mask_ratio"]),
        mixed_patch_embed=TokenizedZeroConvPatchAttn3D,
        patch_norm=bool(model_config["enable_patch_norm"]),
        gate_attention=model_config["gate_attention"],
    )


def state_dict_from_checkpoint(checkpoint: object) -> dict:
    if isinstance(checkpoint, dict):
        if "model_state_dict" in checkpoint:
            return checkpoint["model_state_dict"]
        if "state_dict" in checkpoint:
            return checkpoint["state_dict"]
    if not isinstance(checkpoint, dict):
        raise TypeError(f"Unsupported checkpoint type: {type(checkpoint)!r}")
    return checkpoint


def create_backbone(checkpoint: dict, config: dict, device: torch.device) -> nn.Module:
    model = create_model(config)
    state_dict = state_dict_from_checkpoint(checkpoint)
    load_msg = model.load_state_dict(state_dict, strict=False)

    unexpected = len(load_msg.unexpected_keys)
    missing = len(load_msg.missing_keys)
    print(
        f"Loaded checkpoint weights with {missing} missing keys and {unexpected} unexpected keys."
    )
    if unexpected:
        print("Unexpected keys are ignored by strict=False loading.")

    if hasattr(model, "encoder"):
        backbone = model.encoder
    elif hasattr(model, "backbone"):
        backbone = model.backbone
    else:
        backbone = model

    backbone.to(device)
    backbone.eval()
    return backbone


@torch.no_grad()
def extract_tokens(backbone: nn.Module, sample: Tensor) -> tuple[Tensor, list[Tensor]]:
    """
    Copy of Omni-fMRI extract_feat.py `extract_tokens` with small change to support a
    batch of samples.

    Returns:
        cls: (B, D) final-layer CLS token per item
        patches: list of length B of (N_i, D) final-layer patch tokens
    """
    input_dict = backbone.patch_tokenizer(sample)
    current_img_size = sample.shape[2:]

    tokens, _, _, _, _ = backbone.mixed_patch(
        sample,
        backbone.pos_embed,
        input_dict,
        current_img_size=current_img_size,
    )

    seqlens = torch.as_tensor(input_dict["seqlens"], device=sample.device, dtype=torch.long)
    arange = torch.arange(tokens.shape[1], device=sample.device).unsqueeze(0)
    valid_mask = arange < seqlens.unsqueeze(1)

    tokens_packed = tokens[valid_mask]
    cu_seqlens = torch.cat(
        [
            torch.zeros(1, device=sample.device, dtype=torch.int32),
            seqlens.cumsum(0, dtype=torch.int32),
        ]
    ).contiguous()
    max_seqlen = int(seqlens.max().item())

    layer_outputs = backbone.forward_features(
        tokens_packed,
        cu_seqlens=cu_seqlens,
        max_seqlen=max_seqlen,
    )
    last_tokens = layer_outputs[-1]

    # Unpack per-item sequences: [CLS, patch tokens...].
    splits = torch.split(last_tokens, seqlens.tolist(), dim=0)
    cls = torch.stack([seq[0] for seq in splits], dim=0)  # (M, D)
    patches = [seq[1:] for seq in splits]  # list of (N_i, D)
    return cls, patches


class OmniFmriWrapper(nn.Module):
    __space__: str = "mni"

    def __init__(self) -> None:
        super().__init__()

        ckpt_path = fetch_omnifmri_checkpoint()
        with torch.serialization.safe_globals(
            [np._core.multiarray.scalar, np.dtype, np.dtypes.Float64DType]
        ):
            ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)

        cfg = ckpt["config"]
        self.backbone = create_backbone(ckpt, cfg, torch.device("cpu"))

        # Model consumes a single 40-frame segment as `in_chans` (time-as-channels).
        self.expected_seq_len = 40
        self.max_windows = 8

    def forward(self, batch: dict[str, Tensor]) -> Embeddings:
        x = batch["bold"]
        B, T, X, Y, Z = x.shape

        # Split the temporal axis into non-overlapping windows, each fed to the model
        # as a separate `in_chans`-frame volume (time-as-channels).
        num_windows = min(T // self.expected_seq_len, self.max_windows)
        T = num_windows * self.expected_seq_len
        x = rearrange(x[:, :T], "b (w t) x y z -> (b w) t x y z", w=num_windows)

        cls, patches = extract_tokens(self.backbone, x)
        D = cls.shape[-1]

        # Average CLS token over windows -> (B, 1, D).
        cls_embeds = rearrange(cls, "(b w) d -> b w d", w=num_windows).mean(dim=1, keepdim=True)

        # Concatenate patch tokens across windows for each sample.
        per_sample = [
            torch.cat(patches[b * num_windows : (b + 1) * num_windows], dim=0) for b in range(B)
        ]

        # Dynamic patching yields a variable patch count per sample; pad to the batch max,
        # filling with each sample's mean token so mean-pooling stays unbiased.
        max_len = max(p.shape[0] for p in per_sample)
        patch_embeds = cls.new_zeros((B, max_len, D))
        for b, p in enumerate(per_sample):
            n = p.shape[0]
            patch_embeds[b, :n] = p
            if n < max_len:
                patch_embeds[b, n:] = p.mean(dim=0, keepdim=True)

        return Embeddings(cls_embeds=cls_embeds, reg_embeds=None, patch_embeds=patch_embeds)


class OmniFmriTransform:
    """
    0. Unnormalize voxelwise z-scored data
    1. temporal resampling to a uniform 0.72s TR
    2. global z-score normalization over brain voxels
    3. pad/crop to whole number of 40-frame windows
    4. unmask input to full 4D MNI volume (background 0)
    5. flip to LAS orientation
    6. symmetric spatial crop/pad to (96, 96, 96)
    7. reshape to (T, X, Y, Z)

    https://github.com/OneMore1/Omni-fMRI/blob/main/data_preparation/preprocessing.py
    """

    def __init__(self, coord_normalize: bool = False):
        self.coord_normalize = coord_normalize

        # Mask calculation from brainmarks.readers (same MNI152 2mm space as NeuroSTORM).
        roi_path = tflow.get(
            "MNI152NLin6Asym", desc="brain", resolution=2, suffix="mask", extension="nii.gz"
        )
        mask = nisc.read_mni152_2mm_data(roi_path) > 0  # (Z, Y, X)

        self.mask = torch.from_numpy(mask)
        self.mask_shape = mask.shape

        # Omni-fMRI input size is (in_chans, X, Y, Z) = (40, 96, 96, 96), time-as-channels.
        self.expected_seq_len = 40
        self.spatial_target = 96
        self.max_windows = 8

        # target temporal resampling. The paper resamples datasets with differing
        # temporal resolutions to a uniform 0.72s TR (via cubic B-spline; we use linear
        # interpolation for efficiency / consistency with the other models).
        self.target_tr = 0.72

    def __call__(self, sample: dict[str, Tensor]) -> dict[str, Tensor]:
        """
        Transform bold volumes to model input format.

        sample dicts requires keys:
            - bold: (T,V) normalized bold signal,
            - mean: (1,V) mean of bold signal,
            - std: (1,V) standard deviation of bold signal

        sample dict is modified in place:
            - bold: (T, X, Y, Z)

        """
        # unnormalize
        if not self.coord_normalize:
            bold = sample["bold"] * sample["std"] + sample["mean"]
        else:
            bold = sample["bold"]
        tr = float(sample["tr"])

        # temporal resampling to a uniform 0.72s TR.
        # nb, we resample while in sparse (T, V) format for efficiency.
        if abs(tr - self.target_tr) >= 0.1:
            bold = resample_to_target_tr(bold, tr, self.target_tr)

        # global normalization over the whole run.
        # Omni z-scores non-zero (brain) voxels; here the sparse (T, V) form holds exactly
        # the brain voxels, so global mean/std over `bold` is equivalent.
        # https://github.com/OneMore1/Omni-fMRI/blob/main/data_preparation/preprocessing.py#L214
        bold = (bold - bold.mean()) / (bold.std() + 1e-8)

        # Pad if too short - repeat mean (consistent with other models)
        T = len(bold)
        if T < self.expected_seq_len:
            mean = bold.mean(dim=0).repeat(self.expected_seq_len - T, 1)
            bold = torch.cat([bold, mean], dim=0)
            T = self.expected_seq_len

        # Crop to a fixed number of non-overlapping windows
        num_windows = min(T // self.expected_seq_len, self.max_windows)
        T = num_windows * self.expected_seq_len
        bold = bold[:T, :]

        # unflatten to full volume; Omni z-scoring leaves the background at 0.
        T, V = bold.shape
        Z, Y, X = self.mask_shape
        mask = self.mask.to(device=bold.device)
        volume = torch.zeros((T, Z, Y, X), device=bold.device)
        volume[:, mask] = bold
        volume = rearrange(volume, "t z y x -> t x y z")

        # flip x axis. the provided MNI data are in RAS orientation, but the model was
        # trained on FSL MNI152 (LAS) data (cf. data_preparation/MNI152_T1_1mm_brain_mask.nii.gz).
        volume = torch.flip(volume, (1,))

        # symmetric center crop/pad to 96, following Omni normalize_spatial_dimensions
        # (left = total // 2): x 91->96 pad(2,3), y 109->96 crop(6,7), z 91->96 pad(2,3).
        # https://github.com/OneMore1/Omni-fMRI/blob/main/data_preparation/preprocessing.py#L126
        assert (X, Y, Z) == (91, 109, 91), "unexpected volume shape"
        # F.pad orders the trailing dims first: (z_l, z_r, y_l, y_r, x_l, x_r)
        volume = F.pad(volume, (2, 3, -6, -7, 2, 3), value=0.0)

        sample["bold"] = volume  # (T, X, Y, Z)
        return sample


def resample_to_target_tr(
    x: Tensor,
    tr: float,
    target_tr: float,
    mode: str = "linear",
) -> Tensor:
    # x: [T, D]
    x = F.interpolate(
        x.T.unsqueeze(0),
        size=round(float(tr) * len(x) / float(target_tr)),
        mode=mode,
    )  # [1, D, T]
    return x.squeeze(0).T


@register_model
def omnifmri(**kwargs) -> tuple[OmniFmriTransform, OmniFmriWrapper]:
    return OmniFmriTransform(**kwargs), OmniFmriWrapper()
