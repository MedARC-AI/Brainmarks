# Brain-Semantoks Model Wrapper for fmri-fm-eval
# Paper: "Learning Semantic Tokens of Brain Dynamics with a Self-Distilled Foundation Model"
# HuggingFace: https://huggingface.co/SamGijsen/Brain-Semantoks
# GitHub: https://github.com/SamGijsen/Brain-Semantoks

"""
Brain-Semantoks model wrapper for fmri-fm-eval.

Model requirements:
- Input: [B, C=457, T] where C = 400 (Schaefer) + 50 (Tian S3) + 7 (Buckner cerebellar)
- TR: 2.0 seconds
- Sequence length: 100 timepoints (5 temporal patches of 20 each)
- ROI ordering: Schaefer uses 7-network ordering (NOT 17-network)
- Normalization: z-score per ROI per subject

Output:
- cls_embeds: [B, 1, D] - CLS token embedding
- patch_embeds: [B, N*P, D] - network-temporal patch tokens
  where N = num_networks (9) and P = num_temporal_patches (5)

Data space: "schaefer400_tians3_buckner7" (457 ROIs)

HuggingFace: https://huggingface.co/SamGijsen/Brain-Semantoks
"""

import math

import torch
import torch.nn as nn
from huggingface_hub import hf_hub_download

from fmri_fm_eval.models.base import Embeddings, ModelWrapper
from fmri_fm_eval.models.registry import register_model

HUGGINGFACE_REPO = "SamGijsen/Brain-Semantoks"

TARGET_TR = 2.0
TARGET_LENGTH = 100
PATCH_SIZE = 20
NUM_ROIS = 457  # 400 Schaefer + 50 Tian + 7 Buckner
NUM_NETWORKS = 9  # 7 Schaefer networks + 1 Tian + 1 Buckner
EMBEDDING_DIM = 768


class BrainSemantoksTransform:
    """
    Preprocesses fMRI data for Brain-Semantoks model.

    Expected input sample (from Arrow dataset):
        - bold: [T, D=457] - z-scored timeseries
        - mean: [1, D] - original means (for unnormalization if needed)
        - std: [1, D] - original stds (for unnormalization if needed)
        - tr: float - repetition time in seconds

    Output sample:
        - bold: [D=457, T=100] - transposed and padded to target length
        - mask: [N=9, P=5] - True for fully padded temporal patches (same across networks)
    """

    def __init__(
        self,
        target_tr: float = TARGET_TR,
        target_length: int = TARGET_LENGTH,
        patch_size: int = PATCH_SIZE,
        num_networks: int = NUM_NETWORKS,
    ):
        self.target_tr = target_tr
        self.target_length = target_length
        self.patch_size = patch_size
        self.num_networks = num_networks
        self.max_patches = target_length // patch_size  # 5 patches

    def __call__(self, sample: dict) -> dict:
        bold = sample["bold"]  # [T, D=457]
        tr = float(sample["tr"])

        # 1. Temporal resampling to target TR (2.0s)
        if abs(tr - self.target_tr) > 0.01:
            bold = self._resample_temporal(bold, tr, self.target_tr)

        # 2. Handle sequence length
        T = bold.shape[0]

        if T >= self.target_length:
            # Truncate to target length
            bold = bold[: self.target_length]
            num_patches_with_data = self.max_patches
        else:
            # Pad to target length (model requires exactly 100 timepoints)
            # Count patches that have at least some valid data
            num_patches_with_data = math.ceil(T / self.patch_size)

            # Zero-pad at end to reach target_length
            pad_length = self.target_length - T
            bold = torch.nn.functional.pad(bold, (0, 0, 0, pad_length), value=0.0)

        # 3. Create mask for fully padded temporal patches
        # Shape: [N, P] where N=num_networks, P=max_patches
        # True = masked (fully padded, no valid data), False = valid (has data)
        # Same mask across all networks since they share temporal structure
        temporal_mask = torch.ones(self.max_patches, dtype=torch.bool)
        temporal_mask[:num_patches_with_data] = False
        # Expand to [N, P] - same mask for all networks
        mask = temporal_mask.unsqueeze(0).expand(self.num_networks, -1)

        # 4. Transpose to [D, T] format expected by model
        bold = bold.T  # [D, T]

        # Update sample
        sample["bold"] = bold
        sample["mask"] = mask

        return sample

    def _resample_temporal(
        self, bold: torch.Tensor, source_tr: float, target_tr: float
    ) -> torch.Tensor:
        """Resample timeseries using cubic interpolation with anti-aliasing for downsampling."""
        from fmri_fm_eval import nisc

        bold_np = bold.numpy()
        # Anti-aliasing filter requires minimum sequence length (~30 samples for padding)
        # Disable for very short sequences to avoid scipy filter error
        antialias = len(bold_np) >= 30
        bold_resampled = nisc.resample_timeseries(
            bold_np, source_tr, target_tr, kind="cubic", antialias=antialias
        )
        return torch.from_numpy(bold_resampled).to(bold.dtype)


class BrainSemantoksWrapper(ModelWrapper):
    __space__: str = "schaefer400_tians3_buckner7"

    def __init__(self, encoder: nn.Module):
        super().__init__()
        self.encoder = encoder

    def forward(self, batch: dict) -> Embeddings:
        """
        Args:
            batch: dict with keys:
                - bold: [B, D=457, T=100] - input timeseries
                - mask: [B, N=9, P=5] - True for fully padded temporal patches

        Returns:
            Embeddings namedtuple with:
                - cls_embeds: [B, 1, D] - CLS token
                - reg_embeds: None - not used
                - patch_embeds: [B, N*P, D] - network-temporal patch tokens
        """
        x = batch["bold"]  # [B, D=457, T=100]
        mask = batch.get("mask", None)  # [B, N, P] - True for fully padded patches

        # Forward through encoder
        # The model uses mask to replace fully padded patches with learned mask embedding
        output = self.encoder(x, atlas=0, mask=mask)

        # Extract embeddings
        # output["global_cls"]: [B, D]
        # output["tokens"]: [B, N*P+1, D] (includes CLS at position 0)

        cls_embeds = output["global_cls"].unsqueeze(1)  # [B, 1, D]
        patch_embeds = output["tokens"][:, 1:]  # [B, N*P, D] - exclude CLS

        return Embeddings(
            cls_embeds=cls_embeds,
            reg_embeds=None,
            patch_embeds=patch_embeds,
        )


@register_model
def brain_semantoks(**kwargs):
    """
    Create Brain-Semantoks model for fmri-fm-eval.
    Downloads pretrained weights from HuggingFace and initializes the model.
    """

    checkpoint_path = hf_hub_download(HUGGINGFACE_REPO, "brainsemantoks_ckpt_epoch_100.pth")
    network_map_path = hf_hub_download(HUGGINGFACE_REPO, "network_mapping.npz")

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)

    encoder = _create_encoder(network_map_path)

    # Load teacher weights
    teacher_weights = _extract_teacher_weights(checkpoint)
    encoder.load_state_dict(teacher_weights)
    encoder.eval()

    transform = BrainSemantoksTransform()
    wrapper = BrainSemantoksWrapper(encoder)

    return transform, wrapper


def _create_encoder(network_map_path: str) -> nn.Module:
    """
    Initialize the CNN_TF encoder with config from checkpoint.
    """
    from model.semantoks import CNN_TF

    # Tokenizer config from checkpoint config
    tokenizer_config = [
        {"type": "dense", "kernel_size": 3, "out_channels": 384, "depthwise": False},
        {
            "type": "sgconv",
            "kernel_size": 4,
            "out_channels": 384,
            "num_scales": 3,
            "decay_min": 2.0,
            "decay_max": 2.0,
        },
    ]

    encoder = CNN_TF(
        target_time_length=TARGET_LENGTH,
        patch_size=PATCH_SIZE,
        network_data_path=network_map_path,
        atlas_names=["schaefer400", "tian3", "buckner7"],
        atlas_network_counts=[7, 1, 1],  # 7 Schaefer networks + 1 Tian + 1 Buckner = 9 total
        embedding_dim=EMBEDDING_DIM,
        depth=8,
        heads=12,
        mlp_dim=3072,
        drop_path_rate=0.0,
        layer_scale_init_value=0.1,
        emb_dropout=0.0,
        do_masking=True,
        tokenizer_config=tokenizer_config,
        tokenizer_final_norm="layer",
        tokenizer_pooling_type="mean",
        flash_attention="auto",
    )
    return encoder


def _extract_teacher_weights(checkpoint: dict) -> dict:
    """
    Extract teacher encoder weights from checkpoint.

    Brain-Semantoks uses DINO-style training with student/teacher.
    For inference, we use the teacher encoder.
    """
    state_dict = checkpoint.get("model_state_dict", checkpoint)

    # Remove 'module.' prefix if present (DataParallel)
    if any(k.startswith("module.") for k in state_dict.keys()):
        state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}

    # Extract teacher encoder weights
    teacher_prefix = "teacher_encoder."
    teacher_weights = {
        k.replace(teacher_prefix, ""): v
        for k, v in state_dict.items()
        if k.startswith(teacher_prefix)
    }

    if not teacher_weights:
        raise ValueError(
            f"No teacher encoder weights found in checkpoint. "
            f"Available key prefixes: {set(k.split('.')[0] for k in state_dict.keys())}"
        )

    return teacher_weights
