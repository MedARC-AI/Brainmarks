# Brainmarks

[![Preprint](https://img.shields.io/badge/arXiv-preprint-green?logo=bookstack&logoColor=white)](https://arxiv.org/abs/2510.13768)
[![Discord](https://dcbadge.limes.pink/api/server/https://discord.gg/tVR4TWnRM9?style=flat)](https://discord.gg/tVR4TWnRM9)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)

Brainmarks is an open evaluation suite for fMRI foundation models.

## Installation

```bash
pip install brainmarks
# or
uv add brainmarks
```

Model wrappers for third-party encoders are optional extras:

```bash
pip install "brainmarks[brain-jepa,brainlm,swift,brainharmonix,brain-semantoks,neurostorm]"
```

To install the latest development version from GitHub:

```bash
pip install "brainmarks @ git+https://github.com/MedARC-AI/brainmarks"
```

**From source:**

```bash
git clone https://github.com/MedARC-AI/brainmarks
cd brainmarks
uv sync --python 3.11
```

## Usage

Brainmarks has two main evaluation modes.

**Probe**: trains a frozen-backbone classifier head (linear, attention, or MLP):

```bash
python -m brainmarks.main_probe <model> <representation> <classifier> <dataset>
# e.g.
python -m brainmarks.main_probe brainlm_vitmae_111m patch attn nsd_cococlip
```

**Logistic**: extracts embeddings once and fits a logistic regression:

```bash
python -m brainmarks.main_logistic <model> <representation> <dataset>
# e.g.
python -m brainmarks.main_logistic brainlm_vitmae_111m patch aabc_sex
```

`representation` selects which embedding type the model exposes to the head: `cls`, `reg` (registers), or `patch`. Pass `--help` to either command to see the full list of available models and datasets. Use `--config` to pass a YAML config file and `--overrides key=value` for per-run overrides.

```bash
# e.g.
python -m brainmarks.main_logistic \
  brainlm_vitmae_111m \
  patch \
  aabc_sex \
  --overrides \
  batch_size=16 \
  num_workers=4 \
  device=cpu
```

All available options are documented in the default configs: [default_probe.yaml](src/brainmarks/config/default_probe.yaml), [default_logistic.yaml](src/brainmarks/config/default_logistic.yaml).

## Datasets

Benchmark datasets are distributed in Huggingface Arrow format hosted in the Brainmarks R2 bucket. To request access, fill out [this form](https://forms.gle/VGnakBFCBoNnUt2C7).

Once you have credentials, configure them as environment variables:

```bash
export AWS_ACCESS_KEY_ID=...
export AWS_SECRET_ACCESS_KEY=...
export AWS_ENDPOINT_URL_S3=...   # Cloudflare R2 endpoint
```

Datasets are downloaded automatically on first use and saved in the Huggingface dataset cache.

## Adding a model

Brainmarks uses [namespace package plugin discovery](https://packaging.python.org/en/latest/guides/creating-and-discovering-plugins/#using-namespace-packages). To add a model from your own repo without modifying this one:

0. Install `brainmarks` as a dependency in your project environment.

1. Create a `brainmarks` namespace package in your repo:

    ```bash
    mkdir -p my_repo/src/brainmarks/models
    ```

2. Copy [`src/brainmarks/models/template.py`](src/brainmarks/models/template.py) as a starting point and implement `ModelWrapper`, `ModelTransform`, and a `@register_model` constructor.

3. Validate with the smoke test:

    ```bash
    python -m brainmarks.models.test_models my_model
    ```

See [template.py](src/brainmarks/models/template.py) for more details.

## Adding a dataset

Adding a dataset involves two parts: curation scripts that preprocess raw data into Arrow shards, and a loader module that registers the dataset with Brainmarks.

**Curation scripts** live in [datasets/](datasets/), one subdirectory per source dataset. See [datasets/HCP-YA/](datasets/HCP-YA/) for a reference example — it contains metadata, preprocessing scripts, and a README describing the raw data layout and curation steps.

**Loader modules** live in [src/brainmarks/datasets/](src/brainmarks/datasets/). Each module defines one or more functions decorated with `@register_dataset` that load Arrow shards (local or from S3) into an `HFDataset`. See [src/brainmarks/datasets/hcpya.py](src/brainmarks/datasets/hcpya.py) as a reference.

Dataset loader modules are discovered via the same [namespace package plugin mechanism](https://packaging.python.org/en/latest/guides/creating-and-discovering-plugins/#using-namespace-packages) as models, so they can live in an external repo.

## Support

For help with any issues, reach out to us on [MedARC Discord](https://discord.gg/tVR4TWnRM9) in the `#neuro-fm` channel.

## Citation

```bibtex
@article{lane2025scaling,
  title   = {Scaling Vision Transformers for Functional {MRI} with Flat Maps},
  author  = {Lane, Connor and Tripathy, Mihir and Murali, Leema Krishna and
             Grandhi, Ratna Sagari and Yang, Shamus Sim Zi and Gijsen, Sam and
             Das, Debojyoti and Ram, Manish and Singh, Utkarsh Kumar and
             Villanueva, Cesar Kadir Torrico and Wei, Yuxiang and Beddow, Will and
             Cort\'{e}s, Gianfranco and Cho, Suin and Kaplan, Daniel Z. and
             Warner, Benjamin and Abraham, Tanishq Mathew and Scotti, Paul S.},
  journal = {arXiv preprint arXiv:2510.13768},
  year    = {2025},
  url     = {https://arxiv.org/abs/2510.13768}
}
```
