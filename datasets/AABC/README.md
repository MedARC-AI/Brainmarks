# AABC (Aging Brain Cohort) Dataset

The AABC dataset is a multi-visit resting-state and task fMRI cohort of ~2,214 subjects collected by the Human Connectome Project. It includes four scan types per subject-visit: resting-state (REST), CARIT, FACENAME, and VISMOTOR. TR is 0.72s throughout.

## Downloading the Data

AABC data is available through the HCP data portal:

- **Access**: https://db.humanconnectome.org
- You will need to register for an account and agree to the data use terms.
- Once approved, download the preprocessed MNINonLinear fMRI outputs (CIFTI `.dtseries.nii` and/or NIfTI `.nii.gz` files).
- The phenotypic CSV (`AABC_subjects_*.csv`) is also downloaded from the same portal.

After downloading, place the raw subject folders under `data/raw/` inside this dataset directory, or create a symlink to wherever the data lives:

```bash
ln -s /path/to/AABC_data datasets/AABC/data/raw
```

All scripts expect the data at `data/raw/`. Expected folder structure:

```
data/raw/
  HCA6000030_V1_MR/
    MNINonLinear/
      Results/
        rfMRI_REST/
          rfMRI_REST_Atlas_MSMAll_hp0_clean_rclean_tclean.dtseries.nii
        tfMRI_CARIT_PA/
          ...
  HCA6000031_V1_MR/
    ...
```

---

## Pipeline Overview

Run the scripts in this order:

```
1. make_aabc_metadata.py
2. make_aabc_subject_batch_splits.py
3. make_aabc_targets.py
4. make_aabc_eval.py  (or make_aabc_eval.sh to run all spaces)
5. make_aabc_pretrain.py
```

---

## Scripts

### 1. `make_aabc_metadata.py`

Scans all CIFTI files under `AABC_ROOT` and builds a metadata parquet table.

**What it does:**
- Recursively finds all `*_Atlas_MSMAll_hp0_clean_rclean_tclean.dtseries.nii` files
- Parses subject ID, visit, modality, and task from the directory structure
- Reads the CIFTI header to get the number of frames (`n_frames`)
- Saves the result to `metadata/aabc_metadata.parquet`

**Output:** `metadata/aabc_metadata.parquet`

**Run:**
```bash
uv run python datasets/AABC/scripts/make_aabc_metadata.py
```

---

### 2. `make_aabc_subject_batch_splits.py`

Splits subjects into 20 non-overlapping batches, stratified by sex and respecting family structure (no family members split across batches).

**What it does:**
- Loads subject list from `metadata/aabc_metadata.parquet`
- Reads `pedid` (family group ID) and `sex` from the phenotypic CSV (`AABC_subjects_*.csv`) at `AABC_ROOT`
- Uses `StratifiedGroupKFold` to create 20 balanced batches
- Saves the result to `metadata/aabc_subject_batch_splits.json`

**Batch allocation:**
| Batches | Purpose |
|---------|---------|
| 0–9 | Pretraining |
| 10–19 | Evaluation (train/val/test split) |

**Output:** `metadata/aabc_subject_batch_splits.json`

**Run:**
```bash
uv run python datasets/AABC/scripts/make_aabc_subject_batch_splits.py
```

> **Note:** Requires the phenotypic CSV to be present at `data/raw/AABC_subjects_*.csv`.

---

### 3. `make_aabc_targets.py`

Generates per-subject prediction target files (binned labels) for all phenotypic variables.

**What it does:**
- Loads the phenotypic CSV from `$AABC_ROOT/AABC_subjects_*.csv`
- Restricts to subjects that have a REST scan on disk
- For each target, produces:
  - `aabc_target_map_{target}.json`: `{subject_id: bin_label}` mapping
  - `aabc_target_info_{target}.json`: bin edges, counts, and per-bin statistics

**Targets produced:**

| Target | Type | Description |
|--------|------|-------------|
| `sex` | binary (0=F, 1=M) | Biological sex |
| `age_open` | 4-bin quantile | Age in years |
| `Memory_Tr35_60y` | 4-bin quantile | Memory composite score |
| `FluidIQ_Tr35_60y` | 4-bin quantile | Fluid intelligence |
| `CrystIQ_Tr35_60y` | 4-bin quantile | Crystallized intelligence |
| `neo_n` | quantile bins | NEO-FFI Neuroticism |
| `neo_e` | quantile bins | NEO-FFI Extraversion |
| `neo_o` | 3-bin quantile | NEO-FFI Openness |
| `neo_a` | 3-bin quantile | NEO-FFI Agreeableness |
| `neo_c` | 3-bin quantile | NEO-FFI Conscientiousness |

**Output:** `metadata/targets/aabc_target_map_*.json` and `metadata/targets/aabc_target_info_*.json`

**Run:**
```bash
uv run python datasets/AABC/scripts/make_aabc_targets.py
```

---

### 4. `make_aabc_eval.py`

Builds the HuggingFace Arrow evaluation dataset for a given parcellation space.

**What it does:**
- Uses subjects from eval batches (10–19)
- Selects one visit per subject (randomly, with fixed seed) to avoid data leakage
- Takes a single 500-TR window from the REST scan per subject
- Performs stratified train/val/test split (80/10/10) by age, sex, and FluidIQ
- Normalizes each window (z-score per parcel) via `nisc.scale`
- Saves a HuggingFace `DatasetDict` with splits: `train`, `validation`, `test`

**Output:** `data/processed/aabc.{space}.arrow`

**Each sample contains:**

| Field | Type | Description |
|-------|------|-------------|
| `sub` | string | Subject ID (e.g. `HCA6000030`) |
| `visit` | string | Visit label (e.g. `V1`) |
| `mod` | string | Modality (always `MR`) |
| `task` | string | Always `REST` |
| `path` | string | Relative path to source file |
| `start` | int32 | Start TR index of window |
| `end` | int32 | End TR index of window |
| `tr` | float32 | Repetition time (0.72s) |
| `segment` | int32 | Window segment index (always 0) |
| `bold` | float16 `[T, D]` | z-scored BOLD window |
| `mean` | float32 `[1, D]` | Per-parcel mean before scaling |
| `std` | float32 `[1, D]` | Per-parcel std before scaling |

**Run (single space):**
```bash
uv run python datasets/AABC/scripts/make_aabc_eval.py --space flat --num_proc 8
```

**Available spaces:** `schaefer400`, `schaefer400_tians3`, `flat`, `a424`, `mni`, `mni_cortex`, `schaefer400_tians3_buckner7`

---

### 5. `make_aabc_eval.sh`

Convenience shell script that runs `make_aabc_eval.py` for all supported spaces sequentially.

**Run:**
```bash
bash datasets/AABC/scripts/make_aabc_eval.sh
```

Logs are appended to `logs/make_aabc_eval.log`.

---

### 6. `make_aabc_pretrain.py`

Builds the pretraining dataset in WebDataset TAR format.

**What it does:**
- Uses subjects from pretraining batches (0–9)
- Reads scan paths from `metadata/aabc_metadata.parquet`
- For each subject-visit REST scan, creates up to 3 non-overlapping 500-TR windows
- Normalizes each window (z-score) and stores as float16
- Writes output as sharded TAR files (default 700 MB per shard) using WebDataset

**Output:** `{outdir}/aabc-{space}/aabc-{N:06d}.tar`

**Each TAR entry contains:**
- `{key}.npy`: raw float16 bytes of the BOLD window (shape `[T, D]`)
- `{key}.json`: metadata (subject, visit, task, TR, shape, mean, std, segment)

**Run:**
```bash
uv run python datasets/AABC/scripts/make_aabc_pretrain.py \
    --space flat \
    --outdir /path/to/output \
    --shard_size_mb 700
```

Use `--overwrite` to replace existing shards.

---

## Metadata Files

| File | Description |
|------|-------------|
| `metadata/aabc_metadata.parquet` | Per-scan metadata (subject, visit, task, n_frames, path) |
| `metadata/aabc_subject_batch_splits.json` | 20-batch subject splits (family-aware, sex-stratified) |
| `metadata/aabc_partition_split.json` | Pre-existing partition assignments |
| `metadata/targets/aabc_target_map_{target}.json` | Subject-to-bin-label mapping per phenotype |
| `metadata/targets/aabc_target_info_{target}.json` | Bin edges and statistics per phenotype |
