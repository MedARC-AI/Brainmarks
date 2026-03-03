# UKBB (UK Biobank) Dataset

The UKBB dataset contains resting-state fMRI from 740 subjects. TR is 0.735s with 360 frames (~5 minutes) per scan.

## Downloading the Data

UKBB data is available through the UK Biobank data access portal:

- **Access**: https://www.ukbiobank.ac.uk/enable-your-research/apply-for-access
- You must apply for access and be approved before downloading any data.
- Once approved, download the preprocessed rfMRI data (CIFTI `.dtseries.nii` files) and the phenotypic participant CSV.

After downloading, place (or symlink) the raw subject files under `data/sourcedata/`:

```bash
ln -s /path/to/ukbb_data datasets/UKBB/data/sourcedata
```

Expected folder structure under `data/sourcedata/`:

```
data/sourcedata/
  sub-{eid}_ses-{ses}_mod-rfMRI.bold.npy
  sub-{eid}_ses-{ses}_mod-rfMRI.dtseries.nii
  sub-{eid}_ses-{ses}_mod-rfMRI.meta.json
  ...
```

### Phenotypic CSV

The participant phenotypic CSV contains sensitive data and is **not** stored in the repo. Set the `UKBB_METADATA_CSV` environment variable to its path before running the scripts that need it:

```bash
export UKBB_METADATA_CSV=/path/to/ukbb_participants.csv
```

The CSV must contain columns: `participant.eid`, `participant.p31` (sex), `participant.p21003_i0` (age).

---

## Pipeline Overview

Run the scripts in this order:

```
1. make_ukbb_subject_batch_splits.py   (requires UKBB_METADATA_CSV)
2. make_ukbb_metadata.py
3. make_ukbb_targets.py                (requires UKBB_METADATA_CSV)
4. make_ukbb_arrow.py  (or make_ukbb_eval.sh to run all spaces)
```

---

## Scripts

### 1. `make_ukbb_subject_batch_splits.py`

Splits subjects into 20 non-overlapping batches, stratified by sex and age.

**What it does:**
- Discovers subjects from `.bold.npy` files under `data/sourcedata/`
- Loads age and sex from the phenotypic CSV (`UKBB_METADATA_CSV`)
- Creates 3 quantile-based age bins and combines with sex to form 6 stratification groups
- Uses `StratifiedKFold` to produce 20 balanced batches
- Saves the result to `metadata/ukbb_subject_batch_splits.json`

**Batch allocation:**

| Batches | Purpose |
|---------|---------|
| 0–13 | Train |
| 14–16 | Validation |
| 17–19 | Test |

**Output:** `metadata/ukbb_subject_batch_splits.json`

**Run:**
```bash
export UKBB_METADATA_CSV=/path/to/ukbb_participants.csv
uv run python datasets/UKBB/scripts/make_ukbb_subject_batch_splits.py
```

---

### 2. `make_ukbb_metadata.py`

Builds a metadata parquet table from the `.meta.json` sidecar files and subject batch splits.

**What it does:**
- Reads all `sub-*_ses-*_mod-rfMRI.meta.json` files from `data/sourcedata/`
- Looks up each subject's train/val/test split and batch ID from the batch splits JSON
- Records the path to the corresponding `.bold.npy` file, TR, and number of frames
- Saves the result to `metadata/ukbb_metadata.parquet`

**Output:** `metadata/ukbb_metadata.parquet`

**Run:**
```bash
uv run python datasets/UKBB/scripts/make_ukbb_metadata.py
```

---

### 3. `make_ukbb_targets.py`

Generates per-subject prediction target files from the phenotypic CSV.

**What it does:**
- Loads age and sex from the phenotypic CSV (`UKBB_METADATA_CSV`)
- Restricts to subjects that have imaging data under `data/sourcedata/`
- Produces quantile-binned age labels and binary gender labels
- Saves a combined CSV and per-target JSON files

**Targets produced:**

| Target | Type | Description |
|--------|------|-------------|
| `Age` | 3-bin quantile | Age in years |
| `Gender` | binary (F/M) | Biological sex |

**Outputs:**
- `metadata/ukbb_pheno_targets.csv` — combined table with Subject, Age, Age_Q, Gender
- `metadata/targets/ukbb_target_map_Age.json` — `{subject_id: bin_label}`
- `metadata/targets/ukbb_target_info_Age.json` — bin edges and counts
- `metadata/targets/ukbb_target_map_Gender.json` — `{subject_id: "F"/"M"}`
- `metadata/targets/ukbb_target_info_Gender.json` — class counts

**Run:**
```bash
export UKBB_METADATA_CSV=/path/to/ukbb_participants.csv
uv run python datasets/UKBB/scripts/make_ukbb_targets.py
```

---

### 4. `make_ukbb_arrow.py`

Builds the HuggingFace Arrow evaluation dataset for a given parcellation space.

**What it does:**
- Reads scan paths and split assignments from `metadata/ukbb_metadata.parquet`
- Reads phenotypic targets from `metadata/ukbb_pheno_targets.csv`
- Loads each subject's CIFTI `.dtseries.nii` file, truncates to 360 frames
- Normalizes each scan (z-score per parcel) via `nisc.scale`
- Saves a HuggingFace `DatasetDict` with splits: `train`, `validation`, `test`

**Output:** `data/processed/ukbb.{space}.arrow`

**Each sample contains:**

| Field | Type | Description |
|-------|------|-------------|
| `sub` | string | Subject EID |
| `gender` | string | `"F"` or `"M"` |
| `age_q` | int32 | Age quantile bin (0, 1, 2) |
| `path` | string | Relative path to source file |
| `start` | int32 | Start frame (always 0) |
| `end` | int32 | End frame (always 360) |
| `n_frames` | int32 | Number of frames (360) |
| `tr` | float32 | Repetition time (0.735s) |
| `bold` | float16 `[360, D]` | z-scored BOLD data |
| `mean` | float32 `[1, D]` | Per-parcel mean before scaling |
| `std` | float32 `[1, D]` | Per-parcel std before scaling |

**Run (single space):**
```bash
uv run python datasets/UKBB/scripts/make_ukbb_arrow.py --space flat --num_proc 8
```

**Available spaces:** `flat`, `schaefer400`, `schaefer400_tians3`, `a424`

> **Note:** `mni`, `mni_cortex`, and `schaefer400_tians3_buckner7` are not supported as UKBB only provides CIFTI files.

---

### 5. `make_ukbb_eval.sh`

Convenience shell script that runs `make_ukbb_arrow.py` for all supported spaces sequentially.

**Run:**
```bash
bash datasets/UKBB/scripts/make_ukbb_eval.sh
```

Logs are written to `logs/make_ukbb_eval.log`.

---

## Metadata Files

| File | Description |
|------|-------------|
| `metadata/ukbb_subject_batch_splits.json` | 20-batch subject splits (sex+age stratified) |
| `metadata/ukbb_metadata.parquet` | Per-scan metadata (subject, TR, n_frames, split, path) |
| `metadata/ukbb_pheno_targets.csv` | Combined phenotypic targets table |
| `metadata/targets/ukbb_target_map_Age.json` | Subject-to-bin-label mapping for age |
| `metadata/targets/ukbb_target_info_Age.json` | Age bin edges and counts |
| `metadata/targets/ukbb_target_map_Gender.json` | Subject-to-label mapping for gender |
| `metadata/targets/ukbb_target_info_Gender.json` | Gender class counts |
