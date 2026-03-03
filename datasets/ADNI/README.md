# ADNI Preprocessing Scripts

## Step 1: DICOM → BIDS

Conversion was done using [ADNI2bids](https://github.com/mihirneal/ADNI2bids),
a toolkit supporting 100+ ADNI protocol variants across all ADNI phases.

## Step 2: fMRIPrep

Run fmriprep on a single subject:

    ./run_fmriprep.sh <SUBJECT_ID>

Run on all subjects in parallel (requires GNU parallel):

    ./run_parallel_fmriprep.sh subjects.txt

See each script's header for configurable env vars.

## Step 3: Curation & Split

Requires ADNI access. Download the `ADNIMERGE` CSV from [LONI IDA](https://ida.loni.usc.edu)
and place it in `../metadata/`. Then run:

    uv run python adni_curation.py

Or point to a different ADNIMERGE filename:

    ADNI_MERGE_CSV=/path/to/ADNIMERGE.csv uv run python adni_curation.py

Output: `../metadata/adni_fmri_benchmark_split.csv`
This CSV is consumed by `make_adni_targets.py` (defaults to that path; override with `ADNI_CSV_PATH`).
