#!/bin/bash

set -euo pipefail

# all target spaces required by different models
spaces=(
    schaefer400
    schaefer400_tians3
    flat
    a424
    mni
    mni_cortex
    schaefer400_tians3_buckner7
)

SPACEIDS="0 1 2 3 4 6"
# ROOT="data/fmriprep"
ROOT="s3://medarc/fmri-fm-eval/PPMI/fmriprep"
OUT_ROOT="s3://medarc/fmri-datasets/eval"

log_path="logs/make_ppmi_dataset.log"

for ii in $SPACEIDS; do
    space=${spaces[ii]}
    uv run python scripts/make_ppmi_dataset.py \
        --root "${ROOT}" \
        --out-root "${OUT_ROOT}" \
        --space "${space}" \
        2>&1 | tee -a "${log_path}"
done
