#!/bin/bash

if [[ -z $1 || $1 == "-h" || $1 == "--help" ]]; then
    echo "make_hcpya_arrow.sh DATASET"
    exit
fi

DATASET=$1
SPACEIDS="0 1 2 3 4 5 6"

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

# nb, volume data not currently stored locally
# but remote is fine since the script is not blocked waiting for download
roots=(
    data/sourcedata/HCP_1200
    data/sourcedata/HCP_1200
    data/sourcedata/HCP_1200
    s3://hcp-openaccess/HCP_1200
    s3://hcp-openaccess/HCP_1200
    s3://hcp-openaccess/HCP_1200
    s3://hcp-openaccess/HCP_1200
)

log_path="logs/make_hcpya_${DATASET}_arrow.log"

for ii in $SPACEIDS; do
    space=${spaces[ii]}
    root=${roots[ii]}
    uv run python scripts/make_hcpya_${DATASET}_arrow.py \
        --space "${space}" \
        --root "${root}" \
        2>&1 | tee -a "${log_path}"
done
