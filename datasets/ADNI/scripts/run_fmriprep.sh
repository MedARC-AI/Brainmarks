#!/bin/bash
# Usage: run_fmriprep.sh <SUBJECT_ID>
# Env vars: ADNI_BIDS_DIR, ADNI_OUT_DIR, ADNI_WORK_DIR, ADNI_LOG_DIR, FS_LICENSE_FILE
set -euo pipefail

[[ -z "${1:-}" ]] && { echo "Usage: $0 SUBID"; exit 1; }
subid="$1"

BIDS_DIR="${ADNI_BIDS_DIR:-/data/ADNI_bids}"
OUT_DIR="${ADNI_OUT_DIR:-/data/ADNI_fmriprep/output}"
WORK_DIR="${ADNI_WORK_DIR:-/data/ADNI_fmriprep/workdir}/${subid}"
LOG_DIR="${ADNI_LOG_DIR:-/data/ADNI_fmriprep/logs}"
FS_LICENSE="${FS_LICENSE_FILE:-$(readlink -f "$(dirname "$0")/license.txt")}"
FSAVG_DIR="$(readlink -f "$(dirname "$0")/resources/fsaverage")"

mkdir -p "$OUT_DIR" "$WORK_DIR" "$LOG_DIR"

docker run --rm \
    --name "$subid" \
    -v "${BIDS_DIR}:/data:ro" \
    -v "${OUT_DIR}:/out" \
    -v "${FSAVG_DIR}:/out/sourcedata/freesurfer/fsaverage:ro" \
    -v "${WORK_DIR}:/work" \
    -v "${FS_LICENSE}:/opt/freesurfer/license.txt:ro" \
    nipreps/fmriprep:25.2.3 \
    /data /out participant \
    --participant-label "$subid" \
    --fs-license-file /opt/freesurfer/license.txt \
    --work-dir /work \
    --output-spaces T1w MNI152NLin6Asym:res-2 \
    --ignore fieldmaps slicetiming sbref t2w fmap-jacobian \
    --cifti-output 91k \
    --skip-bids-validation \
    --omp-nthreads 2 --nthreads 2 --mem_mb 40000 \
    --subject-anatomical-reference sessionwise \
    --stop-on-first-crash \
    2>&1 | tee -a "${LOG_DIR}/${subid}.log"
