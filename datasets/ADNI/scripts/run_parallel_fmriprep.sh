#!/usr/bin/env bash
# Usage: run_parallel_fmriprep.sh <subjects_file>
# subjects_file: plain text, one SUBJECT_ID per line
# Env: PARALLEL_JOBS (default 48), plus all vars from run_fmriprep.sh
set -euo pipefail

[[ -z "${1:-}" ]] && { echo "Usage: $0 <subjects_file>"; exit 1; }
SUBJECTS_FILE="$1"
[[ ! -f "$SUBJECTS_FILE" ]] && { echo "File not found: $SUBJECTS_FILE"; exit 1; }

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUN_SCRIPT="${SCRIPT_DIR}/run_fmriprep.sh"
LOG_DIR="${ADNI_LOG_DIR:-/data/ADNI_fmriprep/logs}"
JOBS="${PARALLEL_JOBS:-48}"

chmod +x "$RUN_SCRIPT"
mkdir -p "$LOG_DIR"

SUBJECT_COUNT=$(grep -c . "$SUBJECTS_FILE")
echo "Running fMRIPrep for $SUBJECT_COUNT subjects with $JOBS parallel jobs"
echo "Joblog: ${LOG_DIR}/parallel_joblog.log"

parallel -j "$JOBS" \
    --line-buffer \
    --tag \
    --joblog "${LOG_DIR}/parallel_joblog.log" \
    --resume-failed \
    "$RUN_SCRIPT" {} \
    :::: "$SUBJECTS_FILE"

echo "Done. Check ${LOG_DIR}/ for per-subject logs."
