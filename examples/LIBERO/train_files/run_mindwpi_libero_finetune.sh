#!/usr/bin/env bash
set -euo pipefail
REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../.." && pwd)"
export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
exec "${PYTHON:-python}" -m vlaflow.launch --config "${REPO_ROOT}/examples/LIBERO/train_files/mindwpi_libero_finetune.yaml" --phase finetune "$@"
