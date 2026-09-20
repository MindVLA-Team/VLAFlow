#!/usr/bin/env bash
set -euo pipefail
REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../.." && pwd)"
export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
exec "${PYTHON:-python}" -m vlaflow.launch --config "${REPO_ROOT}/examples/OXE_Mix/train_files/mindpi_oxemix_pretrain.yaml" --phase pretrain "$@"
