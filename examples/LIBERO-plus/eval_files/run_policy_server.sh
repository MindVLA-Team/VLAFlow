#!/usr/bin/env bash
set -euo pipefail
REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../.." && pwd)"
export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
: "${PRETRAINED_CHECKPOINT:?Set PRETRAINED_CHECKPOINT to the fine-tuned weight file}"
exec "${POLICY_PYTHON:-python}" -m deployment.model_server.server_policy --ckpt_path "${PRETRAINED_CHECKPOINT}" --port "${POLICY_PORT:-10093}" --use_bf16 "$@"
