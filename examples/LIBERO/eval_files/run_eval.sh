#!/usr/bin/env bash
set -euo pipefail
REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../.." && pwd)"
export PYTHONPATH="${REPO_ROOT}${LIBERO_HOME:+:${LIBERO_HOME}}${PYTHONPATH:+:${PYTHONPATH}}"
export MUJOCO_GL="${MUJOCO_GL:-egl}"
: "${PRETRAINED_CHECKPOINT:?Set PRETRAINED_CHECKPOINT to the LIBERO-finetuned weight file}"
for suite in ${TASK_SUITES:-libero_spatial libero_object libero_goal libero_10}; do
    "${SIM_PYTHON:-python}" "${REPO_ROOT}/examples/LIBERO/eval_files/eval_libero.py" \
        --args.pretrained-path "${PRETRAINED_CHECKPOINT}" --args.host "${POLICY_HOST:-127.0.0.1}" \
        --args.port "${POLICY_PORT:-10093}" --args.task-suite-name "$suite" \
        --args.num-trials-per-task "${TRIALS_PER_TASK:-50}" \
        --args.output-dir "${OUTPUT_ROOT:-${REPO_ROOT}/outputs}/LIBERO" "$@"
done
