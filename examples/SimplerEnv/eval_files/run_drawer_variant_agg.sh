#!/usr/bin/env bash
set -euo pipefail
REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../.." && pwd)"
export PYTHONPATH="${REPO_ROOT}${SIMPLER_ROOT:+:${SIMPLER_ROOT}}${PYTHONPATH:+:${PYTHONPATH}}"
: "${PRETRAINED_CHECKPOINT:?Set PRETRAINED_CHECKPOINT}"
: "${SIMPLER_ROOT:?Set SIMPLER_ROOT to the installed simulator checkout}"
MODEL_PATH="${PRETRAINED_CHECKPOINT}"
ckpt_path="${PRETRAINED_CHECKPOINT}"
SimplerEnv_PATH="${SIMPLER_ROOT}"
port="${POLICY_PORT:-10093}"
run_count=0
case_index=0
TSET_NUM=1
run_case() {
    local result_dir="${OUTPUT_ROOT:-${REPO_ROOT}/outputs}/simpler/${task_name}/case_${case_index}"
    local command=("${SIM_PYTHON:-python}" "${REPO_ROOT}/examples/SimplerEnv/eval_files/start_simpler_env.py"
        --host "${POLICY_HOST:-127.0.0.1}" --logging-dir "$result_dir" "$@")
    if [[ "${DRY_RUN:-0}" == "1" ]]; then
        printf '%q ' "${command[@]}"
        printf '\n'
    else
        "${command[@]}"
    fi
    case_index=$((case_index + 1))
}

task_name=drawer_variant_agg
declare -a ckpt_paths=(
  "${MODEL_PATH}"
)
declare -a env_names=(
  CloseTopDrawerCustomInScene-v0
  CloseMiddleDrawerCustomInScene-v0
  CloseBottomDrawerCustomInScene-v0
  OpenTopDrawerCustomInScene-v0
  OpenMiddleDrawerCustomInScene-v0
  OpenBottomDrawerCustomInScene-v0
)
EXTRA_ARGS="--enable-raytracing"
run_count=0
scene_name=frl_apartment_stage_simple
EvalSim() {
:
    run_case --ckpt-path "${ckpt_path}"      --robot google_robot_static      --port $port      --control-freq 3 --sim-freq 513 --max-episode-steps 113      --env-name ${env_name} --scene-name ${scene_name}      --robot-init-x 0.65 0.85 3 --robot-init-y -0.2 0.2 3      --robot-init-rot-quat-center 0 0 0 1 --robot-init-rot-rpy-range 0 0 1 0 0 1 0.0 0.0 1      --obj-init-x-range 0 0 1 --obj-init-y-range 0 0 1      ${EXTRA_ARGS}
}
for ckpt_path in "${ckpt_paths[@]}"; do
  for env_name in "${env_names[@]}"; do
    if (( (run_count + 1) == 32 )); then
      EvalSim
    else
      EvalSim
    fi
    run_count=$((run_count + 1))
  done
done
declare -a scene_names=(
  "modern_bedroom_no_roof"
  "modern_office_no_roof"
)
for scene_name in "${scene_names[@]}"; do
  for ckpt_path in "${ckpt_paths[@]}"; do
    for env_name in "${env_names[@]}"; do
      EXTRA_ARGS="--additional-env-build-kwargs shader_dir=rt"
      if (( (run_count + 1) == 32 )); then
        EvalSim
      else
        EvalSim
      fi
      run_count=$((run_count + 1))
    done
  done
done
scene_name=frl_apartment_stage_simple
for ckpt_path in "${ckpt_paths[@]}"; do
  for env_name in "${env_names[@]}"; do
    EXTRA_ARGS="--additional-env-build-kwargs shader_dir=rt light_mode=brighter"
    if (( (run_count + 1) == 32 )); then
      EvalSim
    else
      EvalSim
    fi
    run_count=$((run_count + 1))
    EXTRA_ARGS="--additional-env-build-kwargs shader_dir=rt light_mode=darker"
    if (( (run_count + 1) % 32 == 0 )); then
      EvalSim
    else
      EvalSim
    fi
    run_count=$((run_count + 1))
  done
done
scene_name=frl_apartment_stage_simple
for ckpt_path in "${ckpt_paths[@]}"; do
  for env_name in "${env_names[@]}"; do
    EXTRA_ARGS="--additional-env-build-kwargs shader_dir=rt station_name=mk_station2"
    if (( (run_count + 1) % 32 == 0 )); then
      EvalSim
    else
      EvalSim
    fi
    run_count=$((run_count + 1))
    EXTRA_ARGS="--additional-env-build-kwargs shader_dir=rt station_name=mk_station3"
    if (( (run_count + 1) % 32 == 0 )); then
      EvalSim
    else
      EvalSim
    fi
    run_count=$((run_count + 1))
  done
done
