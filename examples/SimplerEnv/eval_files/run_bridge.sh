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

task_name=bridge
scene_name=bridge_table_1_v1
robot=widowx
rgb_overlay_path="${SimplerEnv_PATH}/ManiSkill2_real2sim/data/real_inpainting/bridge_real_eval_1.png"
robot_init_x=0.147
robot_init_y=0.028
declare -a ENV_NAMES=(
  StackGreenCubeOnYellowCubeBakedTexInScene-v0
  PutCarrotOnPlateInScene-v0
  PutSpoonOnTableClothInScene-v0
)
for i in "${!ENV_NAMES[@]}"; do
  env="${ENV_NAMES[i]}"
  for ((run_idx=1; run_idx<=TSET_NUM; run_idx++)); do
:
    run_case --port $port        --ckpt-path "${ckpt_path}"        --robot ${robot}        --policy-setup widowx_bridge        --control-freq 5        --sim-freq 500        --max-episode-steps 120        --env-name "${env}"        --scene-name ${scene_name}        --rgb-overlay-path "${rgb_overlay_path}"        --robot-init-x ${robot_init_x} ${robot_init_x} 1        --robot-init-y ${robot_init_y} ${robot_init_y} 1        --obj-variation-mode episode        --obj-episode-range 0 24        --robot-init-rot-quat-center 0 0 0 1        --robot-init-rot-rpy-range 0 0 1 0 0 1 0 0 1
    run_count=$((run_count + 1))
  done
done
declare -a ENV_NAMES_V2=(
  PutEggplantInBasketScene-v0
)
scene_name=bridge_table_1_v2
robot=widowx_sink_camera_setup
rgb_overlay_path="${SimplerEnv_PATH}/ManiSkill2_real2sim/data/real_inpainting/bridge_sink.png"
robot_init_x=0.127
robot_init_y=0.06
for i in "${!ENV_NAMES_V2[@]}"; do
  env="${ENV_NAMES_V2[i]}"
  for ((run_idx=1; run_idx<=TSET_NUM; run_idx++)); do
:
:
:
    run_case --ckpt-path "${ckpt_path}"        --port $port        --robot ${robot}        --policy-setup widowx_bridge        --control-freq 5        --sim-freq 500        --max-episode-steps 120        --env-name "${env}"        --scene-name ${scene_name}        --rgb-overlay-path "${rgb_overlay_path}"        --robot-init-x ${robot_init_x} ${robot_init_x} 1        --robot-init-y ${robot_init_y} ${robot_init_y} 1        --obj-variation-mode episode        --obj-episode-range 0 24        --robot-init-rot-quat-center 0 0 0 1        --robot-init-rot-rpy-range 0 0 1 0 0 1 0 0 1
:
    run_count=$((run_count + 1))
  done
done
