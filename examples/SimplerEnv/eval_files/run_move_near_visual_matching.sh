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

task_name=move_near_visual_matching
declare -a arr=(
  "${MODEL_PATH}"
)
env_name=MoveNearGoogleBakedTexInScene-v0
scene_name=google_pick_coke_can_1_v4
rgb_overlay_path="${SimplerEnv_PATH}/ManiSkill2_real2sim/data/real_inpainting/google_move_near_real_eval_1.png"
declare -a urdf_version_arr=(None "recolor_tabletop_visual_matching_1" "recolor_tabletop_visual_matching_2" "recolor_cabinet_visual_matching_1")
run_count=0
for ckpt_path in "${arr[@]}"; do
:
done
for urdf_version in "${urdf_version_arr[@]}"; do
  for ckpt_path in "${arr[@]}"; do
    run_case --ckpt-path "${ckpt_path}"        --robot google_robot_static        --port $port        --control-freq 3 --sim-freq 513 --max-episode-steps 80        --env-name ${env_name} --scene-name ${scene_name}        --rgb-overlay-path "${rgb_overlay_path}"        --robot-init-x 0.35 0.35 1 --robot-init-y 0.21 0.21 1 --obj-variation-mode episode --obj-episode-range 0 60        --robot-init-rot-quat-center 0 0 0 1 --robot-init-rot-rpy-range 0 0 1 0 0 1 -0.09 -0.09 1        --additional-env-build-kwargs urdf_version=${urdf_version}        --additional-env-save-tags baked_except_bpb_orange
    run_count=$((run_count + 1))
  done
done
