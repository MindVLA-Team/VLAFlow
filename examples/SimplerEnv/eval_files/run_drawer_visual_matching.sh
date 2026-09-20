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

task_name=drawer_visual_matching
declare -a ckpt_paths=(
  "${MODEL_PATH}"
)
declare -a env_names=(
  OpenTopDrawerCustomInScene-v0
  OpenMiddleDrawerCustomInScene-v0
  OpenBottomDrawerCustomInScene-v0
  CloseTopDrawerCustomInScene-v0
  CloseMiddleDrawerCustomInScene-v0
  CloseBottomDrawerCustomInScene-v0
)
declare -a urdf_version_arr=("recolor_cabinet_visual_matching_1" "recolor_tabletop_visual_matching_1" "recolor_tabletop_visual_matching_2" None)
run_count=0
for urdf_version in "${urdf_version_arr[@]}"; do
  EXTRA_ARGS="--enable-raytracing --additional-env-build-kwargs station_name=mk_station_recolor light_mode=simple disable_bad_material=True urdf_version=${urdf_version}"
  EvalSim() {
    run_case --ckpt-path "${ckpt_path}"        --robot google_robot_static        --port $port        --control-freq 3 --sim-freq 513 --max-episode-steps 113        --env-name ${env_name} --scene-name dummy_drawer        --robot-init-x 0.644 0.644 1 --robot-init-y -0.179 -0.179 1        --robot-init-rot-quat-center 0 0 0 1 --robot-init-rot-rpy-range 0 0 1 0 0 1 -0.03 -0.03 1        --obj-init-x-range 0 0 1 --obj-init-y-range 0 0 1        --rgb-overlay-path "${SimplerEnv_PATH}/ManiSkill2_real2sim/data/real_inpainting/open_drawer_a0.png"        ${EXTRA_ARGS}
    run_case --ckpt-path "${ckpt_path}"        --robot google_robot_static        --port $port        --control-freq 3 --sim-freq 513 --max-episode-steps 113        --env-name ${env_name} --scene-name dummy_drawer        --robot-init-x 0.765 0.765 1 --robot-init-y -0.182 -0.182 1        --robot-init-rot-quat-center 0 0 0 1 --robot-init-rot-rpy-range 0 0 1 0 0 1 -0.02 -0.02 1        --obj-init-x-range 0 0 1 --obj-init-y-range 0 0 1        --rgb-overlay-path "${SimplerEnv_PATH}/ManiSkill2_real2sim/data/real_inpainting/open_drawer_a1.png"        ${EXTRA_ARGS}
    run_case --ckpt-path "${ckpt_path}"        --robot google_robot_static        --port $port        --control-freq 3 --sim-freq 513 --max-episode-steps 113        --env-name ${env_name} --scene-name dummy_drawer        --robot-init-x 0.889 0.889 1 --robot-init-y -0.203 -0.203 1        --robot-init-rot-quat-center 0 0 0 1 --robot-init-rot-rpy-range 0 0 1 0 0 1 -0.06 -0.06 1        --obj-init-x-range 0 0 1 --obj-init-y-range 0 0 1        --rgb-overlay-path "${SimplerEnv_PATH}/ManiSkill2_real2sim/data/real_inpainting/open_drawer_a2.png"        ${EXTRA_ARGS}
    run_case --ckpt-path "${ckpt_path}"        --robot google_robot_static        --port $port        --control-freq 3 --sim-freq 513 --max-episode-steps 113        --env-name ${env_name} --scene-name dummy_drawer        --robot-init-x 0.652 0.652 1 --robot-init-y 0.009 0.009 1        --robot-init-rot-quat-center 0 0 0 1 --robot-init-rot-rpy-range 0 0 1 0 0 1 0 0 1        --obj-init-x-range 0 0 1 --obj-init-y-range 0 0 1        --rgb-overlay-path "${SimplerEnv_PATH}/ManiSkill2_real2sim/data/real_inpainting/open_drawer_b0.png"        ${EXTRA_ARGS}
    run_case --ckpt-path "${ckpt_path}"        --robot google_robot_static        --port $port        --control-freq 3 --sim-freq 513 --max-episode-steps 113        --env-name ${env_name} --scene-name dummy_drawer        --robot-init-x 0.752 0.752 1 --robot-init-y 0.009 0.009 1        --robot-init-rot-quat-center 0 0 0 1 --robot-init-rot-rpy-range 0 0 1 0 0 1 0 0 1        --obj-init-x-range 0 0 1 --obj-init-y-range 0 0 1        --rgb-overlay-path "${SimplerEnv_PATH}/ManiSkill2_real2sim/data/real_inpainting/open_drawer_b1.png"        ${EXTRA_ARGS}
    run_case --ckpt-path "${ckpt_path}"        --robot google_robot_static        --port $port        --control-freq 3 --sim-freq 513 --max-episode-steps 113        --env-name ${env_name} --scene-name dummy_drawer        --robot-init-x 0.851 0.851 1 --robot-init-y 0.035 0.035 1        --robot-init-rot-quat-center 0 0 0 1 --robot-init-rot-rpy-range 0 0 1 0 0 1 0 0 1        --obj-init-x-range 0 0 1 --obj-init-y-range 0 0 1        --rgb-overlay-path "${SimplerEnv_PATH}/ManiSkill2_real2sim/data/real_inpainting/open_drawer_b2.png"        ${EXTRA_ARGS}
    run_case --ckpt-path "${ckpt_path}"        --robot google_robot_static        --port $port        --control-freq 3 --sim-freq 513 --max-episode-steps 113        --env-name ${env_name} --scene-name dummy_drawer        --robot-init-x 0.665 0.665 1 --robot-init-y 0.224 0.224 1        --robot-init-rot-quat-center 0 0 0 1 --robot-init-rot-rpy-range 0 0 1 0 0 1 0 0 1        --obj-init-x-range 0 0 1 --obj-init-y-range 0 0 1        --rgb-overlay-path "${SimplerEnv_PATH}/ManiSkill2_real2sim/data/real_inpainting/open_drawer_c0.png"        ${EXTRA_ARGS}
    run_case --ckpt-path "${ckpt_path}"        --robot google_robot_static        --port $port        --control-freq 3 --sim-freq 513 --max-episode-steps 113        --env-name ${env_name} --scene-name dummy_drawer        --robot-init-x 0.765 0.765 1 --robot-init-y 0.222 0.222 1        --robot-init-rot-quat-center 0 0 0 1 --robot-init-rot-rpy-range 0 0 1 0 0 1 -0.025 -0.025 1        --obj-init-x-range 0 0 1 --obj-init-y-range 0 0 1        --rgb-overlay-path "${SimplerEnv_PATH}/ManiSkill2_real2sim/data/real_inpainting/open_drawer_c1.png"        ${EXTRA_ARGS}
    run_case --ckpt-path "${ckpt_path}"        --robot google_robot_static        --port $port        --control-freq 3 --sim-freq 513 --max-episode-steps 113        --env-name ${env_name} --scene-name dummy_drawer        --robot-init-x 0.865 0.865 1 --robot-init-y 0.222 0.222 1        --robot-init-rot-quat-center 0 0 0 1 --robot-init-rot-rpy-range 0 0 1 0 0 1 -0.025 -0.025 1        --obj-init-x-range 0 0 1 --obj-init-y-range 0 0 1        --rgb-overlay-path "${SimplerEnv_PATH}/ManiSkill2_real2sim/data/real_inpainting/open_drawer_c2.png"        ${EXTRA_ARGS}
  }
  for ckpt_path in "${ckpt_paths[@]}"; do
    for env_name in "${env_names[@]}"; do
      if (( (run_count + 1) % 32 == 0 )); then
        EvalSim
      else
        EvalSim
      fi
      run_count=$((run_count + 1))
    done
  done
done
