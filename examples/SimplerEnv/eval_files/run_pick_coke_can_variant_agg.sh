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

task_name=pick_coke_can_variant_agg
declare -a arr=("${MODEL_PATH}")
run_count=0
declare -a coke_can_options_arr=("lr_switch=True" "upright=True" "laid_vertically=True")
for ckpt_path in "${arr[@]}"; do
:
done
env_name=GraspSingleOpenedCokeCanInScene-v0
scene_name=google_pick_coke_can_1_v4
for coke_can_option in "${coke_can_options_arr[@]}"; do
  for ckpt_path in "${arr[@]}"; do
    run_case --ckpt-path "${ckpt_path}"        --robot google_robot_static        --port ${port}        --control-freq 3 --sim-freq 513 --max-episode-steps 80        --env-name ${env_name} --scene-name ${scene_name}        --robot-init-x 0.35 0.35 1 --robot-init-y 0.20 0.20 1 --obj-init-x -0.35 -0.12 5 --obj-init-y -0.02 0.42 5        --robot-init-rot-quat-center 0 0 0 1 --robot-init-rot-rpy-range 0 0 1 0 0 1 0 0 1        --additional-env-build-kwargs ${coke_can_option}
    run_count=$((run_count + 1))
  done
done
env_name=GraspSingleOpenedCokeCanInScene-v0
declare -a scene_arr=("Baked_sc1_staging_objaverse_cabinet1_h870"                        "Baked_sc1_staging_objaverse_cabinet2_h870")
for coke_can_option in "${coke_can_options_arr[@]}"; do
  for scene_name in "${scene_arr[@]}"; do
    for ckpt_path in "${arr[@]}"; do
    run_case --ckpt-path "${ckpt_path}"          --robot google_robot_static          --port ${port}          --control-freq 3 --sim-freq 513 --max-episode-steps 80          --env-name ${env_name} --scene-name ${scene_name}          --robot-init-x 0.35 0.35 1 --robot-init-y 0.20 0.20 1 --obj-init-x -0.35 -0.12 5 --obj-init-y -0.02 0.42 5          --robot-init-rot-quat-center 0 0 0 1 --robot-init-rot-rpy-range 0 0 1 0 0 1 0 0 1          --additional-env-build-kwargs ${coke_can_option}
      run_count=$((run_count + 1))
    done
  done
done
env_name=GraspSingleOpenedCokeCanDistractorInScene-v0
scene_name=google_pick_coke_can_1_v4
for coke_can_option in "${coke_can_options_arr[@]}"; do
  for ckpt_path in "${arr[@]}"; do
    run_case --ckpt-path "${ckpt_path}"        --robot google_robot_static        --port ${port}        --control-freq 3 --sim-freq 513 --max-episode-steps 80        --env-name ${env_name} --scene-name ${scene_name}        --robot-init-x 0.35 0.35 1 --robot-init-y 0.20 0.20 1 --obj-init-x -0.35 -0.12 5 --obj-init-y -0.02 0.42 5        --robot-init-rot-quat-center 0 0 0 1 --robot-init-rot-rpy-range 0 0 1 0 0 1 0 0 1        --additional-env-build-kwargs ${coke_can_option}
    run_count=$((run_count + 1))
    run_case --ckpt-path "${ckpt_path}"        --robot google_robot_static        --port ${port}        --control-freq 3 --sim-freq 513 --max-episode-steps 80        --env-name ${env_name} --scene-name ${scene_name}        --robot-init-x 0.35 0.35 1 --robot-init-y 0.20 0.20 1 --obj-init-x -0.35 -0.12 5 --obj-init-y -0.02 0.42 5        --robot-init-rot-quat-center 0 0 0 1 --robot-init-rot-rpy-range 0 0 1 0 0 1 0 0 1        --additional-env-build-kwargs ${coke_can_option} distractor_config=more
    run_count=$((run_count + 1))
  done
done
env_name=GraspSingleOpenedCokeCanInScene-v0
declare -a bg_scene_arr=("google_pick_coke_can_1_v4_alt_background"                           "google_pick_coke_can_1_v4_alt_background_2")
for coke_can_option in "${coke_can_options_arr[@]}"; do
  for scene_name in "${bg_scene_arr[@]}"; do
    for ckpt_path in "${arr[@]}"; do
    run_case --ckpt-path "${ckpt_path}"          --robot google_robot_static          --port ${port}          --control-freq 3 --sim-freq 513 --max-episode-steps 80          --env-name ${env_name} --scene-name ${scene_name}          --robot-init-x 0.35 0.35 1 --robot-init-y 0.20 0.20 1 --obj-init-x -0.35 -0.12 5 --obj-init-y -0.02 0.42 5          --robot-init-rot-quat-center 0 0 0 1 --robot-init-rot-rpy-range 0 0 1 0 0 1 0 0 1          --additional-env-build-kwargs ${coke_can_option}
      run_count=$((run_count + 1))
    done
  done
done
env_name=GraspSingleOpenedCokeCanInScene-v0
scene_name=google_pick_coke_can_1_v4
for coke_can_option in "${coke_can_options_arr[@]}"; do
  for ckpt_path in "${arr[@]}"; do
    run_case --ckpt-path "${ckpt_path}"        --robot google_robot_static        --port ${port}        --control-freq 3 --sim-freq 513 --max-episode-steps 80        --env-name ${env_name} --scene-name ${scene_name}        --robot-init-x 0.35 0.35 1 --robot-init-y 0.20 0.20 1 --obj-init-x -0.35 -0.12 5 --obj-init-y -0.02 0.42 5        --robot-init-rot-quat-center 0 0 0 1 --robot-init-rot-rpy-range 0 0 1 0 0 1 0 0 1        --additional-env-build-kwargs ${coke_can_option} slightly_darker_lighting=True
    run_count=$((run_count + 1))
    run_case --ckpt-path "${ckpt_path}"        --robot google_robot_static        --port ${port}        --control-freq 3 --sim-freq 513 --max-episode-steps 80        --env-name ${env_name} --scene-name ${scene_name}        --robot-init-x 0.35 0.35 1 --robot-init-y 0.20 0.20 1 --obj-init-x -0.35 -0.12 5 --obj-init-y -0.02 0.42 5        --robot-init-rot-quat-center 0 0 0 1 --robot-init-rot-rpy-range 0 0 1 0 0 1 0 0 1        --additional-env-build-kwargs ${coke_can_option} slightly_brighter_lighting=True
    run_count=$((run_count + 1))
  done
done
declare -a env_arr=("GraspSingleOpenedCokeCanAltGoogleCameraInScene-v0"                      "GraspSingleOpenedCokeCanAltGoogleCamera2InScene-v0")
scene_name=google_pick_coke_can_1_v4
for coke_can_option in "${coke_can_options_arr[@]}"; do
  for env_name in "${env_arr[@]}"; do
    for ckpt_path in "${arr[@]}"; do
    run_case --ckpt-path "${ckpt_path}"          --robot google_robot_static          --port ${port}          --control-freq 3 --sim-freq 513 --max-episode-steps 80          --env-name ${env_name} --scene-name ${scene_name}          --robot-init-x 0.35 0.35 1 --robot-init-y 0.20 0.20 1 --obj-init-x -0.35 -0.12 5 --obj-init-y -0.02 0.42 5          --robot-init-rot-quat-center 0 0 0 1 --robot-init-rot-rpy-range 0 0 1 0 0 1 0 0 1          --additional-env-build-kwargs ${coke_can_option}
      run_count=$((run_count + 1))
    done
  done
done
