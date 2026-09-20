# Evaluation

Run the policy server in the VLAFlow environment and the client in the benchmark's own environment. Install LIBERO, LIBERO-plus, or SimplerEnv according to that benchmark's instructions. Keep their environments separate. Clients need NumPy, OpenCV, PyYAML, msgpack, and WebSockets; LIBERO clients also need Tyro, and SimplerEnv needs transforms3d.

## Policy service

Keep the fine-tuned weights beside the run configuration and dataset statistics described in [Checkpoints](checkpoints.md).

```bash
export PRETRAINED_CHECKPOINT=/path/to/run/checkpoints/steps_100000_pytorch_model.pt
export POLICY_PYTHON=/path/to/vlaflow/environment/bin/python
export POLICY_PORT=10093
bash examples/LIBERO/eval_files/run_policy_server.sh
```

Each retained benchmark provides the same `run_policy_server.sh` wrapper. Select GPUs with `CUDA_VISIBLE_DEVICES`; scripts do not assign devices, submit jobs, or start background clusters.

## LIBERO and LIBERO-plus

In another terminal, set `SIM_PYTHON` to the relevant simulator interpreter. Set `LIBERO_HOME` to its checkout (required for LIBERO-plus perturbation metadata), and configure LIBERO's asset paths using the simulator's installation procedure.

```bash
export SIM_PYTHON=/path/to/libero/environment/bin/python
export LIBERO_HOME=/path/to/LIBERO
export PRETRAINED_CHECKPOINT=/path/to/run/checkpoints/steps_100000_pytorch_model.pt
export OUTPUT_ROOT=/path/to/evaluation
bash examples/LIBERO/eval_files/run_eval.sh
```

The standard entry evaluates Spatial, Object, Goal, and Long with 50 trials per task. `TASK_SUITES` and `TRIALS_PER_TASK` can reduce the workload for a smoke check.

For LIBERO-plus, switch to its simulator environment and checkout, then run `examples/LIBERO-plus/eval_files/run_eval.sh`. It evaluates the same LIBERO 100k checkpoint without adaptation, using one rollout per perturbed task by default. The evaluator saves suite totals and perturbation-category counts as JSON. No videos or action plots are written.

## SimplerEnv

Start a policy server using the appropriate Bridge-only or RT-1-only fine-tuned checkpoint. Set `SIMPLER_ROOT` to the installed simulator checkout, `SIM_PYTHON` to its interpreter, and `POLICY_HOST` / `POLICY_PORT` to the running service.

```bash
export SIMPLER_ROOT=/path/to/SimplerEnv
export SIM_PYTHON=/path/to/simpler/environment/bin/python
bash examples/SimplerEnv/eval_files/run_bridge.sh
```

For RT-1, run the four task families `pick_coke_can`, `move_near`, `drawer`, and `put_in_drawer`. Each has `run_<task>_visual_matching.sh` and `run_<task>_variant_agg.sh`. These retain the original task/scenario/perturbation matrices while executing cases sequentially against one server. Simulator overlays remain external simulator inputs; they are not bundled visualization assets.

Use `DRY_RUN=1` to inspect all commands without starting environments. Results are written to `${OUTPUT_ROOT}/simpler/<protocol>/case_<index>/metrics.json`. The evaluator preserves simulator stepping and control modes but removes video and action-trajectory recording.

```bash
python scripts/summarize_evaluation.py /path/to/evaluation/simpler/pick_coke_can_visual_matching
```

The summary reports pooled successful episodes and the mean per-case success rate. Keep protocols separate; pooling VM, VA, different robots, or incomplete task suites does not yield the paper's aggregate score. Complete simulation runs and paper-score reproduction require the external datasets, assets, weights, and benchmark environments.
