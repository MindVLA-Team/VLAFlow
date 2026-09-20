# Training

Run scripts locate the checkout independently of the current working directory. The public model names are `MindPI`, `MindWPI`, and `MindLWPI_Compressed`.

## Recipe matrix

Replace `<model>` with `mindpi`, `mindwpi`, or `mindlwpi_compressed`:

| Stage | Shell entry |
| --- | --- |
| OXEMix pretraining | `examples/OXE_Mix/train_files/run_<model>_oxemix_pretrain.sh` |
| LIBERO fine-tuning | `examples/LIBERO/train_files/run_<model>_libero_finetune.sh` |
| Bridge-only fine-tuning | `examples/SimplerEnv/train_files/run_<model>_bridge_finetune.sh` |
| RT-1-only fine-tuning | `examples/SimplerEnv/train_files/run_<model>_rt1_finetune.sh` |

Pretraining takes no robot checkpoint. Fine-tuning requires `PRETRAINED_CHECKPOINT`. Use `--mode frozen-vlm` only for MindPI pretraining; use `--mode no-pretrain` for the MindPI/MindWPI downstream baselines, with `PRETRAINED_CHECKPOINT` unset. There is no extra training recipe for LIBERO-plus.

## Resources and overrides

Set `DATA_ROOT`, `MODEL_ROOT`, and `OUTPUT_ROOT`. For MindLWPI language supervision, set `GLOBAL_NORM_STATS` or place the statistics at `${DATA_ROOT}/oxemix_global_norm_stats.json`. `RUN_ID` overrides the output subdirectory. Existing output directories should be used only for intentionally matching runs.

`--dry-run` prints the merged YAML, Accelerate command, and effective global batch without loading models or writing outputs. `--set trainer.max_train_steps=10` changes an existing configuration key; unknown keys fail. Example:

```bash
bash examples/OXE_Mix/train_files/run_mindpi_oxemix_pretrain.sh \
  --mode frozen-vlm --set datasets.vla_data.data_mix=oxe_original --dry-run
```

For latent models, use `oxe_original_wm` to select the OXE raw subset, including DROID. Full OXEMix rejects empty sources and dataset loading failures. Full OXEMix uses `oxe_mix_full` / `oxe_mix_full_wm` and requires every source.

Distributed launch accepts `NUM_NODES` (default 1), `GPUS_PER_NODE` (default 1), `NODE_RANK` (default 0), `MASTER_ADDR`, and `MASTER_PORT` (default 29500). Start the same command on each node with its own rank and a reachable master address. The repository contains no cluster submission integration.

`GRADIENT_ACCUMULATION_STEPS` defaults to the recipe value, 1. Global batch = nodes × GPUs per node × per-device batch × accumulation. Resource changes are printed; the launcher does not silently adjust the experiment budget to match the paper.

## Paper defaults

All recipes use Qwen3-VL-4B, a 36-layer action expert of width 1280, 14-D actions, 16-step action chunks, four noise samples per training example, and four Euler inference steps. State tokens are disabled with `framework.use_state=false`; encoder parameters remain available for compatible checkpoints.

| Setting | Pretraining | Fine-tuning |
| --- | --- | --- |
| Steps / warmup | 200,000 / 5,000 | 100,000 / 2,000 |
| VLM / action learning rate | 1e-5 / 1e-4 | 1e-5 / 1e-4 |
| Minimum learning rate | 5e-7 | 5e-7 |
| Latent:action loss, when enabled | 1:1 | 0.1:1 |
| MindLWPI language weight | 0.1 | disabled |
| Per-GPU batch | 8 | 16 |
| Paper global batch | 512 | LIBERO 128; SimplerEnv 256 |

Training uses AdamW, BF16, VLM checkpointing that preserves KV outputs, clipping at 1.0, and ZeRO-2. MindWPI uses uncompressed V-JEPA 2 tokens; MindLWPI uses AvgPool-k4 on both present tokens and future targets. Future-frame offset is 8. MindLWPI action/latent gradients also update the VLM.

The optimizer weight decay retains the implementation value of 1e-8, which is not specified in the paper. Earlier scripts sometimes used other compression factors, loss weights, learning rates, or step budgets. These are not the released defaults; inspect original checkpoint configuration rather than inferring it from a directory name.

## Outputs

Each run saves resolved `config.full.yaml`, dataset normalization statistics, `metrics.jsonl`, and periodic weights under `checkpoints/`. `config.yaml` may contain an access-tracked subset. Keep the full configuration and statistics together with weights. Normal training does not require external experiment tracking or produce plots/videos.
