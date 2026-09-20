# Data preparation

Use LeRobot-format datasets with `meta/modality.json`. Data transforms preserve the original action conventions, normalization, masks, and dataset exclusions; this cleanup does not redefine coordinate systems or action units.

## Pretraining layout

Set `DATA_ROOT` to a parent containing:

```text
data/
  openx-embodiment-lerobot/  # raw OXE datasets, including droid
  OXE_AugE_v2/              # augmented datasets converted to LeRobot v2.1
  RoboCOIN_SUB/             # retained RoboCOIN embodiments
  oxemix_global_norm_stats.json
```

Dataset names and embodiment mappings are defined under each source's `train_files/data_registry/`. OXE_Mix imports all three source registries. Source directories are discovered through `DATA_ROOT`; the source registries are needed even though they have no standalone training script.

After obtaining the datasets, generate their modality metadata:

```bash
python examples/OXE/gen_oxe_modality_json.py --oxe_root "$DATA_ROOT/openx-embodiment-lerobot"
python scripts/gen_robocoin_modality.py --root "$DATA_ROOT/RoboCOIN_SUB"
python scripts/gen_oxeauge_modality.py --root "$DATA_ROOT/OXE_AugE"
python scripts/convert_oxeauge_v30_to_v21.py \
  --input "$DATA_ROOT/OXE_AugE" --output "$DATA_ROOT/OXE_AugE_v2"
```

The converter needs `ffmpeg`; it extracts dataset videos as training inputs, not visualization exports. Review its conversion summary and execute the generated linking script for already-v2.1 datasets if present. If the augmented data is already converted, generate modality metadata directly under `OXE_AugE_v2`.

MindLWPI natural-language action descriptions require global quantiles from the same mixture:

```bash
python scripts/precompute_oxemix_global_stats.py \
  --data_root_dir "$DATA_ROOT" --data_mix oxe_mix_full \
  --min_loaded_fraction 1 --output_path "$DATA_ROOT/oxemix_global_norm_stats.json"
```

Generate statistics from your actual dataset mixture; no experiment-specific reference statistics are bundled. Global language-description quantiles are distinct from the per-dataset statistics saved for policy action denormalization.

## Downstream data

For LIBERO, place the four LeRobot suites under `DATA_ROOT` using the names in its registry and install the provided `train_files/modality.json` in each dataset's `meta/` directory.

For SimplerEnv, the 14-D recipes expect `ipec_bridge_orig/` and `ipec_fractal20220817_data_lerobot/`. Use the Bridge and Fractal modality files in `examples/SimplerEnv/train_files/`. Train Bridge and RT-1 separately for the paper's main evaluation protocol.

The loaders map native actions to 14 dimensions and carry validity masks. Evaluation extracts the right-arm 7-D slice before denormalization. Preserve the configured mappings and action representation when preparing new data.
