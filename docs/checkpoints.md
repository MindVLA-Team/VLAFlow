# Checkpoints and migration

Keep the run layout when moving weights:

```text
run/
  config.full.yaml
  dataset_statistics.json
  checkpoints/steps_200000_pytorch_model.pt
```

The public framework names remain `MindPI`, `MindWPI`, and `MindLWPI_Compressed`. Mainline module names within state dictionaries, including `qwen_vl_interface`, `action_model`, and `latent_extractor`, are preserved. Python imports and trainer filenames now use `vlaflow`; there is no old-package import alias.

To reuse a matching old checkpoint, copy its configuration and statistics with the weights. Update local backbone paths, data/output paths, and obsolete package/training references in the copied configuration. Preserve architecture-defining dimensions and normalization. For MindLWPI, verify `latent_compress_method=avgpool` and `latent_compress_ratio=4`; uncompressed, MLP, k16, MoE, and multi-frame variants are not supported by this release.

`framework.use_state=false` follows the paper. Set it to `true` only when a compatible older model was intentionally trained with state inputs, and supply matching observations at inference. Parameter dimensions must still match. Disable `enable_lang_action` for downstream MindLWPI use; prediction never generates action-description text.

The policy server loads weights strictly. Missing or unexpected weights and incompatible dimensions are errors, not a successful migration. Correcting public recipe defaults does not change the hyperparameters with which existing weights were trained.
