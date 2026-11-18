# Article recovery experiment

These Python files and results are copied from the completed recovery archive.
The entry point for the article's 48-run comparison is
`stage5_loss_comparison.py`, not the historical Stage 5/6 scripts in
`../legacy_recovery/`.

## Existing command interface

Run commands from this folder, in a TensorFlow environment:

```bash
python stage5_loss_comparison.py
```

Without flags this runs preflight only: it loads the preserved split, creates
a model and instantiates optimizers. It does not fit a model.

The existing training and final-test commands are `--screen` and `--final`.
Do not run them to review the saved tables: `--final` can overwrite the original
table, and it requires the omitted trained weights. A new experiment should use
a separate checkout/output copy. The screen skips existing result.json records,
so existing records without their weights are not a fresh-training workspace.

For a deliberately new reproduction in a separate copy with no prior Stage 5
results, the original command is:

```bash
python stage5_loss_comparison.py --screen
```

After all twelve rates are frozen and the corresponding weights exist, the
original final-evaluation command is:

```bash
python stage5_loss_comparison.py --final
```

`results/stage3_cross_validation/data_splits.npz` is included for the saved
425 development IDs and Fold 1's 340/85 split. IDs 500-574 define the fresh test
noise realizations. Source code and records have not been changed in packaging.

## Files to review

- `experiment_config.py`: protocol, optimizers, losses and split loader.
- `stage2_unet.py`: compact U-Net architecture.
- `stage4_lr_sensitivity.py`: model/data/preflight helpers used by Stage 5.
- `stage5_loss_comparison.py`: rate screening, selection and final metrics.
- `results/stage5_loss_comparison/final_results.csv`: 12 selected test comparisons.
- `results/stage5_loss_comparison/noisy_baseline.json`: baseline on fresh test IDs.

Stage 1/3 scripts and images are historical pilot material. Some pilot scripts
execute work at import, so they must not be imported for a harmless check.
The Adam/RMSProp/plain-SGD recovery figures in the pilot folder do not depict
the revised twelve-model comparison. Numerical SSIM in the revised Stage 5
uses `tf.image.ssim`; do not substitute the historical scikit-image calculation.
