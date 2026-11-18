# Coding and experiment instructions

- Read `PROJECT_CONTEXT.md` before substantial coding or experimental changes; treat it as the authoritative project summary.
- Inspect existing scripts, data, and saved splits first. Preserve pilot scripts, datasets, and results unless the user explicitly authorizes changes.
- Put revised experiments in new files, beginning with `experiment_config.py` and `stage4_lr_sensitivity.py`. Reuse the existing compact U-Net where appropriate.
- Keep the 75 held-out test samples untouched during all tuning. Select learning rates using validation results, freeze them, then perform the main comparison.
- Reuse saved development/test splits and identical cross-validation folds. Use seed 42, consistent training controls, and the same initial weights where possible.
- Save configurations, histories, metrics, timing, checkpoints, and run status reproducibly in separate experiment output directories. Record failures/divergence without losing completed runs.
- Explain the purpose of major changes before implementation. Document unresolved methodological choices before execution; never invent project decisions or results.
- Do not launch long training runs merely to validate context files or setup changes. Run checks appropriate to the actual coding task and report what was verified.
