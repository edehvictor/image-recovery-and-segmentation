# Repository instructions

Read README.md, PROJECT_CONTEXT.md and docs/PROVENANCE.md before changes.
Preserve imported result records and experimental source; do not rewrite results
or run training/evaluation as a cleanup check. New summaries go under reports/.
The primary article recovery workflow is recovery/stage5_loss_comparison.py.
The separate historical Stage 4/5/6 workflow is under legacy_recovery/.
Do not merge their selection rules or describe the unfinished Stage 6 as complete.
Never use held-out test scores to choose learning rates or other settings.
Keep data, logs, environments, weights and full prediction archives out of Git.
Record assistance and verification in AI_USAGE_LOG.md. Do not claim independent
human verification unless the person has actually confirmed it.
