# AI usage log

| Date | Tool | Purpose and assistance | Verification status |
| --- | --- | --- | --- |
| 6 October 2026 | OpenAI Codex | Read repository context/review; document the supervisor-directed replacement of the exposed pilot holdout; prepare a fixed experiment configuration and a guarded 16-configuration LR-screening runner. | Agent verified syntax, safe imports of the new modules, saved 425 development IDs, fold 1's 340/85 partition, exclusion of fresh test IDs 500–574, and pure utility/guard behavior. TensorFlow import failed because `tensorflow.python` is missing in `venv313`; optimizer instantiation and U-Net construction remain unverified. No training or test-image evaluation was run. |

Independent student verification of the new code and computational results is pending. This entry records assistance and does not claim experimental findings or student approval to begin training. Pilot scripts and pilot outputs were preserved.


## 8 October 2026 supervisor submission

OpenAI Codex assisted with comments/docstrings in the eight requested Python files, manuscript revision and mathematical typesetting, primary-source convergence-reference checking, image organization and ZIP preparation. Saved numerical experiment records were reused without rerunning training or evaluation. Agent verification: syntax compilation, executable-AST comparison excluding docstrings, matching source copies, duplicate-image hashes, and SHA-256 preservation of all 112 saved result files. Student independent verification of this manuscript revision has not been asserted. The original proposal's recovery and segmentation scope is retained; unfinished comparisons and segmentation computations are labelled outstanding. No GitHub push was made.


## 9 October 2026 repository submission archive

OpenAI Codex audited Git status and confirmed that commit ddf5079 already contains the eight source annotations and the 8 October assistance record. The previously untracked dated submission folder is being committed as an immutable submission snapshot, including the revised DOCX/PDF article, five distinct segmentation references, saved recovery figures, supporting result copies and submission notes. Generated submission ZIPs are ignored because the folder records their contents. Agent repeated syntax-only compilation, executable-AST comparison excluding docstrings, matching code-copy checks, and SHA-256 checks for all 112 existing saved result files; all passed. No training, model inference, noise generation or new experimental evaluation was performed. Independent student verification of this repository audit is not asserted.


## 10 October 2026 Stage 5 MAE/Huber screening preparation

OpenAI Codex created `stage5_loss_lr_screening.py`, reusing the guarded Stage 4 model/data/hash helpers and unchanged experiment_config constructors. Assistance covered pair-specific MAE/Huber rate selection, strict raw-minimum model restoration independent of EarlyStopping, common recovery metrics, fit-only timing, divergence recording, atomic per-attempt outputs and validated restart-from-epoch-1 resume. PROJECT_CONTEXT.md now records completed MSE screening, the established pair-specific rate methodology and the Stage 5 restoration override.

Agent verification: syntax/source checks and actual `--check-only` execution in `/home/devnomso/fyp-tensorflow/bin/python` (Python 3.13.16, TensorFlow 2.21.0, Keras 3.15.1, NumPy 2.5.3, scikit-image 0.26.0) passed. Verified 425 development IDs, disjoint 340/85 Fold 1 partition, reserved IDs 500-574, 32 instantiated optimizer-loss-LR configurations, deterministic operations, stopping/raw-best callback construction, and one 120681-trainable-parameter U-Net with matching input/output shape. Canonical initialization SHA-256: `3d8af7fb3e7d315c2358a71bef8213afa7e6e29217099ad230fe89951a531926`. Final checks compare all 112 saved result files plus eight existing source/config files against the pre-change hash baseline.

No model.fit, inference, new metric evaluation, noisy-image generation, MSE screening rerun, final-test evaluation, segmentation computation or GitHub push occurred. Stage 5 training and completed-run resumption were not executed; the corresponding logic was inspected in source only. Student independent verification of this implementation is not asserted. No experimental results were invented.
