# Stage 4 preflight report

Date: 6 October 2026  
Repository: `C:\Users\HP ELITEBOOK 830 G8\Desktop\FINAL YEAR PROJECT\image-recovery-project`

**Final recommendation: NOT READY to authorize the 16-run screening from this report.** The saved-split and source checks passed, but the requested WSL runtime preflight could not be launched by this chat. Optimizer instantiation and one fresh U-Net build in that interpreter remain unverified here. This is an execution-access limitation, not evidence of a TensorFlow or Stage 4 code failure.

**No training occurred.** No `model.fit()`, evaluation, inference, noisy-image generation, or pilot runner was invoked by the agent. No Stage 4 results directory was created.

## 1. Interpreters and evidence

Requested interpreter:

```text
/home/devnomso/fyp-tensorflow/bin/python
```

The agent attempted to launch that exact interpreter through Windows `wsl.exe --exec`. WSL returned exit code 1 before Python started:

```text
Access is denied.
Error code: Wsl/E_ACCESSDENIED
```

The same error persisted after a turn-scoped read grant for the WSL environment. That grant enabled filesystem inspection through `\\wsl.localhost\Ubuntu\home\devnomso\fyp-tensorflow`; it did not enable WSL execution. Neither a WSL `sys.executable` result nor successful Stage 4 runtime output was obtained. The user's acknowledgements to requests for terminal output contained no execution log.

For independent, lightweight import, array, guard, and syntax/source checks, the agent actually used:

```text
C:\Users\HP ELITEBOOK 830 G8\Desktop\FINAL YEAR PROJECT\image-recovery-project\venv313\Scripts\python.exe
```

That supplementary interpreter reports Python **3.13.15** and NumPy **2.5.3**. It was invoked with `-B`, without importing TensorFlow. These supplementary checks are not presented as execution in the requested Linux interpreter.

| Environment item | Value | Evidence |
| --- | --- | --- |
| WSL | WSL 2 | User supplied; runtime unavailable to agent |
| Linux distribution | Ubuntu 26.04 | User supplied; runtime not verified |
| Python in WSL | 3.13.16 | User supplied; virtual-environment configuration confirms CPython 3.13, not the patch version |
| TensorFlow | 2.21.0 | User supplied and installed WSL package METADATA read |
| NumPy | 2.5.3 | User supplied and installed WSL package METADATA read |
| scikit-learn | 1.9.1 | User supplied and installed WSL package METADATA read |
| scikit-image | 0.26.0 | User supplied and installed WSL package METADATA read |
| Matplotlib | 3.11.2 | User supplied and installed WSL package METADATA read |
| h5py | 3.14.0 | User supplied and installed WSL package METADATA read |
| Keras | 3.15.1 | Installed WSL package METADATA read |

Installed package metadata confirms files in the environment; it does not establish that those packages import or execute successfully with the requested interpreter.

## 2. All requested checks

**PASS** means the indicated source or supplementary check passed. **PARTIAL** means safe imports passed in the supplementary Windows interpreter but the requested WSL execution is still missing. **BLOCKED** means the requested TensorFlow runtime operation was not performed. Source checks are explicitly distinguished from runtime checks.

| No. | Requested check | Status | Evidence and limit |
| ---: | --- | --- | --- |
| 1 | `experiment_config.py` imports | PARTIAL | Imported successfully without training in the supplementary interpreter; WSL import not executed. |
| 2 | `stage4_lr_sensitivity.py` imports/checks without training | PARTIAL | Safe import and syntax parsing passed in the supplementary interpreter. Default/`--check-only` route was inspected; full WSL preflight is blocked. |
| 3 | Exactly 425 saved development IDs | PASS | `cv_indices` loaded; length and unique count both 425. |
| 4 | Fold 1 has 340 train / 85 validation IDs | PASS | `fold_1_train` and `fold_1_validation` loaded; counts 340 and 85. |
| 5 | Fold 1 train/validation disjoint | PASS | Intersection has zero IDs. |
| 6 | Fold 1 union equals the development set | PASS | Sorted concatenation equals sorted saved 425 development IDs exactly. |
| 7 | Fresh final-test IDs exactly 500–574 | PASS | Configuration tuple equals `tuple(range(500, 575))`; 75 unique IDs. |
| 8 | Fresh test/development disjoint | PASS | Intersection has zero IDs. No test images generated. |
| 9 | Adam instantiated at all four rates | BLOCKED | Factory specifies `tf.keras.optimizers.Adam(learning_rate=...)`; actual WSL instantiation not executed. |
| 10 | Adagrad instantiated at all four rates | BLOCKED | Factory specifies `tf.keras.optimizers.Adagrad(learning_rate=...)`; actual WSL instantiation not executed. |
| 11 | RMSProp instantiated at all four rates | BLOCKED | Factory specifies `tf.keras.optimizers.RMSprop(learning_rate=...)`; actual WSL instantiation not executed. |
| 12 | SGD instantiated at all four rates with momentum 0.9 and Nesterov | BLOCKED | Factory explicitly supplies `momentum=0.9`, `nesterov=True`; actual WSL instances/settings not verified. |
| 13 | Exactly 16 optimizer/LR configurations | PASS (definition) | Four unique optimizer names × four unique rates gives 16. Preflight loops over all 16; successful runtime instantiation count remains pending. |
| 14 | MSE is the only screening loss | PASS (source) | `LR_SCREENING_LOSS = "MSE"`; every screening compile uses that value through `make_loss`. MAE/Huber are reserved for the later comparison. |
| 15 | Maximum epochs 50 | PASS | `MAX_EPOCHS = 50`, passed to `model.fit`. |
| 16 | Patience 5 | PASS | `EARLY_STOPPING_PATIENCE = 5`, passed to `EarlyStopping`. |
| 17 | `min_delta = 1e-5` | PASS | Exact configured value passed to `EarlyStopping`. |
| 18 | `restore_best_weights=True` | PASS | Exact configured value passed to `EarlyStopping`. |
| 19 | `perf_counter()` times only `model.fit()` | PASS (source) | AST checks confirm start at line 295, sole timed try-body call `fit` at line 297, end in `finally` at line 302. Compile/dataset construction precede timing; evaluation and disk writes follow it. Validation and callbacks performed inside `fit` are part of that interval. |
| 20 | Identical initial weights for every configuration | PASS (source) | One shared copied weight list is passed to all runs; each fresh model receives `set_weights(initial_weights)` and its SHA-256 must match before training. No runtime initialization hash was obtained in WSL. |
| 21 | Divergence cannot terminate the other LR trials | PASS (source) | Equivalent custom callback checks non-finite loss/MSE and model weights, raising `DivergingRun`. Per-run handling records `diverged`; other exceptions record `failed`. Outer loops continue for those statuses; user interruption stops the loop. No divergent fit was run. |
| 22 | Old pilot test IDs never used in Stage 4 | PASS | Only saved development/train/validation arrays are loaded, never `test_indices`. Exact-fold/order guards reject substitutions, including a retired pilot ID. Training and validation contain no IDs from the 75-ID complement of development within 0–499. |

Additional checks:

| Check | Status | Evidence and limit |
| --- | --- | --- |
| Build one fresh U-Net | BLOCKED | Source preflight builds exactly one model with the configured shape. The user's reported successful build, output `(None, 512, 512, 1)`, and 120681 parameters are recorded as user-provided evidence; this agent did not execute that build. |
| No `model.fit()` during preflight | PASS (source and observed agent activity) | Imports are guarded. The preflight function calls no fit, screening runner, clean-image builder, or dataset builder. No training invocation was issued. |
| Output separate from pilot results | PASS | Resolved destination is `results/stage4_lr_sensitivity/`, separate from `results/stage3_cross_validation/`, `results/stage3_smoke_test/`, and Stage 1 figures. This destination does not exist yet. Training would use a new timestamp/UUID subdirectory. |

## 3. Exact split and screening logic

The existing archive is:

```text
C:\Users\HP ELITEBOOK 830 G8\Desktop\FINAL YEAR PROJECT\image-recovery-project\results\stage3_cross_validation\data_splits.npz
```

Archive SHA-256:

```text
7759010B9D4671917B55C9605A4051A2793629A489A65267F2A3E2C086A17E9E
```

The loader reads only `cv_indices`, `fold_1_train`, and `fold_1_validation`. It preserves their saved membership and order; it does not recreate splits or use the retired pilot holdout. IDs are validated as unique integer arrays in the original 0–499 domain. The selected fold must be a disjoint, complete partition of the 425 development IDs.

Fresh final-test IDs are reserved as integers 500 through 574 inclusive. Their identifiers were checked for disjointness only. Neither old nor new test images were generated, evaluated, or used for tuning.

All configurations use the same fold and initial weight array. Only training traversal is shuffled reproducibly. The fixed grid is:

| Optimizer | Constructor | Rates |
| --- | --- | --- |
| Adam | `tf.keras.optimizers.Adam(learning_rate=lr)` | 0.0001, 0.001, 0.01, 0.1 |
| Adagrad | `tf.keras.optimizers.Adagrad(learning_rate=lr)` | 0.0001, 0.001, 0.01, 0.1 |
| RMSProp | `tf.keras.optimizers.RMSprop(learning_rate=lr)` | 0.0001, 0.001, 0.01, 0.1 |
| SGD-Nesterov | `tf.keras.optimizers.SGD(learning_rate=lr, momentum=0.9, nesterov=True)` | 0.0001, 0.001, 0.01, 0.1 |

These constructors and arguments were checked in source; all 16 actual optimizer objects remain to be verified in WSL.

## 4. Discrepancies and concerns

No discrepancy was found between the Stage 4 source and the revised split, optimizer, loss, stopping, initialization, or timing protocol.

Outstanding concerns are:

- **Execution access:** `Wsl/E_ACCESSDENIED` prevents the requested runtime checks in this chat. This must not be reported as successful model construction or optimizer instantiation.
- **Historical documentation:** `PROJECT_CONTEXT.md` records the earlier Windows `venv313` TensorFlow failure. That statement concerns the Windows environment; it does not test the newly reported WSL environment. The original `CODEBASE_REVIEW.md` inventory predates the two Stage 4 files.
- **General split instruction:** `AGENTS.md` still says to reuse saved development/test splits. The explicit revised decision in authoritative `PROJECT_CONTEXT.md` and the user's instructions supersedes reuse of the old pilot test IDs. Current Stage 4 code follows the revised decision.
- **Epoch interpretation:** Raw minimum validation MSE/epoch and min-delta-qualified restored weights can differ. The script records both. This behavior matches the revised protocol and is not a defect.

No source correction was made based on these checks.

## 5. Changes and preservation

Only `CODEX_REPORT.md` was created for this request. No existing source, context, split, weight, history, figure, or pilot result was changed.

SHA-256 checks of the 41 existing project files outside dependency environments, bytecode caches, and Git internals found no changes before report publication. The same baseline is checked again after publication. The six pilot scripts, all pilot result artifacts, `experiment_config.py`, and `stage4_lr_sensitivity.py` are covered.

The current preflight produced no result files. After a separately approved training invocation, the existing runner will create a new `results/stage4_lr_sensitivity/screening_<UTC timestamp>_<unique ID>/` containing protocol/environment metadata, development/screening IDs, initial weights, aggregate CSV/JSON results, selected rates, and each configuration's history CSV/JSON, progress, run metadata, and optional error trace.

## 6. Remaining runtime check and recommendation

Run only this check command in WSL and retain its output:

```bash
/home/devnomso/fyp-tensorflow/bin/python -B '/mnt/c/Users/HP ELITEBOOK 830 G8/Desktop/FINAL YEAR PROJECT/image-recovery-project/stage4_lr_sensitivity.py' --check-only
```

A successful result must report 16 instantiated optimizer configurations, the correct model input/output shape, finite initial weights and their hash, and `training_started: false`, then exit without creating results. That output is still needed to complete the requested runtime verification.

**Recommendation: NOT READY on the currently verified evidence.** The static protocol and saved split passed; WSL imports, all 16 optimizer instantiations, and one fresh model build remain pending. Do not launch the 16-run learning-rate experiment until runtime preflight has been verified and the user explicitly approves training.

