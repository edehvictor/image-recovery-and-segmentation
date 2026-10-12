# Project Context

This file is the authoritative context for coding and experimental work on the final-year project.

## Project Title

Comparative Analysis of Selected Optimization Algorithms and Loss Functions for Deep Learning-Based Image Recovery and Medical Image Segmentation

## Project Structure

The project has two experiments:

1. Controlled image recovery using a Shepp-Logan phantom degraded with Gaussian noise.
2. Medical image segmentation using ISIC 2018 Task 1 skin-lesion images and ground-truth masks.

The recovery experiment must be completed and revised first before proceeding to segmentation.

---

## Current Status

Completed:

- Shepp-Logan phantom generation
- 500 noisy recovery samples
- Compact U-Net implementation
- Smoke test
- Pilot 5-fold recovery cross-validation
- Pilot SSIM evaluation
- Adam diagnostic experiment

The existing recovery results are preliminary and must not be presented as final results.

The completed Stage 4 MSE recovery learning-rate screen contains 16 completed configurations, archived under `results/stage4_lr_sensitivity/combined_final_lr_screening/` with provenance linking the two 7 October screening sessions. Preserve these results; MSE screening will not be rerun. This single-fold screening is selection evidence, not the final optimizer-loss comparison.

Stage 5 MAE/Huber screening completed on 11 October 2026: all 32 runs completed and eight optimizer-loss learning rates were selected. Evidence is saved in `results/stage5_loss_lr_screening/screening_20261010T054644Z_d2ae382d/`. On 12 October, Codex audited all 32 saved run records, complete histories, raw-best checkpoint arrays/checksums and selections without fitting or evaluation. Stage 6 was then created and its guarded WSL preflight passed; the 60-fit comparison has not started. The earlier Windows `venv313` failure is historical and does not describe the working WSL runtime.

The pilot experiment revealed methodological issues that must be corrected:

- only three optimizers were tested
- only MSE loss was tested
- optimizer learning rates were not controlled systematically
- plain SGD was used instead of SGD with Nesterov momentum
- maximum epochs were only 10
- timing results contained anomalies
- final study must include SSIM, RMSE/NRMSE%, computation time and convergence information

Do not overwrite the pilot scripts or pilot results.

---

## Recovery Dataset

Clean image:

Shepp-Logan phantom

Image size:

512 x 512

Pixel range:

0 to 1

Noise model:

Y = X + N

where:

N ~ Normal(0, 0.1^2)

Noise mean:

0

Noise standard deviation:

0.1

Pilot noisy samples:

500 (IDs 0 through 499).

Revised recovery samples:

425 retained development samples plus 75 fresh final-test samples. The original exposed pilot test samples are excluded from the revised experiment.

Development samples:

425

Fresh final test samples:

75, using deterministic sample IDs 500 through 574.

The old 75-sample pilot test set is retired and will not serve as the revised final test set.

Random seed:

42

All noisy samples are generated from the same clean phantom with different Gaussian noise realizations. The revised experiment retains the pilot's TensorFlow stateless generator: each sample uses the two-element seed [42, sample_id], with Gaussian noise added to the float32 512 x 512 clean phantom and clipped to [0, 1]. New final-test IDs specify fresh noise realizations; they must not be generated or evaluated during tuning.

Therefore, recovery cross-validation evaluates robustness to new noise realizations, not anatomical or structural generalization.

---

## Recovery Data Splitting

Methodological decision recorded on 6 October 2026:

- The original 75-sample pilot test set has already been exposed during exploratory work and will not be used as the final test set in the revised experiment.
- Retain exactly the existing 425 development/CV indices and their saved fold assignments from `results/stage3_cross_validation/data_splits.npz`.
- Use new deterministic noisy-sample IDs 500 through 574 (inclusive) as the fresh 75-sample final test set.
- Retire the old pilot test IDs from revised training, validation, screening, and final evaluation.
- The new final test IDs must never be used during learning-rate selection, cross-validation, loss comparison, epoch selection, or any other tuning.
- Split checks may inspect these identifiers for disjointness; they must not generate or inspect their noisy images during tuning.

The fresh final test set must remain untouched during:

- learning-rate selection
- optimizer selection
- loss-function comparison
- epoch selection
- hyperparameter tuning

The 425 development samples are used for 5-fold cross-validation.

Each fold contains:

340 training samples

85 validation samples

The same folds must be used for every optimizer-loss configuration.

---

## Model

Reuse the existing compact U-Net from stage2_unet.py unless a documented correction is necessary.

The recovery model uses:

Input:

512 x 512 x 1

Output:

512 x 512 x 1

Output activation:

sigmoid

The architecture must remain identical across optimizer and loss comparisons.

---

## Optimizers

The final recovery study uses:

- Adam
- Adagrad
- RMSProp
- SGD with Nesterov momentum

For SGD:

momentum = 0.9

nesterov = True

Do not describe the final SGD experiment as plain SGD.

---

## Learning-Rate Sensitivity Study

Stage 4 completed a learning-rate sensitivity experiment before the main optimizer-loss comparison. Its fixed screening loss was:

MSE

Learning-rate grid:

0.0001
0.001
0.01
0.1

Test all four learning rates for all four optimizers.

This gives:

4 optimizers x 4 learning rates = 16 learning-rate configurations

For initial LR screening, use saved CV fold 1 only: exactly 340 training IDs and 85 validation IDs. Do not regenerate or reshuffle fold membership. Use this identical fold for all 16 configurations; reshuffle only training traversal order reproducibly.

Build one fresh compact U-Net with seed 42, copy its initial weights, and set each configuration's model to those exact weights. Verify the initialization hash for every run. Recreate the dataset pipeline with the same seed and order policy for every configuration.

Keep MSE fixed, with maximum 50 epochs, patience 5, min_delta 1e-5, and restore_best_weights=True. SGD must use momentum=0.9 and nesterov=True. Measure only model.fit() with time.perf_counter().

Save optimizer, learning rate, run status, raw best epoch, stopping epoch, raw best validation MSE, restored-model validation MSE, validation PSNR, training time, full history, and initialization identity. Record failed/diverging runs and continue the other configurations. The raw minimum and the min_delta-qualified restored weights may differ; report both rather than conflating them.

Select the suitable learning rate for each optimizer using validation performance only. Use the lowest raw best validation MSE among completed runs; break exact ties by the smaller learning rate. If an optimizer has no completed run, record no selected rate. This single-fold screen is preliminary selection, followed by the planned five-fold optimizer/loss comparison.

Do not use the test set for learning-rate selection.

The Stage 4 selected rates apply to the MSE objective only. The supervisor-directed extension selects learning rates separately for each optimizer-loss pair. Preserve the completed MSE choices; screen MAE and Huber separately in Stage 5, then freeze each pair's selected rate before the main comparison.

---

## Recovery Loss Functions

The main recovery experiment uses:

- MSE
- MAE
- Huber Loss

Huber delta:

0.1

The optimizer-loss experiment therefore contains:

4 optimizers x 3 losses = 12 configurations

Select and freeze one learning rate separately for each optimizer-loss pair using validation data only. For example, RMSProp + MSE, RMSProp + MAE and RMSProp + Huber may use different selected rates. The completed MSE screen is retained; Stage 5 supplies the remaining MAE/Huber rate selections.

Within a pair, choose the smallest raw best validation loss among completed valid runs, breaking exact ties with the smaller learning rate. Do not compare MAE and Huber loss magnitudes to rank the two objectives. Use common recovery metrics for later cross-loss performance comparisons.

---

## Stage 5: MAE/Huber Learning-Rate Screening

Created and checked on 10 October 2026. The saved experiment finished on 11 October 2026 at 17:24:04 UTC (18:24:04 Africa/Lagos), with 32/32 completed runs, no failed/diverged runs, and eight selected optimizer-loss pairs. The completed records were audited without rerunning training on 12 October.

- Losses: MAE and Huber (delta 0.1), through the unchanged `experiment_config.make_loss` constructors.
- Grid: Adam, Adagrad, RMSProp and SGD-Nesterov x two losses x four learning rates (1e-4, 1e-3, 1e-2, 1e-1) = 32 runs and eight optimizer-loss pairs. MSE is excluded.
- SGD retains momentum 0.9 and nesterov=True. All remaining protocol constants, the saved 425 development IDs, Fold 1's 340/85 membership/order and the compact 120681-parameter U-Net are unchanged.
- One canonical model weight set is built in preflight, copied to every fresh run and verified before fit. Initialization SHA-256 verified in WSL: `3d8af7fb3e7d315c2358a71bef8213afa7e6e29217099ad230fe89951a531926`. Saved development/Fold 1 SHA-256: `b7cca054d0978d0ebeb4a87ab85da4421771e18bdd0fc430e045519ee0a30caf`.
- Deterministic TensorFlow operations were enabled. Preflight instantiated all 32 optimizer-loss-LR configurations and the raw-best/stopping callbacks, built one U-Net with input/output `(None, 512, 512, 1)`, and performed no fit, inference, dataset construction or noise generation.
- EarlyStopping monitors val_loss with mode=min, patience=5, min_delta=1e-5 and restore_best_weights=False. The separate raw-minimum callback selects the first strictly smallest val_loss and retains its weights without a min_delta threshold. After fit, best epoch, best loss, validation metrics and saved model weights all refer to that raw-best epoch.
- Restored validation metrics: MSE, RMSE, NRMSE%=100*RMSE, mean per-image PSNR (max_val=1), and mean per-image scikit-image SSIM (data_range=1, window 7, uniform weights/sample covariance). These restored-model metrics are saved for all 32 completed Stage 5 runs.
- Only `model.fit` is timed with time.perf_counter; setup, disk writing and post-fit evaluation are excluded. In-fit validation and in-memory callback work are included.
- Non-finite loss, validation loss, MSE or model weights mark an individual run as diverged. Other per-run failures are recorded and do not discard completed work.
- New outputs use `results/stage5_loss_lr_screening/screening_<UTC timestamp>_<short id>/`. Check-only creates no experiment directory. Training saves protocol/preflight/environment metadata, canonical model arrays, development/Fold 1 arrays, per-attempt full histories, raw-best model arrays/checksums and run records. Final/updated summaries: all_runs.csv, all_runs.json, selected_learning_rates.json, selected_learning_rates.csv and experiment_summary.json.
- `--resume <directory>` checks the shared protocol, source/software versions, split and initialization hashes. It skips only completed runs whose identities, constructor settings, histories, epochs, metrics and weight artifacts validate. Every failed, interrupted, missing or invalid run restarts at epoch 1 from the saved canonical model weights; optimizer state is not resumed. Previous attempts are retained, aggregates are replaced atomically, and an OS lock prevents concurrent writers.
- Default/`--check-only` execution never fits, including when --resume is supplied. Fitting always requires --train and explicit user approval. No fitting or completed-run resumption was performed by Codex during preparation or the 12 October audit. The later saved artifacts record completed training; all completed-run histories and checkpoints passed the read-only audit.
- Neither the retired pilot test array nor fresh final-test images are loaded/generated/evaluated. IDs 500-574 remain reserved for final evaluation. Main 60-fit recovery comparison and segmentation remain outstanding and are not created or started by Stage 5.


### Completed Stage 5 result summary

The eight selected rates below minimize raw best validation loss within each optimizer-loss pair, with exact ties broken by the smaller rate. All four candidate rates completed for every pair. MSE screening was retained unchanged.

| Optimizer | MAE selected LR | MAE best validation loss | Huber selected LR | Huber best validation loss |
| --- | ---: | ---: | ---: | ---: |
| Adam | 0.001 | 0.123182587326 | 0.001 | 0.0101942420006 |
| Adagrad | 0.1 | 0.00272162584588 | 0.1 | 0.000238836510107 |
| RMSProp | 0.001 | 0.00199504476041 | 0.01 | 0.0000814798477222 |
| SGD-Nesterov | 0.01 | 0.00459532439709 | 0.1 | 0.000163597389474 |

Exact values and restored MSE/RMSE/NRMSE%/PSNR/SSIM are in selected_learning_rates.csv and all_runs.csv/JSON. Recorded fit time across the 32 completed runs totals 64742.327082737 seconds (17.984 hours); this is the sum of fit intervals, not wall-clock experiment duration. Adam's selected MAE and Huber runs both have best epoch 1 and restored MSE approximately 0.0589361, much worse than the other selected runs. Treat this as an observed limitation requiring discussion; completion status alone does not establish useful convergence. No run was replaced or omitted.

These are Fold 1 tuning results, not final recovery results. MAE and Huber loss scales must not be compared directly. Both test sets remained outside this screening code; fresh IDs 500-574 remain reserved for final evaluation. Codex has not independently repeated metric computation or student verification.

## Stage 6: Five-Fold Recovery Comparison

Created on 12 October 2026 in `stage6_recovery_cross_validation.py`. Guarded preflight passed with `/home/devnomso/fyp-tensorflow/bin/python` (Python 3.13.16, TensorFlow 2.21.0, Keras 3.15.1, NumPy 2.5.3, scikit-image 0.26.0). No Stage 6 training has started and the 60 fits require explicit training approval.

Freeze the twelve pair-specific rates before training:

| Optimizer | MSE LR (Stage 4) | MAE LR (Stage 5) | Huber LR (Stage 5) |
| --- | ---: | ---: | ---: |
| Adam | 0.001 | 0.001 | 0.001 |
| Adagrad | 0.1 | 0.1 | 0.1 |
| RMSProp | 0.001 | 0.001 | 0.01 |
| SGD-Nesterov | 0.1 | 0.01 | 0.1 |

- Plan: four optimizers x three losses x five existing saved folds = 60 fits. Every fold contains exactly 340 training and 85 validation IDs, disjoint and covering the same 425 development IDs. Each development ID occurs in validation exactly once across five folds; saved membership and order are retained.
- Preflight audits Stage 4's 16 combined selection records and all 32 Stage 5 per-run histories/checkpoint checksums, recomputes the recorded selections, instantiates all 60 optimizer/loss combinations, and builds exactly one U-Net. Fit, evaluate, predict and dataset/image creation were prohibited by runtime guards during validation.
- All 60 fits start from the same fresh seed-42 canonical initialization (SHA-256 `3d8af7fb3e7d315c2358a71bef8213afa7e6e29217099ad230fe89951a531926`), matching screening initialization. Screened best-model weights are never used to start CV. Training order is reset reproducibly for every run.
- Batch size 1, maximum 50 epochs, patience 5, min_delta 1e-5, Gaussian noise mean 0/std 0.1, Huber delta 0.1, SGD momentum 0.9/nesterov=True. Every objective, including MSE, uses the Stage 5 raw-best policy: EarlyStopping restore_best_weights=False; explicitly restore the first strict raw-minimum validation-loss weights before metrics/saving. Completed Stage 4 results are unchanged.
- Measure only model.fit with time.perf_counter. Save full histories, raw-best weights/checksums, epochs, restored loss, common MSE/RMSE/NRMSE%/PSNR/SSIM, fit time, settings and status per run. Non-finite per-run failures are recorded and the remaining configurations continue; invalid shared protocol stops execution.
- Save new runs only under `results/stage6_recovery_cross_validation/comparison_<UTC timestamp>_<short id>/`. Outputs include protocol/preflight/environment JSON, canonical initial_weights.npz, cv_splits.npz, frozen selected_learning_rates.json, per-run attempt artifacts, all_runs.csv/JSON, optimizer_loss_summary.csv/JSON and experiment_summary.json. No Stage 6 output directory is created by check-only.
- The training runner also saves noisy-input baseline MSE/RMSE/NRMSE%/PSNR/SSIM for each fold's 85 validation images in noisy_baseline.csv/JSON, outside fit timing. Preflight does not compute these metrics.
- Report common metrics, fit time and epochs across folds as mean and sample standard deviation (ddof=1). Include completed-fold counts and status counts; mark a pair complete only when all five folds completed. Do not silently present a partial set as five-fold results or rank different loss functions by loss magnitude.
- Resume requires unchanged frozen rates, protocol, evidence, normalized source hashes, software, fold arrays and canonical initialization. Skip only verified completed run records; other runs restart from epoch 1 with a fresh optimizer. Preserve previous attempts, write aggregates atomically, and lock the experiment against concurrent writers.
- The authorized line-ending restore changed stage2_unet.py's raw byte checksum only. The Stage 5 audit verified equivalence to its recorded source under exact LF/CRLF variants; saved experiment artifacts were not edited. Stage 6 source hashes normalize CRLF to LF to support Windows/WSL checkouts while still detecting content changes.
- The same development pool and Fold 1 were used for rate tuning, so these CV summaries are conditional on the selected rates and are not an independent nested-CV estimate. Final refitting and evaluation on untouched fresh test IDs 500-574 remain separate later work. Neither pilot test IDs nor fresh test images enter Stage 6; segmentation is also outside its scope.

Commands: default or `--check-only` performs preflight only; `--train` is required to fit, including with `--resume <comparison directory>`. No real Stage 6 training or completed-run resume has been exercised yet.

---

## Training Settings

Batch size:

1

Maximum epochs:

50

Early stopping patience:

5

min_delta:

1e-5

Stage 4 EarlyStopping restore_best_weights:

True (historical completed protocol; unchanged).

Stage 5 and Stage 6 EarlyStopping restore_best_weights:

False. EarlyStopping controls stopping only. A separate callback records every strict raw improvement and copies that epoch's model weights, without min_delta. After fit, restore these raw-best weights explicitly before validation and saving.

All configurations share the same maximum epochs, patience and min_delta. These are empirical stopping controls, not mathematical convergence theorems. The difference between historical Stage 4 and Stage 5/6 checkpoint restoration is explicit and does not alter completed MSE results.

Record:

- best epoch
- stopping epoch
- best validation loss
- total training time

---

## Model Initialization

Within each fold, comparable optimizer-loss configurations should begin from the same initial U-Net weights where feasible.

This reduces random initialization as a confounding factor.

Do not silently change initialization strategy between configurations.

---

## Recovery Evaluation Metrics

Calculate the following for every recovered image:

- MSE
- RMSE
- NRMSE%
- PSNR
- SSIM

Definitions:

RMSE = sqrt(MSE)

NRMSE = RMSE / (Xmax - Xmin)

Since the normalized images satisfy:

Xmax = 1
Xmin = 0

then:

NRMSE% = 100 x RMSE

For PSNR:

MAX = 1

For SSIM:

data_range = 1

Lower is better for:

MSE
RMSE
NRMSE%

Higher is better for:

PSNR
SSIM

Also evaluate the noisy input as a baseline.

---

## Computation Time

Measure model training time using:

time.perf_counter()

Time only the model.fit() call.

Do not include:

- plotting
- CSV writing
- model saving
- manual delays

Before final timing runs:

- keep the laptop connected to power
- disable sleep/hibernation
- close unnecessary heavy applications
- use the same machine
- perform an unrecorded warm-up if necessary

Record hardware and software versions.

---

## Cross-Validation Reporting

For each configuration, report validation metrics across the five folds as:

mean ± standard deviation

Use the same fold assignments for every configuration.

---

## Final Refitting

After cross-validation and model selection are complete:

1. Record the best epoch from each fold.
2. Compute the median best epoch.
3. Train the final configuration on all 425 development samples for that number of epochs.
4. Evaluate the final trained model on the fresh, untouched 75-sample final test set (deterministic IDs 500 through 574). Do not reuse the exposed pilot test set.

Do not select the final epoch using test performance.

---

## Recovery Visual Results

For visual comparison, use the same test sample for:

- original clean phantom
- noisy phantom
- recovered phantom

Do not cherry-pick a visually favourable sample after looking through the test results.

Use a predefined or documented sample-selection rule.

---

## Segmentation Experiment

The second experiment uses ISIC 2018 Task 1 skin-lesion images and corresponding ground-truth segmentation masks.

Input:

real skin-lesion image

Target:

ground-truth lesion mask

Prediction:

predicted lesion mask

The task is segmentation, not diagnosis or disease classification.

Segmentation losses:

- Binary Cross-Entropy
- Dice Loss

Segmentation metrics:

- Dice coefficient
- IoU

The same four optimizers will ultimately be compared:

- Adam
- Adagrad
- RMSProp
- SGD-Nesterov

Do not automatically reuse recovery learning rates for segmentation.

Learning-rate selection for segmentation must be justified separately because optimizer behaviour can be task-dependent.

---

## Existing Scripts

Preserve the existing pilot scripts:

stage1_phantom.py
stage2_unet.py
stage3_smoke_test.py
stage3_cross_validation.py
stage3_ssim.py
stage3_adam_diagnostic.py

New revised experiment files should be created separately.

Revised experiment files:

experiment_config.py
stage4_lr_sensitivity.py
stage5_loss_lr_screening.py
stage6_recovery_cross_validation.py

Results directories:

results/stage4_lr_sensitivity/
results/stage5_loss_lr_screening/
results/stage6_recovery_cross_validation/

Use a new timestamped subdirectory for every new experiment. All revised runners default to preflight and require an explicit --train flag for fitting. Stage 4 MSE and Stage 5 MAE/Huber screening are completed; preserve them. Stage 5 and Stage 6 support validated --resume but still require --train to fit. No Stage 6 training is authorized yet.

Do not overwrite pilot results.

---

## Reproducibility

Save or record:

- random seed
- split indices
- fold assignments
- initialization strategy
- optimizer name
- optimizer settings
- learning rate
- loss function
- Huber delta
- batch size
- max epochs
- patience
- min_delta
- best epoch
- stopping epoch
- training time
- TensorFlow version
- Python version
- NumPy version
- hardware information
- training history
- run status

Failed or diverging runs must be recorded rather than silently omitted.

---

## AI Usage

Maintain an AI_USAGE_LOG.md file.

Record:

- date
- AI tool used
- purpose
- type of assistance
- what was independently verified by the student

AI-assisted code or writing must not be treated as automatically correct.

The student must execute, inspect and verify the computational results.

---

## Coding Rules for Codex

Before making substantial changes:

1. Read PROJECT_CONTEXT.md.
2. Inspect the existing implementation.
3. Do not overwrite working pilot scripts.
4. Do not use the held-out test set for tuning.
5. Reuse saved data splits.
6. Keep comparisons controlled.
7. Save experimental outputs automatically.
8. Do not fabricate results.
9. Explain major methodological or code changes.
10. Update PROJECT_CONTEXT.md only when a methodological decision has actually been established.
