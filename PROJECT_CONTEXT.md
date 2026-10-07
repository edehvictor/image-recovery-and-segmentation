# Project Context

This file is the authoritative context for coding and experimental work on the final-year project.

## Project Title

Comparative Analysis of Selected Optimizers and Loss Functions for Deep Learning-Based Image Recovery and Medical Image Segmentation

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

Revised setup (6 October 2026): the replacement test-set decision, fixed configuration, and LR-screening script are prepared. Syntax/import and saved-split checks passed without training. The current `venv313` TensorFlow import fails because `tensorflow.python` is missing, so optimizer instantiation and fresh U-Net construction remain pending. The 16-run experiment has not started and requires explicit user approval.

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

Before the optimizer-loss comparison, perform a learning-rate sensitivity experiment.

Keep the loss fixed as:

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

After selection, freeze the chosen learning rate for each optimizer before the loss-function experiment.

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

The selected learning rate for an optimizer must remain fixed when comparing its three loss functions.

Example:

If RMSProp is assigned learning rate η*, then:

RMSProp + MSE
RMSProp + MAE
RMSProp + Huber

must all use the same η*.

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

restore_best_weights:

True

Use the same stopping conditions for all configurations.

Correct terminology:

All optimizers use the same convergence/stopping conditions but may exhibit different convergence behaviour.

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

Results directory:

results/stage4_lr_sensitivity/

Use a new timestamped experiment subdirectory for each screening invocation. The screening script defaults to lightweight preflight and starts the 16 fits only with an explicit --train flag after user approval. No full training run is authorized yet.

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
