# Current project context

This file describes the assembled code and records as inspected on 18 November
2026. Historical notes in legacy_recovery/ remain historical, including statements
that were true when written but are now superseded.

## Article recovery experiment

Primary source: the returned Experiment Completed archive, under recovery/.
Four optimizers (Adam, Adagrad, RMSProp, SGD with Nesterov momentum 0.9), three
losses (MSE, MAE, Huber delta 0.1), four rates (0.0001, 0.001, 0.01, 0.1): 48 runs.
Rates are chosen separately per pair by lowest restored validation MSE, with
the lower rate breaking an exact tie. Early stopping monitors the training
objective's validation loss; that is distinct from the rate-selection metric.
Saved Fold 1 has 340 training and 85 validation IDs from 425 development IDs.
Fresh final-test IDs are 500 through 574. The original exposed pilot test set
is not the revised final test set. All samples share one clean phantom; results
measure new noise realizations, not generalization to unseen anatomy.
The saved final_results.csv contains 12 final-test comparisons.

## ISIC segmentation experiment

Primary source: Segmentation_Code result.zip, under segmentation/.
Official ISIC 2018 Task 1 training images and masks were split by image ID:
1,816 training, 389 validation and 389 local held-out test images. This is not
the official challenge test cohort. The records do not establish patient-level
grouping. RGB inputs and binary targets are resized to 256 x 256, batch size 4,
seed 42, maximum 50 epochs, early-stopping patience 5, min_delta 1e-5 and restored
best weights, with validation Dice monitored. Augmentation is off.
Four optimizers x BCE/Dice x four rates gives 32 completed full screening runs.
Two separate trial records are retained but excluded from selection.
Each pair's rate maximizes restored_validation.dice; ties choose the lower rate.
Eight selected runs have final-test records. Test probabilities are resized to
original mask resolution and thresholded at 0.5. Dice, IoU, precision, accuracy,
specificity and sensitivity are reported as means of defined per-image scores;
pooled-pixel scores are recorded separately. Do not confuse them.
The returned protocol still has 'draft' in its filename, but its actual
confirmed_for_training value is true. Its text and numerical settings are
preserved as evidence of what was run.

## Historical recovery workflow

legacy_recovery/ preserves the Desktop code and small results. Its Stage 4 MSE
and Stage 5 MAE/Huber screens use different selection/checkpoint details from
the article recovery archive. Do not combine them as one protocol.
The saved Stage 6 summary records 4 completed of 60 planned runs and 56 pending;
it is not a completed five-fold result and it is not segmentation.

## Packaging scope

No experiment is run or methodological setting changed by consolidation.
Training and evaluation source files retain their original bytes. A reporting
path fix and a test-fixture correction are documented separately. Imported
results retain their original bytes. New table exports are derived records.
