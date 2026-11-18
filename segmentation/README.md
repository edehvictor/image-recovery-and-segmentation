# Completed ISIC segmentation experiment

These files come from Chisom's returned result archive. There are 32 completed
full screening runs (four optimizers, two losses and four rates), plus two
separate trial records. Eight selected runs have saved test results.
`SEGMENTATION_README.md` is the original preparation document and is retained
for provenance; its 'not trained' and draft statements are historical.

`segmentation_protocol_draft.json` retains its original name and bytes but
actually has `confirmed_for_training: true`, maximum_epochs 50, 256 x 256 inputs,
batch size 4, seed 42, no augmentation and a 70/15/15 image-ID split.
Do not use protocol_trial_1epoch.json for the full experiment.

## Read the existing results

From the repository root:

```bash
python tools/summarize_results.py
```

The command checks all four rates per optimizer-loss pair and validates the
frozen choices against `restored_validation.dice`. It writes fresh summary CSVs
to reports/, leaving imported results unchanged. It does not need the dataset
or model weights. Test scores never enter learning-rate selection.

The older stage7_summary.py helper remains available with `select`, `evaluate`
and `collect`. It now resolves saved run paths on the current computer.
Its select/collect commands write inside the original results folder, so use
the root reporting tool for read-only review of archived records.

## Dataset/model preflight

Obtain the official ISIC 2018 Task 1 training images and ground-truth masks
externally. Do not commit the dataset. From this folder:

```bash
python stage7_isic_segmentation.py --check-only --images /path/to/ISIC2018_Task1-2_Training_Input.zip --masks /path/to/ISIC2018_Task1_Training_GroundTruth.zip
```

The paths must refer to the real files on the execution computer. This model
check requires TensorFlow and the dataset. It is separate from the data-free
tests used for packaging.

## Original experiment commands

Each training call runs one explicitly specified combination. The four optimizer
names are Adam, Adagrad, RMSProp and SGD-Nesterov; loss names are BCE and Dice;
rates are 0.0001, 0.001, 0.01 and 0.1. For example:

```bash
python stage7_isic_segmentation.py --train --optimizer Adam --loss BCE --learning-rate 0.001 --images /path/to/ISIC2018_Task1-2_Training_Input.zip --masks /path/to/ISIC2018_Task1_Training_GroundTruth.zip
```

For a genuinely new reproduction, use a separate copy/output workspace and
complete all 32 combinations before `python stage7_summary.py select`.
After freezing those eight choices, the existing final-test command is:

```bash
python stage7_summary.py evaluate --images /path/to/ISIC2018_Task1-2_Training_Input.zip --masks /path/to/ISIC2018_Task1_Training_GroundTruth.zip
python stage7_summary.py collect
```

Do not repeat these training/evaluation steps simply to obtain the published
table; it already exists. Saved weights are omitted from Git. Moved archival
run records retain absolute split paths; see ../docs/ARTIFACTS.md.

## Metric interpretation

Final metrics are computed at each image's original resolution, after resizing
the predicted probability map and thresholding at 0.5. The primary table is
the mean of defined per-image Dice, IoU, precision, accuracy, specificity and
sensitivity. Test summaries also include pooled-pixel metrics. This task is
lesion-boundary segmentation, not a skin-disease diagnosis classifier.
