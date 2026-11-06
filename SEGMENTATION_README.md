# ISIC 2018 Task 1 segmentation preparation

This is a new, separate segmentation runner. It imports the existing compact
`stage2_unet.build_unet` architecture with an RGB input and a one-channel sigmoid
mask output. It does not alter the recovery scripts or their results.

## Dataset

Obtain the two **official ISIC 2018 Task 1 training** archives:

- Images: https://isic-archive.s3.amazonaws.com/challenges/2018/ISIC2018_Task1-2_Training_Input.zip
- Masks: https://isic-archive.s3.amazonaws.com/challenges/2018/ISIC2018_Task1_Training_GroundTruth.zip

The runner reads the ZIP files directly, so extraction is optional. The image
archive is about 10.4 GB and the mask archive about 26 MB. Do not commit either
dataset to Git. The files must have matching `ISIC_XXXXXXX.jpg` and
`ISIC_XXXXXXX_segmentation.png` names; missing or duplicate matches fail the
check. The official Task 1 training set has 2,594 paired images. The loader
checks image/mask dimensions and binary mask values before resizing. RGB
images use bilinear resizing; masks use nearest-neighbour resizing.

## Run a check without training

After placing this file and `segmentation_protocol_draft.json` beside
`stage2_unet.py` in the repository:

```bash
/home/devnomso/fyp-tensorflow/bin/python -B stage7_isic_segmentation.py \
  --check-only \
  --images /path/to/ISIC2018_Task1-2_Training_Input.zip \
  --masks /path/to/ISIC2018_Task1_Training_GroundTruth.zip
```

The check verifies pair IDs, decodes sample images/masks, builds the RGB U-Net,
instantiates all four optimizers, performs one untrained forward pass, and
evaluates both loss functions. It **does not call `model.fit()`**. A tiny local
fixture may be checked with `--allow-small-sample`, but that option is not
accepted for training.

## Decisions to confirm with Chisom before training

`segmentation_protocol_draft.json` has `confirmed_for_training: false` and
training refuses to start in this state. Its 256-pixel image size, batch size 4,
70/15/15 ID-level split, maximum 50 epochs, and no augmentation are **draft
values**, not reported research decisions. Confirm these settings, the split
policy (including possible duplicate lesions/patients), and segmentation
learning rates at the meeting. Do **not** copy the recovery learning rates
without segmentation validation evidence.

After the protocol is agreed, set `confirmed_for_training` to `true`. Each
training command runs one explicitly chosen optimizer, loss, and learning rate,
using the same saved split and deterministic initialization. Example syntax:

```bash
/home/devnomso/fyp-tensorflow/bin/python -B stage7_isic_segmentation.py \
  --train --optimizer Adam --loss BCE --learning-rate 0.001 \
  --images /path/to/ISIC2018_Task1-2_Training_Input.zip \
  --masks /path/to/ISIC2018_Task1_Training_GroundTruth.zip
```

The run saves its protocol, split reference, initial-weight hash, epoch history,
best restored weights, validation Dice/IoU, training time, and status under a
new `results/stage7_isic_segmentation/` directory. It never reads final-test
image bytes during training. The same commands can be run for each agreed
optimizer/loss/rate combination. There is no automatic 32-run sweep and no
training has been performed in this preparation.

**Final test evaluation is a separate step, only after all selections are
frozen.** It reads each held-out test image once per completed run, evaluates
Dice and IoU at the original mask size, and saves one predicted mask per image.
It refuses to repeat a completed run's test evaluation:

```bash
/home/devnomso/fyp-tensorflow/bin/python -B stage7_isic_segmentation.py \
  --final-evaluate results/stage7_isic_segmentation/RUN_DIRECTORY \
  --images /path/to/ISIC2018_Task1-2_Training_Input.zip \
  --masks /path/to/ISIC2018_Task1_Training_GroundTruth.zip
```

Do not describe the model's output masks as diagnostic predictions. This task
is lesion-boundary segmentation.
