# Image recovery and medical image segmentation

Code and saved experiment records for **Comparative Analysis of Selected
Optimization Algorithms and Loss Functions for Deep Learning-Based Image
Recovery and Medical Image Segmentation**.

This repository brings the two experiments together without combining their
datasets, losses or selection rules. Imported numerical results are preserved.

## Repository map

| Folder | Purpose |
| --- | --- |
| `recovery/` | Article recovery code from the completed experiment archive: 48 screening runs, 12 selected optimizer-loss pairs and final-test results. |
| `segmentation/` | Code returned with the completed ISIC experiment: 32 screening runs, eight selected pairs, test summaries and per-image metrics. |
| `legacy_recovery/` | Original repository workflow and records, including pilot results and the incomplete Stage 6 five-fold comparison. |
| `tools/` | Checks and portable table generation from saved records; no TensorFlow or dataset needed. |
| `docs/` | Provenance, verification, artifact exclusions and AI declaration. |

`stage2_unet.py` is retained beside each workflow to preserve the original
imports and architecture versions. These copies are intentional. Do not
replace them with one shared file without checking historical source hashes.

## Review the existing results without training

From this repository's root, with Python 3.10 or newer:

```bash
python tools/summarize_results.py
python -m unittest discover -s segmentation -p "test_*.py" -v
python tools/verify_repository.py
```

The first command writes separate recovery and segmentation CSV tables to
`reports/`, recreates the learning-rate choices using validation scores and
checks them against the frozen selections. It never imports TensorFlow,
fits a model, reads dataset images or performs test inference.

The verification command checks source/result hashes, Python syntax, split
disjointness and agreement between the published records and generated tables.
See `docs/VALIDATION.md` for checks actually performed during packaging.

## Reproduce model checks or experiments

Use a separate Python environment with the packages in `requirements.txt`.
That list records direct dependencies, not a verified lockfile. The original
segmentation runtime versions and hardware were not recorded in the returned
archive; do not attribute the laptop's WSL versions to that experiment.

See `recovery/README.md` and `segmentation/README.md` for the existing CLI commands,
data requirements, training flags and artifact-restoration instructions.
Training is not part of the commands above.

Large logs, environments, dataset archives, model checkpoints and 3,112 predicted
segmentation masks are deliberately excluded from this Git tree. The returned
archives and original repository retain them. Table review needs none of them;
inference with a saved model requires the corresponding weights and dataset.
See `docs/ARTIFACTS.md`. There is no public artifact download supplied yet.

On Windows, use a short checkout path such as `C:\research\image-recovery-project`
and enable `git config core.longpaths true`: historical attempt paths are deep.

## Authorship and AI assistance

Edeh Victor Chinomso's disclosure is in `docs/AI_DECLARATION.md`; the assistance
record is in `AI_USAGE_LOG.md`. Other contributors should confirm their own
details and disclosures. No software licence is imposed by this packaging;
the authors must choose one before granting reuse permissions.

Git history from the original repository is retained. The original Desktop
working folder is unchanged. Publishing this prepared copy is a separate step.
