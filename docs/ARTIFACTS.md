# Artifacts outside this code release

This is a code-and-records release, not a complete checkpoint or dataset mirror.

| Artifact | Original location | Why excluded |
| --- | --- | --- |
| Recovery trained weights | Experiment Completed archive, Code/results/ | Model binaries are kept separately from source and small result records. |
| Segmentation trained weights and 3,112 predicted masks | Segmentation_Code result.zip | Preserve the returned archive for inference and image review. |
| Historical Stage 5/6 checkpoint arrays | Original Desktop repository results/ | Needed for historical resume/preflight audits, not table review. |
| Official ISIC image/mask data | Original experiment computer / official provider | Large external dataset; not supplied in the returned results ZIP. |
| Long training logs and environments | Original folders | Bulky machine-specific files; histories and structured results are included. |
| Old supervisor submission | Original folder and Git history | Dated duplicate source/documents, not the current code entry point. |

Retain the original archives. No files were deleted from them. Exclusion counts
are in excluded_artifact_counts.json. No public download for the checkpoints has
been created; a reproducibility release should provide one before claiming that
all trained models can be downloaded from GitHub.

To restore recovery weights, copy the original best.weights.h5 files into their
matching recovery/results/stage5_loss_comparison/runs/<run_id>/ folders.
Restore segmentation weights/masks to matching run IDs under
segmentation/results/stage7_isic_segmentation/. Keep these binaries ignored by Git.

Fresh segmentation runs save local split paths. Archived run.json records retain
the original computer's absolute split_file path. For a deliberately new inference
job from an archive, use a separate copy of the run record with split_file pointing
to the corresponding included split JSON. Do not overwrite the original records
or rerun completed test evaluation as part of packaging. Generating the result
tables and checking their arithmetic needs neither paths to data nor weights.

No original ISIC photographs or ground-truth masks are redistributed here.
