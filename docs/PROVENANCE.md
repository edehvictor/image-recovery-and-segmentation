# Code and record provenance

Three sources are retained separately:

1. `recovery_article_archive`: the Code directory inside the supplied
   Experiment Completed archive. This is the source of the article's revised
   recovery results, including stage5_loss_comparison.py.
2. `segmentation_result_archive`: the Segmentation_Code result.zip returned
   by Chisom. Source, protocol, split, histories, run records, test summaries,
   per-image metrics and the existing final results table are included.
3. `original_repository_worktree`: the user's Desktop image-recovery-project.
   This is preserved under legacy_recovery/, including untracked small Stage 6
   records. The original source folder was not edited. Its Git HEAD at packaging
   was 164eade, and its existing tracked working changes were line-ending-only
   according to git diff --ignore-space-at-eol --stat.

docs/source_manifest.json records each imported file's source-relative path,
size and original SHA-256. It contains no inferred experimental results.

## Deliberate changes to imported files

- segmentation/stage7_summary.py resolves run folders within this checkout,
  checks the optimizer/loss/rate identity, and saves new selections with relative
  directory names. This fixes references to the original computer. It does not
  change the scoring or selection rule.
- segmentation/test_stage7_isic_segmentation.py exercises the unconfirmed
  training guard using an explicitly unconfirmed copy. The source archive's
  actual protocol is confirmed, so the old test's expectation was stale.

The original versions remain in the supplied ZIP. All other imported files,
including model, training and evaluation source, are byte-for-byte preserved.
The verification script reports and checks the two declared modifications.

Original result paths referring to the experiment computer are retained in JSON
for provenance. tools/summarize_results.py resolves those records locally.
segmentation/stage7_isic_segmentation.py retains its original final-evaluation
implementation, which uses the stored split_file path. See ARTIFACTS.md before
attempting evaluation from a moved archive; already evaluated runs refuse repeats.

The dated supervisor submission folder and documents are omitted from the new
working tree. They remain in the original folder and existing Git history.
Omitting files from the current tree does not remove them from old commits.
