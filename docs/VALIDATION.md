# Packaging validation

Checked on 18 November 2026. These are record/code checks, not new experiments.

| Check | Outcome |
| --- | --- |
| Imported file manifest | 692 files; all unchanged apart from two documented reporting/test edits. |
| Python syntax | 26 Python files parsed successfully. |
| Lightweight segmentation tests | Nine tests passed: pairing, duplicate/missing IDs, ZIP access, split reproducibility, metric arithmetic/undefined rates, training guard and portable reporting paths. |
| Recovery split import | Original config imports without TensorFlow; 425 development, 340 training, 85 validation; original split validation passes; fresh test IDs are 500-574. |
| Recovery records | 48 unique completed optimizer-loss-rate records; all 12 choices match lowest restored validation MSE; rates/times/epochs match final table. |
| Segmentation records | 32 unique full screening runs; eight choices match highest restored validation Dice; trial records excluded. |
| Segmentation split | 1,816/389/389 unique IDs with disjoint membership and 2,594 total. |
| Segmentation consistency | All full runs have matching protocol, dataset and initial-weight hashes. |
| Test table arithmetic | Eight saved summaries agree with means of defined per-image metrics and the archived final table; each contains all 389 held-out IDs. |
| Record preservation | Reporting writes only derived CSV tables to ignored reports/. |

Verification used the bundled Python runtime for data-free checks. It does not
establish that a newly installed TensorFlow environment reproduces training.
No model was fitted, no image inference was run and no final-test model evaluation
was repeated. The ISIC dataset and archived checkpoints are not part of this
publishing copy. Original archive records preserve their experimental provenance.

Re-run the checks from the repository root:

```bash
python tools/verify_repository.py
python -m unittest discover -s segmentation -p "test_*.py" -v
python tools/summarize_results.py --check-only
```

Original results were not regenerated. Recovery test scores can be checked
against saved records here, but recomputing them from images requires the omitted
weights and runtime. Segmentation metric aggregation checks use saved per-image
values; they do not substitute for rechecking against the original ground truth.
