"""Post-processing for stage7 segmentation runs.

  python stage7_summary.py select     # best learning rate per optimizer-loss pair
  python stage7_summary.py evaluate --images ... --masks ...   # test the selected runs once
  python stage7_summary.py collect    # write the final results table

Selection uses validation Dice only; test data is never touched before 'evaluate'.
"""

import argparse
import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent / "results" / "stage7_isic_segmentation"
SELECTED = ROOT / "selected_runs.json"


def resolve_run_dir(item):
    """Resolve an archived run locally, even if saved on another computer."""
    name = str(item["run_dir"]).replace("\\", "/").rstrip("/").rsplit("/", 1)[-1]
    if name in {"", ".", ".."}:
        raise ValueError("Invalid saved run directory")
    run_dir = ROOT / name
    record = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
    for key in ("optimizer", "loss", "learning_rate"):
        if record[key] != item[key]:
            raise ValueError(f"Selected run {name} has a different {key}")
    return run_dir


def load_runs():
    runs = []
    for file in sorted(ROOT.glob("*/run.json")):
        record = json.loads(file.read_text(encoding="utf-8"))
        # Skip failed runs and the 1-epoch trial protocol.
        if record.get("status") != "completed" or record["protocol"]["maximum_epochs"] <= 1:
            continue
        record["run_dir"] = str(file.parent)
        runs.append(record)
    return runs


def select():
    runs = load_runs()
    print(f"{len(runs)} completed full runs found\n")
    selected = []
    for optimizer in ("Adam", "Adagrad", "RMSProp", "SGD-Nesterov"):
        for loss in ("BCE", "Dice"):
            group = [r for r in runs if r["optimizer"] == optimizer and r["loss"] == loss]
            for r in sorted(group, key=lambda r: r["learning_rate"]):
                print(f"  {optimizer:13s} {loss:5s} lr={r['learning_rate']:<7g} "
                      f"val Dice={r['restored_validation']['dice']:.4f} epochs={r['epochs_run']}")
            if not group:
                print(f"  {optimizer} {loss}: NO COMPLETED RUNS")
                continue
            best = max(group, key=lambda r: (r["restored_validation"]["dice"], -r["learning_rate"]))
            selected.append({"optimizer": optimizer, "loss": loss,
                             "learning_rate": best["learning_rate"],
                             "validation_dice": best["restored_validation"]["dice"],
                             "run_dir": Path(best["run_dir"]).name})
            print(f"  --> selected lr={best['learning_rate']:g}\n")
    SELECTED.write_text(json.dumps(selected, indent=2), encoding="utf-8")
    print(f"Saved {SELECTED} ({len(selected)} selected runs; expected 8)")


def evaluate(images, masks):
    import stage7_isic_segmentation as stage7  # imported here so select/collect need no TensorFlow
    for item in json.loads(SELECTED.read_text(encoding="utf-8")):
        run_dir = resolve_run_dir(item)
        if (run_dir / "test_summary.json").exists():
            print(f"Already evaluated: {run_dir.name}")
            continue
        print(f"Evaluating {item['optimizer']} | {item['loss']} | lr {item['learning_rate']:g}")
        stage7.final_evaluate(images, masks, run_dir)


def collect():
    rows = []
    for item in json.loads(SELECTED.read_text(encoding="utf-8")):
        run_dir = resolve_run_dir(item)
        summary = json.loads((run_dir / "test_summary.json").read_text(encoding="utf-8"))
        record = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
        macro = summary["macro_scores"]
        rows.append({
            "optimizer": item["optimizer"], "loss": item["loss"],
            "best_learning_rate": item["learning_rate"],
            "validation_dice": item["validation_dice"],
            "test_dice": macro["dice"], "test_iou": macro["iou"],
            "test_precision": macro["precision"], "test_accuracy": macro["accuracy"],
            "test_specificity": macro["specificity"], "test_sensitivity": macro["sensitivity"],
            "training_time_seconds": record["fit_seconds"], "epochs_run": record["epochs_run"],
        })
    out = ROOT / "final_segmentation_results.csv"
    with out.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    for r in rows:
        print(f"{r['optimizer']:13s} {r['loss']:5s} lr={r['best_learning_rate']:<7g} "
              f"Dice={r['test_dice']:.4f} IoU={r['test_iou']:.4f}")
    print(f"\nSaved {out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("step", choices=("select", "evaluate", "collect"))
    parser.add_argument("--images", type=Path)
    parser.add_argument("--masks", type=Path)
    args = parser.parse_args()
    if args.step == "select":
        select()
    elif args.step == "evaluate":
        if not (args.images and args.masks):
            parser.error("evaluate needs --images and --masks")
        evaluate(args.images, args.masks)
    else:
        collect()
