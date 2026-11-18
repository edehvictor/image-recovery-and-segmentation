"""Compare 4 optimizers x 3 losses for Shepp-Logan image recovery.

Phase 1 (--screen): for each of the 12 optimizer-loss pairs, train 4 learning
rates on saved Fold 1 (340 train / 85 validation). Each pair's learning rate is
the one with the lowest restored validation MSE (ties: smaller rate).

Phase 2 (--final): evaluate each pair's selected model ONCE on the fresh test
IDs 500-574 and report MSE, RMSE, PSNR, SSIM and training time.

Default execution is a preflight check only. Finished runs are skipped on
re-run, so an interrupted screen can simply be started again.
Reuses stage4_lr_sensitivity.py and experiment_config.py without changing them.
"""

import argparse
import json
import math
import time

import numpy as np

import experiment_config as config
import stage4_lr_sensitivity as s4


OUTPUT = config.PROJECT_ROOT / "results" / "stage5_loss_comparison"
COMBOS = [(o, l) for l in config.RECOVERY_LOSSES for o in config.OPTIMIZERS]

RESULT_FIELDS = (
    "run_id", "optimizer", "loss", "learning_rate", "status", "validation_mse",
    "validation_psnr", "epochs_run", "training_time_seconds", "error",
)
FINAL_FIELDS = (
    "optimizer", "loss", "best_learning_rate", "test_mse", "test_rmse",
    "test_psnr", "test_ssim", "training_time_seconds", "epochs_run",
)


def run_one(tf, splits, clean_image, initial_weights, optimizer, loss, lr):
    """Train one optimizer/loss/rate from the shared initial weights."""
    run_id = f"{optimizer.lower()}_{loss.lower()}_lr_{lr:.0e}"
    directory = OUTPUT / "runs" / run_id
    result_file = directory / "result.json"
    if result_file.exists():
        print(f"Skipping {run_id}: already done")
        return json.loads(result_file.read_text())

    directory.mkdir(parents=True, exist_ok=True)
    row = dict.fromkeys(RESULT_FIELDS)
    row.update(run_id=run_id, optimizer=optimizer, loss=loss,
               learning_rate=lr, status="completed", error="")
    recorder = s4.make_recorder(tf)
    model = None
    try:
        model = s4.build_fresh_model(tf)
        model.set_weights(initial_weights)
        model.compile(
            optimizer=config.make_optimizer(optimizer, lr),
            loss=config.make_loss(loss),
            metrics=[tf.keras.metrics.MeanSquaredError(name="mse"),
                     s4.make_psnr_metric(tf)],
        )
        train = s4.make_screening_dataset(tf, splits.training, splits, clean_image, True)
        val = s4.make_screening_dataset(tf, splits.validation, splits, clean_image, False)
        early_stopping = tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", mode="min", patience=config.EARLY_STOPPING_PATIENCE,
            min_delta=config.MIN_DELTA, restore_best_weights=config.RESTORE_BEST_WEIGHTS,
        )
        start = time.perf_counter()
        try:
            model.fit(train, validation_data=val, epochs=config.MAX_EPOCHS,
                      callbacks=[recorder, early_stopping], verbose=1, shuffle=False)
        finally:
            row["training_time_seconds"] = time.perf_counter() - start
        restored = model.evaluate(val, verbose=0, return_dict=True)
        if not np.isfinite(restored["mse"]):
            raise s4.DivergingRun("Non-finite restored validation MSE")
        row["validation_mse"] = float(restored["mse"])
        row["validation_psnr"] = float(restored["psnr_metric"])
        model.save_weights(directory / "best.weights.h5")
    except s4.DivergingRun as exc:
        # A divergent rate (e.g. SGD at 0.1) is a valid result, not a crash.
        row.update(status="diverged", error=str(exc))
    finally:
        row["epochs_run"] = len(recorder.rows)
        fields = list(recorder.rows[0]) if recorder.rows else list(s4.HISTORY_FIELDS)
        s4.save_csv(directory / "history.csv", fields, recorder.rows)
        del model
        tf.keras.backend.clear_session()

    s4.save_json(result_file, row)  # written last, so an interrupted run is redone
    return row


def screen(checks):
    tf, splits, initial_weights, _, _ = checks
    clean_image = s4.make_clean_image(tf)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    rows = []
    for optimizer, loss in COMBOS:
        for lr in config.LEARNING_RATES:
            print(f"\n=== {optimizer} | {loss} | lr {lr:g} ===")
            rows.append(run_one(tf, splits, clean_image, initial_weights, optimizer, loss, lr))
            s4.save_csv(OUTPUT / "screening_results.csv", RESULT_FIELDS, rows)

    selected = []
    for optimizer, loss in COMBOS:
        done = [r for r in rows if r["optimizer"] == optimizer
                and r["loss"] == loss and r["status"] == "completed"]
        if not done:
            raise RuntimeError(f"No completed run for {optimizer} / {loss}")
        selected.append(min(done, key=lambda r: (r["validation_mse"], r["learning_rate"])))
    s4.save_json(OUTPUT / "selected_learning_rates.json", selected)
    print("\nSelected learning rates:")
    for r in selected:
        print(f"  {r['optimizer']:13s} {r['loss']:6s} lr={r['learning_rate']:g}  "
              f"val MSE={r['validation_mse']:.6f}")
    return 0


def make_test_dataset(tf, clean_image):
    """Fresh final-test IDs 500-574, same per-ID noise recipe as training."""
    ids = np.asarray(config.FINAL_TEST_IDS, dtype=np.int32)

    def pair(index):
        seed = tf.stack((tf.constant(config.SEED, tf.int32), tf.cast(index, tf.int32)))
        noise = tf.random.stateless_normal(
            tf.shape(clean_image), seed=seed,
            mean=config.GAUSSIAN_MEAN, stddev=config.GAUSSIAN_STD, dtype=tf.float32)
        noisy = tf.clip_by_value(clean_image + noise, config.PIXEL_MIN, config.PIXEL_MAX)
        return noisy, clean_image

    return tf.data.Dataset.from_tensor_slices(ids).map(pair).batch(1)


def score(tf, model, dataset):
    """Mean MSE, RMSE, PSNR and SSIM. model=None scores the noisy input itself."""
    mses, psnrs, ssims = [], [], []
    for noisy, clean in dataset:
        output = noisy if model is None else model(noisy, training=False)
        mses.append(float(tf.reduce_mean(tf.square(clean - output))))
        psnrs.append(float(tf.image.psnr(clean, output, max_val=1.0)[0]))
        ssims.append(float(tf.image.ssim(clean, output, max_val=1.0)[0]))
    mse = float(np.mean(mses))
    return {"mse": mse, "rmse": math.sqrt(mse),
            "psnr": float(np.mean(psnrs)), "ssim": float(np.mean(ssims))}


def final(checks):
    tf, _, _, _, _ = checks
    selected_file = OUTPUT / "selected_learning_rates.json"
    if not selected_file.exists():
        raise FileNotFoundError("Run --screen first; no selected_learning_rates.json")
    selected = json.loads(selected_file.read_text())

    test = make_test_dataset(tf, s4.make_clean_image(tf))
    baseline = score(tf, None, test)
    print(f"Noisy input baseline: {baseline}")

    rows = []
    for r in selected:
        model = s4.build_fresh_model(tf)
        model.load_weights(OUTPUT / "runs" / r["run_id"] / "best.weights.h5")
        m = score(tf, model, test)
        rows.append({
            "optimizer": r["optimizer"], "loss": r["loss"],
            "best_learning_rate": r["learning_rate"],
            "test_mse": m["mse"], "test_rmse": m["rmse"],
            "test_psnr": m["psnr"], "test_ssim": m["ssim"],
            "training_time_seconds": r["training_time_seconds"],
            "epochs_run": r["epochs_run"],
        })
        print(f"{r['optimizer']:13s} {r['loss']:6s} lr={r['learning_rate']:g}  "
              f"MSE={m['mse']:.6f} RMSE={m['rmse']:.5f} "
              f"PSNR={m['psnr']:.3f} SSIM={m['ssim']:.4f}")
        del model
        tf.keras.backend.clear_session()

    s4.save_csv(OUTPUT / "final_results.csv", FINAL_FIELDS, rows)
    s4.save_json(OUTPUT / "noisy_baseline.json", baseline)
    print(f"\nSaved {OUTPUT / 'final_results.csv'}")
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--screen", action="store_true",
                      help="Phase 1: 48 learning-rate screening runs on Fold 1")
    mode.add_argument("--final", action="store_true",
                      help="Phase 2: one test evaluation per pair on IDs 500-574")
    args = parser.parse_args()

    checks = s4.preflight()  # split checks, optimizer checks, shared initial weights
    if args.screen:
        return screen(checks)
    if args.final:
        return final(checks)
    print(f"Preflight complete: {len(COMBOS)} pairs x {len(config.LEARNING_RATES)} "
          f"rates = {len(COMBOS) * len(config.LEARNING_RATES)} runs. Nothing trained.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())