"""Validate and export saved recovery/segmentation tables; no ML dependencies."""
from __future__ import annotations

import argparse
import csv
import io
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OPTIMIZERS = ('Adam', 'Adagrad', 'RMSProp', 'SGD-Nesterov')
RATES = (0.0001, 0.001, 0.01, 0.1)
METRICS = ('dice', 'iou', 'precision', 'accuracy', 'specificity', 'sensitivity')


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read_json(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def read_csv(path):
    return list(csv.DictReader(io.StringIO(path.read_text(encoding='utf-8-sig'))))


def close(a, b):
    return math.isclose(float(a), float(b), rel_tol=1e-10, abs_tol=1e-12)


def run_basename(value):
    name = str(value).replace('\\', '/').rstrip('/').rsplit('/', 1)[-1]
    require(name not in {'', '.', '..'}, 'Invalid run directory')
    return name


def pair_map(rows):
    pairs = {(r['optimizer'], r['loss']): r for r in rows}
    require(len(pairs) == len(rows), 'Duplicate optimizer-loss entries')
    return pairs


def check_grid(rows, losses):
    expected = {(o, l, rate) for o in OPTIMIZERS for l in losses for rate in RATES}
    actual = {(r['optimizer'], r['loss'], float(r['learning_rate'])) for r in rows}
    require(len(rows) == len(expected) and actual == expected,
            'Screening records do not contain the complete unique expected grid')
    require(all(r['status'] == 'completed' for r in rows), 'A full screening run is incomplete')


def recovery_table(root=ROOT):
    base = root / 'recovery/results/stage5_loss_comparison'
    runs = [read_json(p) for p in sorted((base/'runs').glob('*/result.json'))]
    check_grid(runs, ('MSE', 'MAE', 'Huber'))
    selected = pair_map(read_json(base/'selected_learning_rates.json'))
    table = read_csv(base/'final_results.csv')
    final = pair_map(table)
    require(len(selected) == len(final) == 12, 'Expected twelve recovery comparisons')
    for pair, row in final.items():
        group = [r for r in runs if (r['optimizer'], r['loss']) == pair]
        require(group and all(math.isfinite(r['validation_mse']) for r in group),
                f'Invalid recovery validation MSE: {pair}')
        best = min(group, key=lambda r: (r['validation_mse'], r['learning_rate']))
        require(best['run_id'] == selected[pair]['run_id'], f'Recovery selection differs: {pair}')
        require(close(row['best_learning_rate'], best['learning_rate']), f'Recovery rate differs: {pair}')
        require(close(row['training_time_seconds'], best['training_time_seconds']), 'Recovery fit time differs')
        require(int(row['epochs_run']) == best['epochs_run'], 'Recovery epoch count differs')
        require(close(float(row['test_rmse'])**2, row['test_mse']), 'Recovery RMSE and MSE disagree')
    return table


def segmentation_table(root=ROOT):
    base = root / 'segmentation/results/stage7_isic_segmentation'
    records = [(p.parent.name, read_json(p)) for p in sorted(base.glob('*/run.json'))]
    full = [(name, r) for name, r in records if r['protocol']['maximum_epochs'] > 1]
    check_grid([r for _, r in full], ('BCE', 'Dice'))
    require(len({r['initial_weights_sha256'] for _, r in full}) == 1, 'Mixed segmentation initializations')
    require(len({r['protocol_sha256'] for _, r in full}) == 1, 'Mixed segmentation protocols')
    require(len({r['dataset_ids_sha256'] for _, r in full}) == 1, 'Mixed segmentation datasets')
    split_files = {run_basename(r['split_file']) for _, r in full}
    require(len(split_files) == 1, 'Mixed segmentation split files')
    split = read_json(base/next(iter(split_files)))['split']
    sets = [set(split[k]) for k in ('train','validation','test')]
    require([len(split[k]) for k in ('train','validation','test')] == [1816,389,389], 'Wrong split counts')
    require([len(s) for s in sets] == [1816,389,389], 'Duplicate split IDs')
    require(len(set.union(*sets)) == 2594, 'Segmentation splits overlap')
    selected = pair_map(read_json(base/'selected_runs.json'))
    archive = pair_map(read_csv(base/'final_segmentation_results.csv'))
    require(len(selected) == len(archive) == 8, 'Expected eight segmentation selections')
    table = []
    for optimizer in OPTIMIZERS:
        for loss in ('BCE','Dice'):
            pair = optimizer, loss
            group = [(n,r) for n,r in full if (r['optimizer'],r['loss']) == pair]
            require(all(math.isfinite(r['restored_validation']['dice']) for _,r in group), 'Nonfinite validation Dice')
            name, best = max(group, key=lambda t:(t[1]['restored_validation']['dice'], -t[1]['learning_rate']))
            require(run_basename(selected[pair]['run_dir']) == name, f'Segmentation selection differs: {pair}')
            summary = read_json(base/name/'test_summary.json')
            per_image = read_csv(base/name/'test_metrics.csv')
            require(len(per_image) == summary['number_of_test_images'] == 389, 'Expected 389 test rows')
            require({r['image_id'] for r in per_image} == sets[2], 'Wrong test IDs in saved metrics')
            row = dict(optimizer=optimizer, loss=loss,
                       best_learning_rate=best['learning_rate'],
                       validation_dice=best['restored_validation']['dice'])
            for metric in METRICS:
                values = [float(r[metric]) for r in per_image if r[metric] != '']
                mean = sum(values)/len(values) if values else None
                reported = summary['macro_scores'][metric]
                require((mean is None and reported is None) or close(mean, reported),
                        f'Saved macro metric differs: {pair} {metric}')
                row['test_'+metric] = reported
            row.update(training_time_seconds=best['fit_seconds'], epochs_run=best['epochs_run'])
            for key, value in row.items():
                if key in {'optimizer','loss'}:
                    require(archive[pair][key] == value, f'Wrong final row {pair}')
                else:
                    require(close(archive[pair][key], value), f'Final table differs: {pair} {key}')
            table.append(row)
    return table


def write_table(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, default=ROOT/'reports')
    parser.add_argument('--check-only', action='store_true')
    args = parser.parse_args()
    recovery = recovery_table()
    segmentation = segmentation_table()
    if not args.check_only:
        destination = args.output_dir.resolve()
        for folder in ('recovery','segmentation','legacy_recovery'):
            require(not destination.is_relative_to((ROOT/folder).resolve()),
                    'Derived reports must not overwrite imported experiment folders')
        write_table(destination/'recovery_test_results.csv', recovery)
        write_table(destination/'segmentation_test_results.csv', segmentation)
        print(f'Tables saved to {destination}')
    print('PASS: 48 recovery screening records / 12 final rows; 32 segmentation screening records / 8 final rows.')
    print('Selection and saved-table agreement verified. No training or image inference occurred.')


if __name__ == '__main__':
    main()
