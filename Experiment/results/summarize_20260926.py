"""Recompute descriptive statistics for the three fixed-setting pilot runs."""
import json
import statistics
from pathlib import Path

import torch

root = Path(__file__).resolve().parent
records = []
validation_hashes = None
for seed in range(3):
    folder = root / f'20260926_qwen05b_w4_s{seed}'
    metrics = json.loads((folder / 'metrics.json').read_text())
    config = json.loads((folder / 'config.json').read_text())
    assert metrics['_summary']['all_one_regression'] == 'passed'
    hashes = config['validation_window_hashes']
    if validation_hashes is None:
        validation_hashes = hashes
    assert hashes == validation_hashes, 'Validation windows must match across seeds'
    assert not set(hashes).intersection(config['calibration_window_hashes'])
    base = metrics['rescomp_official_core']
    weighted = metrics['task_weighted_rescomp']
    shuffled = metrics['shuffled_weight_rescomp']
    base_weight = torch.load(folder / 'rescomp_official_core.pt', weights_only=True)['weight']
    task_weight = torch.load(folder / 'task_weighted_rescomp.pt', weights_only=True)['weight']
    record = {
        'seed': seed,
        'ppl': {k: v['ppl'] for k, v in metrics.items() if 'ppl' in v},
        'task_minus_base_ce': weighted['ce'] - base['ce'],
        'task_minus_shuffled_ce': weighted['ce'] - shuffled['ce'],
        'task_relative_ppl_change_percent': 100 * (weighted['ppl'] / base['ppl'] - 1),
        'changed_weight_fraction': float((base_weight != task_weight).float().mean()),
        'task_mse_relative_change_percent': 100 * (weighted['calibration_task_mse'] / base['calibration_task_mse'] - 1),
        'base_quantization_seconds': base['quantization_seconds'],
        'task_quantization_seconds': weighted['quantization_seconds'],
        'peak_cuda_GiB': metrics['_summary']['peak_cuda_bytes'] / 2**30,
        'total_seconds': metrics['_summary']['total_seconds'],
        'sequence_delta_ce': [a - b for a, b in zip(weighted['sequence_ce'], base['sequence_ce'])],
    }
    assert abs(statistics.mean(record['sequence_delta_ce']) - record['task_minus_base_ce']) < 1e-10
    records.append(record)
summary = {
    'runs': records,
    'mean_ppl': {k: statistics.mean(r['ppl'][k] for r in records) for k in records[0]['ppl']},
    'mean_task_minus_base_ce': statistics.mean(r['task_minus_base_ce'] for r in records),
    'mean_task_minus_shuffled_ce': statistics.mean(r['task_minus_shuffled_ce'] for r in records),
    'note': 'Descriptive pilot only. Shared validation data; seeds are not independent test sets.',
}
(root / '20260926_summary.json').write_text(json.dumps(summary, indent=2, allow_nan=False), encoding='utf-8')
print(json.dumps({**summary, 'runs': [{k: v for k, v in r.items() if k != 'sequence_delta_ce'} for r in records]}, indent=2))
