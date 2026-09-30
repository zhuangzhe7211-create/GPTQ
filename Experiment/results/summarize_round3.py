"""Recompute round-3 comparisons; same article/seed matching for every method."""
import csv
import json
from pathlib import Path
import statistics as st

import numpy as np

root = Path(__file__).resolve().parent
test = json.loads((root/'round3_holdout/metrics.json').read_text())
selection = json.loads((root/'round3_holdout/selection.json').read_text())
manifest = json.loads((root/'round3_holdout/manifest.json').read_text())
assert test['_summary']['all_selections_frozen_before_evaluation']
assert len({w['article_heading_row'] for w in manifest['windows']}) == 128
families = list(selection['selected']['6'])
bootstrap_indices = np.random.default_rng(20260927).integers(0, 128, size=(5000, 128))


def interval(delta):
    mean = float(np.mean(delta))
    lower, upper = np.quantile(np.asarray(delta)[bootstrap_indices].mean(1), [.025, .975])
    return {'mean_delta_ce': mean, 'exploratory_95_interval': [float(lower), float(upper)]}


summary = {'layers': {}, 'across_layer_mean_contrasts': {}, 'shuffle_validation': {}, 'controlled_ablation': {},
           'calibration_samples': 64, 'holdout_articles': 128, 'scored_tokens_per_candidate': 32640,
           'warning': 'Exploratory paired article bootstrap conditional on these calibrations/layers. No multiplicity correction.',
           'sweep_seconds': 0., 'holdout_seconds': test['_summary']['evaluation_seconds']}
all_deltas = {f: [] for f in families if f != 'rescomp'}
rows = []
for layer in (6, 12, 18):
    runs = [json.loads((root/f'round3_l{layer}_s{s}/metrics.json').read_text()) for s in range(3)]
    diagnostics = [json.loads((root/f'round3_l{layer}_s{s}/diagnostics.json').read_text()) for s in range(3)]
    assert all(m['_summary']['all_one_regression'] == 'passed' for m in runs)
    summary['sweep_seconds'] += sum(m['_summary']['total_seconds'] for m in runs)
    layer_summary = {'methods': {}, 'relative_input_error': st.mean(d['relative_input_error'] for d in diagnostics),
                     'mean_teacher_gradient_seconds': st.mean(d['teacher_gradient_seconds'] for d in diagnostics),
                     'prefix_holdout_ppl': test[f'l{layer}/common_rtn_prefix']['ppl']}
    base = np.array([test[f'l{layer}/s{s}/rescomp']['sequence_ce'] for s in range(3)])
    for family in families:
        key = selection['selected'][str(layer)][family]
        values = [test[f'l{layer}/s{s}/{family}'] for s in range(3)]
        differences = np.array([v['sequence_ce'] for v in values]) - base
        delta = differences.mean(0)
        value = {
            'selected': key, 'seed_holdout_ppl': [v['ppl'] for v in values],
            'mean_holdout_ppl': st.mean(v['ppl'] for v in values),
            'mean_holdout_ce': st.mean(v['ce'] for v in values),
            'mean_validation_ppl': st.mean(m[key]['ppl'] for m in runs),
            'seed_delta_ce_vs_rescomp': differences.mean(1).tolist(),
            'wins_over_rescomp': int((differences.mean(1) < 0).sum()),
            'mean_quantization_seconds': st.mean(m[key]['quantization_seconds'] for m in runs),
            **interval(delta),
        }
        layer_summary['methods'][family] = value
        rows.append({'layer': layer, 'family': family, **{k: v for k, v in value.items() if not isinstance(v, list)}})
        if family != 'rescomp':
            all_deltas[family].append(delta)
    # Equal rho/group/alpha ablation on validation isolates the RMS transformation.
    summary['shuffle_validation'][str(layer)] = {
        'rms_aligned_ce': st.mean(m['rms1_a0.5']['ce'] for m in runs),
        'rms_shuffled_ce': st.mean(m[f'rms1_shuffle{s}_a0.5']['ce'] for m in runs for s in (100, 101, 102)),
        'rms_wins_out_of_9': sum(m['rms1_a0.5']['ce'] < m[f'rms1_shuffle{s}_a0.5']['ce'] for m in runs for s in (100, 101, 102)),
        'rms_minus_square_same_alpha_ce': st.mean(m['rms1_a0.5']['ce']-m['square1_a0.5']['ce'] for m in runs)}
    summary['layers'][str(layer)] = layer_summary
for family, arrays in all_deltas.items():
    summary['across_layer_mean_contrasts'][family] = {
        **interval(np.mean(arrays, axis=0)),
        'wins_out_of_9': sum(summary['layers'][str(l)]['methods'][family]['wins_over_rescomp'] for l in (6, 12, 18)),
        'note': 'Equal-weight mean of paired CE contrasts, not full-model PPL'}
for candidate, control in [('rms1', 'square1'), ('weak4', 'square4')]:
    values = []
    ablation = {}
    for layer in (6, 12, 18):
        matched = selection['selected'][str(layer)][candidate].split('_a')[-1] == selection['selected'][str(layer)][control].split('_a')[-1]
        difference = np.mean([np.array(test[f'l{layer}/s{s}/{candidate}']['sequence_ce']) -
                              np.array(test[f'l{layer}/s{s}/{control}']['sequence_ce']) for s in range(3)], axis=0)
        ablation[str(layer)] = {**interval(difference), 'matched_selected_alpha': matched}
        values.append(difference)
    ablation['mean_across_layers'] = interval(np.mean(values, axis=0))
    summary['controlled_ablation'][f'{candidate}_minus_{control}'] = ablation
(root/'round3_summary.json').write_text(json.dumps(summary, indent=2, allow_nan=False), encoding='utf-8')
with (root/'round3_summary.csv').open('w', newline='', encoding='utf-8-sig') as stream:
    writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
    writer.writeheader()
    writer.writerows(rows)
print(json.dumps(summary, indent=2))
