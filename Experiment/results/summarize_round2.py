"""Descriptive round-2 summary and exploratory article-cluster paired bootstrap."""
from collections import defaultdict
import json
from pathlib import Path
import random
import statistics as st

root = Path(__file__).resolve().parent
runs = [json.loads((root / f'round2_s{s}/metrics.json').read_text()) for s in range(3)]
selection = json.loads((root / 'round2_test/selection.json').read_text())
test = json.loads((root / 'round2_test/metrics.json').read_text())
manifest = json.loads((root / 'round2_test/test_manifest.json').read_text())
assert test['_summary']['selection_frozen_before_test']
for seed, run in enumerate(runs):
    old = json.loads((root / f'20260926_qwen05b_w4_s{seed}/metrics.json').read_text())
    assert run['teacher_a0.25']['ce'] == old['task_weighted_rescomp']['ce']
    assert run['rescomp_a0.25']['ce'] == old['rescomp_official_core']['ce']

summary = {'validation': {}, 'test': {}, 'shuffle_validation': {}, 'paired_comparisons': {},
           'round1_reproduction': 'exact validation CE match for all three seeds',
           'sweep_seconds': sum(m['_summary']['total_seconds'] for m in runs)}
for family, chosen in selection['selected'].items():
    summary['validation'][family] = {
        'selected': chosen,
        'mean_ppl': st.mean(m[chosen]['ppl'] for m in runs),
        'mean_quantization_seconds': st.mean(m[chosen]['quantization_seconds'] for m in runs),
    }
    summary['test'][family] = {
        'seed_ppl': [test[f's{s}/{family}']['ppl'] for s in range(3)],
        'mean_ppl': st.mean(test[f's{s}/{family}']['ppl'] for s in range(3)),
        'mean_ce': st.mean(test[f's{s}/{family}']['ce'] for s in range(3)),
    }
for family in ('teacher', 'prefix', 'shared'):
    aligned = [m[f'{family}_a0.25']['ce'] for m in runs]
    shuffled = [m[f'{family}_shuffle{r}_a0.25']['ce'] for m in runs for r in (100, 101, 102)]
    summary['shuffle_validation'][family] = {
        'aligned_mean_ce': st.mean(aligned), 'shuffled_mean_ce': st.mean(shuffled),
        'aligned_wins_out_of_9': sum(m[f'{family}_a0.25']['ce'] < m[f'{family}_shuffle{r}_a0.25']['ce']
                                     for m in runs for r in (100, 101, 102))}

# Average seeds within a window first; they are not independent test examples.
# Resample article clusters to avoid treating paragraphs from one article as independent.
for family, baseline in [('teacher', 'rescomp'), ('prefix', 'rescomp'), ('shared', 'rescomp'),
                         ('gptaq', 'rescomp'), ('fixed_teacher', 'fixed_rescomp'), ('rescomp', 'fixed_rescomp')]:
    delta = [st.mean(test[f's{s}/{family}']['sequence_ce'][i] - test[f's{s}/{baseline}']['sequence_ce'][i]
                     for s in range(3)) for i in range(len(manifest['windows']))]
    clusters = defaultdict(list)
    for item, value in zip(manifest['windows'], delta):
        clusters[item['article_heading_row']].append(value)
    items = [(sum(values), len(values)) for values in clusters.values()]
    rng = random.Random(20260926)
    replicates = []
    for _ in range(5000):
        sampled = rng.choices(items, k=len(items))
        replicates.append(sum(x for x, n in sampled) / sum(n for x, n in sampled))
    replicates.sort()
    summary['paired_comparisons'][f'{family}_minus_{baseline}'] = {
        'delta_ce': st.mean(delta), 'article_clusters': len(items),
        'exploratory_95_percentile_interval': [replicates[124], replicates[4874]],
        'warning': 'Exploratory conditional-on-calibration interval; no multiple-comparison correction or general superiority claim'}
(root / 'round2_summary.json').write_text(json.dumps(summary, indent=2, allow_nan=False), encoding='utf-8')
print(json.dumps(summary, indent=2))
