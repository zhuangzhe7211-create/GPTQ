"""Export the measured per-layer paired CE effects and kernel costs."""
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

root = Path(__file__).resolve().parent
summary = json.loads((root/'round3_summary.json').read_text())
families = ['gptaq', 'square4', 'weak4', 'weak1', 'square1', 'rms1']
colors = ['#596579', '#d75c4c', '#d7a642', '#4484ae', '#508966', '#835eac']
fig, axes = plt.subplots(1, 3, figsize=(13, 4.8), sharey=True)
for ax, layer in zip(axes, ('6', '12', '18')):
    for index, (family, color) in enumerate(zip(families, colors)):
        row = summary['layers'][layer]['methods'][family]
        mean = row['mean_delta_ce']*1000
        lo, hi = np.array(row['exploratory_95_interval'])*1000
        ax.errorbar(mean, index, xerr=[[mean-lo], [hi-mean]], fmt='o', color=color, capsize=4)
    ax.axvline(0, color='#777777', linewidth=.8, linestyle='--')
    ax.set_title(f'Layer index {layer}')
    ax.set_yticks(range(len(families)), families)
    ax.set_xlabel('CE difference vs tuned ResComp (x 1,000)\nNegative favors the candidate')
    ax.grid(axis='x', alpha=.2)
axes[0].invert_yaxis()
fig.suptitle('Fresh held-out articles: paired per-layer effects\nExploratory 95% article bootstrap intervals; three calibration seeds', fontsize=12)
fig.tight_layout()
destination = root.parent/'reports/figures'
destination.mkdir(exist_ok=True)
fig.savefig(destination/'round3_effects.png', dpi=180)
fig.savefig(destination/'round3_effects.svg')
plt.close(fig)
fig, ax = plt.subplots(figsize=(8,4))
names = ['rescomp', *families]
seconds = [np.mean([summary['layers'][str(l)]['methods'][f]['mean_quantization_seconds'] for l in (6,12,18)]) for f in names]
ax.bar(names, seconds, color=['#333333',*colors])
ax.set_ylabel('Mean module quantization time (seconds)')
ax.set_title('Kernel cost only; teacher-gradient collection is additional')
ax.grid(axis='y', alpha=.2)
fig.tight_layout()
fig.savefig(destination/'round3_cost.png', dpi=180)
plt.close(fig)
print(destination)
