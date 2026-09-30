"""Article-paired round-4 analysis; run only after the frozen holdout completes."""
import csv
import json
from pathlib import Path
import statistics as st

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

root = Path(__file__).resolve().parent
test = json.loads((root/'round4_holdout/metrics.json').read_text())
selection = json.loads((root/'round4_holdout/selection.json').read_text())
assert test['_summary']['all_selections_frozen_before_evaluation']
bootstrap = np.random.default_rng(20260928).integers(0,128,size=(10000,128))


def contrast(array):
    per_article = np.asarray(array).mean(0)
    lo,hi = np.quantile(per_article[bootstrap].mean(1),[.025,.975])
    delta = float(per_article.mean())
    return {'delta_ce':delta,'ci95':[float(lo),float(hi)],
            'geometric_relative_ppl_percent':float(100*np.expm1(delta)),
            'seed_delta_ce':np.asarray(array).mean(1).tolist(),
            'wins':int((np.asarray(array).mean(1)<0).sum())}


summary = {'contexts':{},'prefix':{},'sweep_seconds':0.,
           'holdout_seconds':test['_summary']['evaluation_seconds'],
           'fp_holdout_ppl':test['floating_point']['ppl'],
           'warning':'Exploratory article-paired intervals conditional on three calibration seeds; no multiplicity correction.'}
rows=[]
for context,selected in selection['selected'].items():
    runs=[json.loads((root/f'round4_{context}_s{s}/metrics.json').read_text()) for s in range(3)]
    diagnostics=[json.loads((root/f'round4_{context}_s{s}/diagnostics.json').read_text()) for s in range(3)]
    summary['sweep_seconds']+=sum(r['_summary']['total_seconds'] for r in runs)
    arrays={f:np.array([test[f'{context}/s{s}/{f}']['sequence_ce'] for s in range(3)]) for f in selected}
    item={'input_drift':st.mean(d['relative_input_error'] for d in diagnostics),
          'teacher_gradient_seconds':st.mean(d['teacher_gradient_seconds'] for d in diagnostics),'methods':{}}
    # Diagnostic only: development ranking, not another held-out selection rule.
    candidate_keys=[k for k in runs[0] if '_a' in k and not k.startswith('_')]
    item['development_proxy_diagnostic']={metric:[float(np.corrcoef(
        [r[k][metric] for k in candidate_keys],[r[k]['ce'] for k in candidate_keys])[0,1]) for r in runs]
        for metric in ('calibration_mse','calibration_task_mse','validation_mse')}
    item['development_metric_minimizers']={metric:min(candidate_keys,key=lambda k:st.mean(r[k][metric] for r in runs))
        for metric in ('calibration_mse','calibration_task_mse','validation_mse','ce')}
    for family,key in selected.items():
        values=[test[f'{context}/s{s}/{family}'] for s in range(3)]
        result={'selected':key,'mean_ppl':st.mean(v['ppl'] for v in values),
                'seed_ppl':[v['ppl'] for v in values],
                'validation_ppl':st.mean(r[key]['ppl'] for r in runs),
                'kernel_seconds':st.mean(r[key]['quantization_seconds'] for r in runs),
                'vs_rescomp':contrast(arrays[family]-arrays['rescomp']),
                'vs_gptaq':contrast(arrays[family]-arrays['gptaq']),
                'vs_gptq_target':contrast(arrays[family]-arrays['gptq_reference'])}
        item['methods'][family]=result
        rows.append({'context':context,'family':family,'selected':key,'mean_ppl':result['mean_ppl'],
                     'kernel_seconds':result['kernel_seconds'],**result['vs_rescomp']})
    summary['contexts'][context]=item
for prefix in ('rtn','gptq'):
    summary['prefix'][prefix]={'seed_ppl':[test[f'{prefix}/s{s}/prefix']['ppl'] for s in range(3)],
                               'mean_ppl':st.mean(test[f'{prefix}/s{s}/prefix']['ppl'] for s in range(3))}
summary['prefix']['gptq_minus_rtn']=contrast(np.array([test[f'gptq/s{s}/prefix']['sequence_ce'] for s in range(3)])-
                                                       np.array([test[f'rtn/s{s}/prefix']['sequence_ce'] for s in range(3)]))
summary['prefix_build_seconds']=sum(json.loads((root/f'round4_prefix_s{s}/manifest.json').read_text())['build_seconds'] for s in range(3))
summary['continuation_gate']={}
for family in ('weak1','rms1'):
    checks={}
    for module in ('attn','mlp'):
        c=summary['contexts'][f'gptq_{module}']['methods'][family]['vs_rescomp']
        checks[module]={'improves':c['delta_ce']<0,'wins_at_least_2':c['wins']>=2,
                        'at_least_0_1_percent':c['geometric_relative_ppl_percent']<=-.1,'upper_ci_negative':c['ci95'][1]<0}
    summary['continuation_gate'][family]={'checks':checks,'passed':all(all(v.values()) for v in checks.values())}
(root/'round4_summary.json').write_text(json.dumps(summary,indent=2,allow_nan=False),encoding='utf-8')
with (root/'round4_summary.csv').open('w',newline='',encoding='utf-8-sig') as stream:
    writer=csv.DictWriter(stream,fieldnames=rows[0].keys());writer.writeheader();writer.writerows(rows)
fig,axes=plt.subplots(1,2,figsize=(11,4.5),layout='constrained')
labels=['RTN / attention','RTN / MLP','GPTQ / attention','GPTQ / MLP']
for offset,(family,color) in enumerate([('weak1','#157a8c'),('rms1','#cf6b24')]):
    x=np.arange(4)+(offset-.5)*.16
    deltas=np.array([summary['contexts'][c]['methods'][family]['vs_rescomp']['delta_ce'] for c in selection['selected']])*1e3
    ci=np.array([summary['contexts'][c]['methods'][family]['vs_rescomp']['ci95'] for c in selection['selected']])*1e3
    axes[0].errorbar(x,deltas,yerr=np.array([deltas-ci[:,0],ci[:,1]-deltas]),fmt='o',capsize=4,color=color,label=family)
axes[0].axhline(0,color='gray',lw=1);axes[0].set_xticks(range(4),labels,rotation=20,ha='right')
axes[0].set_ylabel('Delta CE vs tuned ResComp (x 0.001)');axes[0].legend()
axes[0].set_title('New 128-article holdout; lower is better')
vals=[summary['fp_holdout_ppl'],summary['prefix']['rtn']['mean_ppl'],summary['prefix']['gptq']['mean_ppl']]
bars=axes[1].bar(['Floating point','RTN prefix','GPTQ prefix'],vals,color=['#777777','#cf6b24','#157a8c'])
axes[1].bar_label(bars,fmt='%.3f',padding=3);axes[1].set_ylabel('Holdout PPL');axes[1].set_ylim(0,max(vals)*1.15)
axes[1].set_title('Only first 12 blocks quantized')
figdir=root.parent/'reports/figures';figdir.mkdir(exist_ok=True)
fig.savefig(figdir/'round4_effects.png',dpi=180);fig.savefig(figdir/'round4_effects.svg');plt.close(fig)
print(json.dumps(summary,indent=2))
