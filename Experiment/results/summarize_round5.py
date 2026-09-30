"""Paired article analysis and resource audit for the fifth round."""
import csv,json
from pathlib import Path
import statistics as st
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

root=Path(__file__).resolve().parent
held=json.loads((root/'round5_holdout/metrics.json').read_text())
selection=json.loads((root/'round5_holdout/selection.json').read_text())
assert held['_summary']['complete']
n=held['_summary']['articles']
indices=np.random.default_rng(20260930).integers(0,n,size=(10000,n))


def compare(a,b):
    differences=np.asarray(a)-np.asarray(b)
    article=differences.mean(0);delta=float(article.mean())
    interval=np.quantile(article[indices].mean(1),[.025,.975])
    return {'delta_ce':delta,'relative_ppl_percent':float(100*np.expm1(delta)),
            'ci95':interval.tolist(),'seed_delta_ce':differences.mean(1).tolist(),
            'wins':int((differences.mean(1)<0).sum())}


s={'articles':n,'scored_tokens':n*255,'contexts':{},'sweep_seconds':0.,
   'evaluation_seconds':held['_summary']['evaluation_seconds'],'floating_point_ppl':held['floating_point']['ppl'],
   'warning':'Exploratory paired article intervals conditional on 3 calibration seeds; no multiplicity correction.'}
rows=[];resources=[]
for prefix,chosen in selection['selected'].items():
    runs=[json.loads((root/f'round5_{prefix}_s{seed}/metrics.json').read_text()) for seed in range(3)]
    s['sweep_seconds']+=sum(r['_summary']['total_seconds'] for r in runs)
    resources.extend(json.loads((root/f'round5_{prefix}_s{seed}/resources.json').read_text()) for seed in range(3))
    for seed,r in enumerate(runs):
        bm=json.loads((root/f'round5_block_{prefix}_s{seed}/metrics.json').read_text())
        s['sweep_seconds']+=bm['_summary']['total_seconds']
        r.update({k:v for k,v in bm.items() if k.startswith('block_')})
        resources.append(json.loads((root/f'round5_block_{prefix}_s{seed}/resources.json').read_text()))
    arrays={f:np.array([held[f'{prefix}/s{seed}/{f}']['sequence_ce'] for seed in range(3)]) for f in chosen}
    context={'control':chosen['control'],'methods':{},
             'prefix_ppl':st.mean(held[f'{prefix}/s{seed}/prefix']['ppl'] for seed in range(3))}
    for family,key in chosen.items():
        values=[held[f'{prefix}/s{seed}/{family}'] for seed in range(3)]
        item={'selected':key,'mean_ppl':st.mean(v['ppl'] for v in values),'seed_ppl':[v['ppl'] for v in values],
              'validation_ppl':st.mean(r[key]['ppl'] for r in runs),
              'optimization_seconds':st.mean(r[key]['optimization_seconds'] for r in runs),
              'changed_codes':st.mean(r[key]['changed_codes_vs_gptq'] for r in runs),
              'vs_control':compare(arrays[family],arrays['control']),
              'vs_branch':compare(arrays[family],arrays['branch']),
              'vs_gptq':compare(arrays[family],arrays['gptq_reference']),
              'validation_reconstruction_mse':st.mean(r[key]['validation_block_mse' if family=='block' else 'validation_mlp_mse'] for r in runs)}
        context['methods'][family]=item
        rows.append({'prefix':prefix,'family':family,'selected':key,'mean_ppl':item['mean_ppl'],
                     'optimization_seconds':item['optimization_seconds'],**item['vs_control']})
    # Development-only check: which optimizer configuration is selected by different proxy objectives?
    keys=[k for k in runs[0] if k.startswith(('branch_','product_','mlp_'))]
    context['development_minimizers']={m:min(keys,key=lambda k:st.mean(r[k][m] for r in runs))
        for m in ('calibration_branch_mse','calibration_product_mse','calibration_mlp_mse','validation_mlp_mse','ce')}
    context['proxy_failure_counts']={}
    for family in ('branch','product','mlp','block'):
        examined=decrease=contradict=0
        for r in runs:
            for key,v in r.items():
                if not key.startswith(family+'_'):continue
                metric=f'calibration_{family}_mse'
                baseline=v['initial_calibration_block_mse'] if family=='block' else r['gptq_reference'][metric]
                examined+=1
                if v[metric]<baseline:
                    decrease+=1
                    if v['ce']>r['gptq_reference']['ce']:contradict+=1
        context['proxy_failure_counts'][family]={'checkpoints':examined,'lower_calibration_loss':decrease,'lower_loss_but_higher_development_ce':contradict}
    s['contexts'][prefix]=context
resources.append(json.loads((root/'round5_holdout/resources.json').read_text()))
s['resources']={'cap_gib':4,'max_allocated_gib':max(r['peak_allocated_gib'] for r in resources),
                'max_reserved_gib':max(r['peak_reserved_gib'] for r in resources),
                'max_sampled_total_mib':max(t['used_mib'] for r in resources for t in r['samples']),
                'max_sampled_temperature_c':max(t['temperature_c'] for r in resources for t in r['samples']),
                'minimum_sampled_free_ram_gib':min(t['ram_available_gib'] for r in resources for t in r['samples'])}
s['continuation_gate']={}
for family in ('product','mlp','block'):
    checks={}
    for prefix in ('rtn','gptq'):
        d=s['contexts'][prefix]['methods'][family]['vs_control'];b=s['contexts'][prefix]['methods'][family]['vs_branch']
        checks[prefix]={'wins_at_least_2':d['wins']>=2,'improves_at_least_0_1_percent':d['relative_ppl_percent']<=-.1,
                        'upper_ci_negative':d['ci95'][1]<0,'beats_branch':b['delta_ce']<0}
    s['continuation_gate'][family]={'checks':checks,'passed':all(all(v.values()) for v in checks.values())}
(root/'round5_summary.json').write_text(json.dumps(s,indent=2,allow_nan=False),encoding='utf-8')
with (root/'round5_summary.csv').open('w',newline='',encoding='utf-8-sig') as f:
    writer=csv.DictWriter(f,fieldnames=rows[0]);writer.writeheader();writer.writerows(rows)
fig,axes=plt.subplots(1,2,figsize=(10,4.5),layout='constrained')
for offset,(family,color) in enumerate([('branch','#7f7f7f'),('product','#157a8c'),('mlp','#cf6b24'),('block','#784ab2')]):
    x=np.arange(2)+(offset-1.5)*.12
    vals=[s['contexts'][p]['methods'][family]['vs_control'] for p in ('rtn','gptq')]
    means=np.array([v['delta_ce'] for v in vals])*1000;ci=np.array([v['ci95'] for v in vals])*1000
    axes[0].errorbar(x,means,yerr=[means-ci[:,0],ci[:,1]-means],fmt='o',label=family,color=color,capsize=4)
axes[0].axhline(0,color='gray',lw=1);axes[0].set_xticks([0,1],['RTN prefix','GPTQ prefix'])
axes[0].set_ylabel('Delta CE vs development-selected control (x 0.001)')
axes[0].set_title(f'Unseen {n}-article holdout; lower is better');axes[0].legend()
for i,p in enumerate(('rtn','gptq')):
    families=['gptq_reference','rescomp','branch','product','mlp','block']
    vals=[s['contexts'][p]['methods'][f]['mean_ppl'] for f in families]
    axes[1].plot(families,vals,'o-',label=p)
axes[1].set_ylabel('Holdout PPL');axes[1].tick_params(axis='x',rotation=25)
axes[1].set_title('Independent single up-projection experiments');axes[1].legend()
folder=root.parent/'reports/figures';fig.savefig(folder/'round5_effects.png',dpi=180);fig.savefig(folder/'round5_effects.svg')
plt.close(fig)
print(json.dumps(s,indent=2))
