"""Audit reused/new candidates and paired results on two fresh confirmation sets."""
import csv,hashlib,json,statistics as st
from pathlib import Path
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import sys
sys.path.insert(0,str(Path(__file__).resolve().parent.parent))
from evaluate_round7 import choose

root=Path(__file__).resolve().parent;torch.set_num_threads(4)
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
held=read(root/'round7_holdout/metrics.json');frozen=read(root/'round7_holdout/selection.json')
manifest=read(root.parent/'data/round7_holdout/manifest.json')
assert held['_summary']['complete'] and len(held)==75
for field in ('config_sha256','checkpoint_sha256'):
    for path,digest in frozen[field].items():assert sha(root.parent/path)==digest
for path,digest in frozen['source_sha256'].items():assert sha(root.parent/path)==digest
assert sha(root.parent/'data/round7_holdout/manifest.json')==frozen['holdout_manifest_sha256']
resume=read(root/'round7_holdout/resume_manifest.json')
assert sha(root.parent/'resume_round7.py')==resume['resume_source_sha256']
assert sha(root/'round7_holdout/selection.json')==resume['original_selection_sha256']
interrupted=root/'round7_interrupted_holdout_write/metrics.json'
assert sha(interrupted)==resume['interrupted_metrics_sha256']
preserved=read(interrupted)
assert len(preserved)==71 and all(held[k]==v for k,v in preserved.items())
draws=np.random.default_rng(20261005).integers(0,32,size=(10000,32))
def compare(a,b):
    d=np.array(a)-np.array(b);groups=d.mean(0).reshape(32,4).mean(1);delta=float(groups.mean())
    return {'delta_ce':delta,'relative_ppl_percent':float(100*np.expm1(delta)),
            'ci95':np.quantile(groups[draws].mean(1),[.025,.975]).tolist(),
            'seed_delta_ce':d.mean(1).tolist(),'wins':int((d.mean(1)<0).sum()),'improved_blocks':int((groups<0).sum())}
s={'datasets':{},'development':{},'scan_seconds':0.,'evaluation':held['_summary'],
   'source':{d:x['source'] for d,x in manifest['datasets'].items()},'new_candidates':18,'reused_candidates':18,
   'resume':resume,
   'warning':'Exploratory 32-contiguous-block bootstrap per corpus, conditional on three seeds. Old PTB test is now development.'}
resources=[];devrows=[];new_count=reused_count=grid_count=0
for layer,selected in frozen['selected'].items():
    runs=[]
    for seed in range(3):
        folder=root/f'round7_l{layer}_s{seed}';m=read(folder/'metrics.json');c=read(folder/'config.json')
        assert m['_summary']['complete'] and m['_summary']['candidate_count']==6
        resources.append(read(folder/'resources.json'));s['scan_seconds']+=m['_summary']['total_seconds']
        for name,digest in c['source_sha256'].items():assert sha(root.parent/name)==digest
        for name,digest in c['parent_checkpoints_sha256'].items():assert sha(root.parent/name)==digest
        scale=torch.load(folder/'grid.pt',weights_only=True)['scale']
        initial=torch.load(folder/'tau0.pt',weights_only=True)['weight']
        full=torch.load(folder/'tau1.pt',weights_only=True)['weight']
        initial_codes=(initial/scale).round();full_codes=(full/scale).round()
        for key,v in m.items():
            if key=='_summary':continue
            payload=torch.load(folder/(key+'.pt'),weights_only=True);weight=payload['weight'];codes=(weight/scale).round()
            assert torch.isfinite(weight).all() and codes.min()>=-8 and codes.max()<=7
            torch.testing.assert_close(weight,codes*scale,rtol=0,atol=0);grid_count+=1
            if payload['bias'] is not None:assert payload['bias'].numel()==896 and torch.isfinite(payload['bias']).all()
            tau=float(key.split('_')[0][3:])
            expected=((1-tau)*initial_codes+tau*full_codes).round().clamp(-8,7)*scale
            torch.testing.assert_close(weight,expected,rtol=0,atol=0)
            if v['reused_from_round6']:
                parent_key={'tau0':'gptq','tau0_bias':'bias','tau1':'block_s192'}[key]
                parent=torch.load(root/f'round6_l{layer}_s{seed}/{parent_key}.pt',weights_only=True)
                torch.testing.assert_close(weight,parent['weight'],rtol=0,atol=0)
                if payload['bias'] is None:assert parent['bias'] is None
                else:torch.testing.assert_close(payload['bias'],parent['bias'],rtol=0,atol=0)
            reused_count+=v['reused_from_round6'];new_count+=not v['reused_from_round6']
            for domain,n in [('wiki',64),('ptb_seen',128)]:
                assert len(v[domain]['sequence_ce'])==n
                assert abs(st.mean(v[domain]['sequence_ce'])-v[domain]['ce'])<1e-12
            devrows.append({'layer':layer,'seed':seed,'candidate':key,'reused':v['reused_from_round6'],
                            'wiki_ppl':v['wiki']['ppl'],'seen_ptb_ppl':v['ptb_seen']['ppl'],'fit_seconds':v['fit_seconds'],
                            'changed_code_percent':100*float((codes!=initial_codes).float().mean()),
                            'mean_absolute_code_move':float((codes-initial_codes).abs().mean()),
                            'bias_rms':0. if payload['bias'] is None else float(payload['bias'].square().mean().sqrt())})
        runs.append(m)
    observed,gaps,worst=choose(runs);assert observed==selected and gaps==frozen['development_gaps'][layer]
    s['development'][layer]={'selected':selected,'gaps':gaps,'worst_domain_gap':worst,
                            'mean_ppl':{k:{d:st.mean(r[k][d]['ppl'] for r in runs) for d in ('wiki','ptb_seen')} for k in gaps}}
assert (new_count,reused_count,grid_count)==(18,18,36)
s['parameter_changes']={layer:{key:{field:st.mean(r[field] for r in devrows if r['layer']==layer and r['candidate']==key)
    for field in ('changed_code_percent','mean_absolute_code_move','bias_rms')}
    for key in s['development'][layer]['gaps']} for layer in s['development']}
rows=[]
for dataset,description in manifest['datasets'].items():
    assert sha(root.parent/f'data/round7_holdout/{dataset}.pt')==description['tokens_sha256']
    for i in range(32):assert len({w['source_block'] for w in description['windows'][i*4:i*4+4]})==1
    context={'floating_point_ppl':held[dataset+'/floating_point']['ppl'],'layers':{}}
    for layer,chosen in frozen['selected'].items():
        arrays={f:[held[f'{dataset}/l{layer}/s{seed}/{f}']['sequence_ce'] for seed in range(3)] for f in chosen}
        methods={}
        for family,key in chosen.items():
            values=[held[f'{dataset}/l{layer}/s{seed}/{family}'] for seed in range(3)]
            for v in values:assert len(v['sequence_ce'])==128 and abs(st.mean(v['sequence_ce'])-v['ce'])<1e-12
            item={'selected':key,'mean_ppl':st.mean(v['ppl'] for v in values),'seed_ppl':[v['ppl'] for v in values],
                  **{f'vs_{baseline}':compare(arrays[family],arrays[baseline]) for baseline in ('gptq','bias_only','block192','wiki_only_hybrid')}}
            methods[family]=item;rows.append({'dataset':dataset,'layer':layer,'family':family,'selected':key,'mean_ppl':item['mean_ppl'],**item['vs_gptq']})
        context['layers'][layer]=methods
    s['datasets'][dataset]=context
resources.append(read(root/'round7_holdout/resources.json'))
s['resources']={'cap_gib':4.,'pause_seconds':.15,'start_temperature_max_c':65,
    'max_allocated_gib':max(r['peak_allocated_gib'] for r in resources),'max_reserved_gib':max(r['peak_reserved_gib'] for r in resources),
    'max_sampled_total_mib':max(t['used_mib'] for r in resources for t in r['samples']),
    'max_sampled_temperature_c':max(t['temperature_c'] for r in resources for t in r['samples']),
    'minimum_sampled_free_ram_gib':min(t['ram_available_gib'] for r in resources for t in r['samples'])}
s['verification']={'grid_candidates':36,'new':new_count,'reused':reused_count,'all_frozen_hashes_match':True,'selection_recomputed':True}
s['gates']={}
for family in ('weight_only','hybrid','bias_only','block192'):
    checks={}
    for dataset,c in s['datasets'].items():
        for layer,methods in c['layers'].items():
            d=methods[family]['vs_gptq'];b=methods[family]['vs_bias_only']
            checks[dataset+'/'+layer]={'improves_vs_gptq':d['delta_ce']<0,'wins_at_least_2':d['wins']>=2,
                'upper_ci_negative':d['ci95'][1]<0,'extra_over_bias_0_1_percent':b['relative_ppl_percent']<=-.1,
                'extra_over_bias_upper_ci_negative':b['ci95'][1]<0,'extra_over_bias_wins_at_least_2':b['wins']>=2}
    s['gates'][family]={'checks':checks,'all_four_improve_with_2_seeds':all(x['improves_vs_gptq'] and x['wins_at_least_2'] for x in checks.values()),
        'all_four_ci_support':all(x['upper_ci_negative'] for x in checks.values()),
        'extra_over_bias_passed':all(x['extra_over_bias_0_1_percent'] and x['extra_over_bias_upper_ci_negative'] and x['extra_over_bias_wins_at_least_2'] for x in checks.values())}
(root/'round7_summary.json').write_text(json.dumps(s,indent=2,allow_nan=False),encoding='utf-8')
for filename,records in [('round7_summary.csv',rows),('round7_all_development.csv',devrows)]:
    with (root/filename).open('w',newline='',encoding='utf-8-sig') as f:
        w=csv.DictWriter(f,fieldnames=records[0]);w.writeheader();w.writerows(records)
fig,axes=plt.subplots(1,2,figsize=(11,4.6),layout='constrained')
for ax,(dataset,c) in zip(axes,s['datasets'].items()):
    for j,(family,color) in enumerate([('block192','#477fa6'),('bias_only','#b38b23'),('weight_only','#2b9474'),('hybrid','#9a5fa8')]):
        for i,layer in enumerate(('12','18')):
            d=c['layers'][layer][family]['vs_gptq'];mean=d['delta_ce'];lo,hi=d['ci95']
            ax.errorbar(i+(j-1.5)*.15,mean,yerr=[[mean-lo],[hi-mean]],fmt='o',color=color,capsize=4,label=family if i==0 else None)
    ax.axhline(0,color='gray',lw=1);ax.set_xticks([0,1],['Layer 12','Layer 18']);ax.set_ylabel('Delta CE vs initial GPTQ')
    ax.set_title(dataset+'; lower is better');ax.legend()
fig.savefig(root.parent/'reports/figures/round7_effects.png',dpi=180);fig.savefig(root.parent/'reports/figures/round7_effects.svg');plt.close(fig)
print(json.dumps(s,indent=2))
