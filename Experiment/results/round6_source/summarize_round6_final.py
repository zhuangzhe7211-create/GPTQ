"""Analyze the frozen external test with paired contiguous-block uncertainty."""
import csv,json,statistics,hashlib
from pathlib import Path
import numpy as np
import torch
from safetensors import safe_open
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

root=Path(__file__).resolve().parent
torch.set_num_threads(4)
def read(path):return json.loads(path.read_text(encoding='utf-8'))
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
held=read(root/'round6_holdout/metrics.json');frozen=read(root/'round6_holdout/selection.json')
manifest=read(root.parent/'data/round6_holdout/manifest.json')
assert held['_summary']['complete'] and len(held)==62
assert sha(root.parent/'data/round6_holdout/manifest.json')==frozen['holdout_manifest_sha256']
for field in ('checkpoint_sha256','config_sha256'):
    for path,digest in frozen[field].items():assert sha(root.parent/path)==digest
for path,digest in frozen['source_sha256'].items():assert sha(root.parent/path)==digest
for i in range(32):assert len({x['source_block'] for x in manifest['windows'][i*4:i*4+4]})==1
bootstrap=np.random.default_rng(20261002).integers(0,32,size=(10000,32))
def compare(a,b):
    difference=np.asarray(a)-np.asarray(b)
    groups=difference.mean(0).reshape(32,4).mean(1)
    delta=float(groups.mean());interval=np.quantile(groups[bootstrap].mean(1),[.025,.975]).tolist()
    return {'delta_ce':delta,'relative_ppl_percent':float(100*np.expm1(delta)),
            'ci95':interval,'seed_delta_ce':difference.mean(1).tolist(),'wins':int((difference.mean(1)<0).sum()),
            'improved_blocks':int((groups<0).sum())}

s={'floating_point_ppl':held['floating_point']['ppl'],'layers':{},'scan_seconds':0.,
   'evaluation_seconds':held['_summary']['evaluation_seconds'],'unique_external_evaluations':held['_summary']['unique_forward_evaluations'],
   'source':manifest['source'],'windows':128,'scored_tokens':32640,'bootstrap_groups':32,
   'warning':'Exploratory paired contiguous-block intervals, conditional on three calibration seeds; no multiplicity correction.'}
resources=[];rows=[];scanrows=[];grid_count=continuous_count=0
for layer,chosen in frozen['selected'].items():
    runs=[]
    for seed in range(3):
        folder=root/f'round6_l{layer}_s{seed}';m=read(folder/'metrics.json');c=read(folder/'config.json')
        assert m['_summary']['complete'] and m['_summary']['candidate_count']==10
        s['scan_seconds']+=m['_summary']['total_seconds'];resources.append(read(folder/'resources.json'))
        extra=root/f'round6_bounded_l{layer}_s{seed}';bm=read(extra/'metrics.json');bc=read(extra/'config.json')
        assert bm['_summary']['complete'] and bm['_summary']['candidate_count']==2
        assert bc['token_indices_sha256']==c['token_indices_sha256'] and bc['initial_weight_sha256']==c['initial_weight_sha256']
        for name,digest in bc['source_sha256'].items():assert sha(root.parent/name)==digest
        s['scan_seconds']+=bm['_summary']['total_seconds'];resources.append(read(extra/'resources.json'))
        m.update({k:v for k,v in bm.items() if k.startswith('bounded_')})
        scale=torch.load(folder/'grid.pt',weights_only=True)['scale']
        model_file=root.parent/'.hf_cache/hub/models--Qwen--Qwen2.5-0.5B/snapshots/060db6499f32faf8b98477b0a26969ef7d8b9987/model.safetensors'
        with safe_open(model_file,framework='pt',device='cpu') as sf:
            original=sf.get_tensor(c['target']+'.weight').float()
        torch.testing.assert_close(scale,original.abs().amax(1,keepdim=True).clamp_min(1e-8)/7,rtol=1.2e-7,atol=0)
        initial=torch.load(folder/'gptq.pt',weights_only=True)['weight']
        for name,digest in c['source_sha256'].items():assert sha(root.parent/name)==digest
        for key,v in m.items():
            if key in ('common_prefix','_summary'):continue
            path=(extra if key.startswith('bounded_') else folder)/(key+'.pt')
            payload=torch.load(path,weights_only=True);weight=payload['weight']
            assert payload['target']==c['target'] and torch.isfinite(weight).all()
            if payload['quantized']:
                codes=(weight/scale).round();assert codes.min()>=-8 and codes.max()<=7
                torch.testing.assert_close(weight,codes*scale,rtol=0,atol=0);grid_count+=1
                if key=='bias':torch.testing.assert_close(weight,initial,rtol=0,atol=0)
            else:continuous_count+=1
            assert len(v['sequence_ce'])==64 and abs(statistics.mean(v['sequence_ce'])-v['ce'])<1e-12
            scanrows.append({'layer':layer,'seed':seed,'candidate':key,**{k:v[k] for k in ['ppl','ce','calibration_block_mse','validation_block_mse','optimization_seconds','quantized','extra_bias_parameters','changed_codes_vs_gptq']}})
        runs.append(m)
    scores={k:statistics.mean(m[k]['ce'] for m in runs) for k in frozen['development_ce'][layer]}
    assert scores==frozen['development_ce'][layer]
    for family in ('block','token','direction','continuous','bounded'):
        assert chosen[family]==min((k for k in scores if k.startswith(family+'_')),key=scores.get)
    arrays={family:np.asarray([held[f'l{layer}/s{seed}/{family}']['sequence_ce'] for seed in range(3)]) for family in chosen}
    context={'methods':{},'prefix_ppl':statistics.mean(held[f'l{layer}/s{seed}/prefix']['ppl'] for seed in range(3))}
    for family,key in chosen.items():
        values=[held[f'l{layer}/s{seed}/{family}'] for seed in range(3)]
        for v in values:
            assert len(v['sequence_ce'])==128 and abs(statistics.mean(v['sequence_ce'])-v['ce'])<1e-12
        item={'selected':key,'mean_ppl':statistics.mean(v['ppl'] for v in values),'seed_ppl':[v['ppl'] for v in values],
              'development_ppl':statistics.mean(r[key]['ppl'] for r in runs),
              'compute_seconds':statistics.mean(r[key]['optimization_seconds'] for r in runs),
              'calibration_mse':statistics.mean(r[key]['calibration_block_mse'] for r in runs),
              'validation_mse':statistics.mean(r[key]['validation_block_mse'] for r in runs),
              'vs_block':compare(arrays[family],arrays['block']),
              'vs_gptq':compare(arrays[family],arrays['gptq']),
              'vs_block192':compare(arrays[family],arrays['block192']),
              'vs_block576':compare(arrays[family],arrays['block576'])}
        context['methods'][family]=item
        rows.append({'layer':layer,'family':family,'selected':key,'mean_ppl':item['mean_ppl'],**item['vs_block']})
    context['development_by_steps']={family:{str(step):statistics.mean(r[f'{family}_s{step}']['ppl'] for r in runs)
        for step in (192,576)} for family in ('block','token','direction','continuous','bounded')}
    s['layers'][layer]=context
resources.append(read(root/'round6_holdout/resources.json'))
s['resources']={'cap_gib':4.,'max_allocated_gib':max(r['peak_allocated_gib'] for r in resources),
    'max_reserved_gib':max(r['peak_reserved_gib'] for r in resources),
    'max_sampled_total_mib':max(t['used_mib'] for r in resources for t in r['samples']),
    'max_sampled_temperature_c':max(t['temperature_c'] for r in resources for t in r['samples']),
    'minimum_sampled_available_ram_gib':min(t['ram_available_gib'] for r in resources for t in r['samples'])}
s['verification']={'grid_checkpoints':grid_count,'continuous_controls':continuous_count,
    'frozen_hashes_checked':True,'development_selection_recomputed':True,'complete_main_scans':6,'complete_bounded_scans':6}
s['continuation_gate']={}
for family in ('token','direction','bounded'):
    checks={}
    for layer,c in s['layers'].items():
        d=c['methods'][family]['vs_block'];checks[layer]={'wins_at_least_2':d['wins']>=2,
            'improves_at_least_0_1_percent':d['relative_ppl_percent']<=-.1,'upper_ci_negative':d['ci95'][1]<0}
    s['continuation_gate'][family]={'checks':checks,'passed':all(all(c.values()) for c in checks.values())}
(root/'round6_summary.json').write_text(json.dumps(s,indent=2,allow_nan=False),encoding='utf-8')
for name,records in [('round6_summary.csv',rows),('round6_all_development.csv',scanrows)]:
    with (root/name).open('w',newline='',encoding='utf-8-sig') as f:
        w=csv.DictWriter(f,fieldnames=records[0]);w.writeheader();w.writerows(records)
fig,axes=plt.subplots(1,2,figsize=(11,4.6),layout='constrained')
for i,(family,color) in enumerate([('block','#3176ae'),('token','#299474'),('direction','#a65193'),('bounded','#cc6432'),('bias','#a57b22')]):
    for j,layer in enumerate(('12','18')):
        d=s['layers'][layer]['methods'][family]['vs_gptq'];lo,hi=d['ci95'];mean=d['delta_ce']
        axes[0].errorbar(j+(i-2)*.14,mean,yerr=[[mean-lo],[hi-mean]],fmt='o',color=color,capsize=3,label=family if j==0 else None)
axes[0].axhline(0,color='gray',lw=1);axes[0].set_xticks([0,1],['Layer 12','Layer 18'])
axes[0].set_title('External PTB: paired CE vs initial GPTQ');axes[0].set_ylabel('Delta CE (lower is better)');axes[0].legend()
for layer in ('12','18'):
    families=['gptq','bias','block192','block576','token','direction','bounded','continuous']
    axes[1].plot(families,[s['layers'][layer]['methods'][f]['mean_ppl'] for f in families],'o-',label='Layer '+layer)
axes[1].set_ylabel('Qwen-tokenized PTB PPL');axes[1].tick_params(axis='x',rotation=30)
axes[1].set_title('Continuous is a floating-weight control');axes[1].legend()
fig.savefig(root.parent/'reports/figures/round6_effects.png',dpi=180);fig.savefig(root.parent/'reports/figures/round6_effects.svg');plt.close(fig)
print(json.dumps(s,indent=2))
