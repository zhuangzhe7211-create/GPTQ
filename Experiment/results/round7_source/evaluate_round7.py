"""Freeze cross-domain development choices, then evaluate two fresh sets."""
import json,statistics,time
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM,AutoTokenizer
from evaluate_refinement import sha
from prepare_round3_holdout import MODEL,REVISION
from run_pilot import read_texts,sequences,tensor_hash
from run_round5 import apply_prefix
import gpu_safety as safety
from round7_safety import initialize


def choose(runs):
    keys=[k for k in runs[0] if k!='_summary']
    gaps={k:{d:statistics.mean(r[k][d]['ce']-r['tau0'][d]['ce'] for r in runs) for d in ('wiki','ptb_seen')} for k in keys}
    worst={k:max(gaps[k].values()) for k in keys}
    selected={'weight_only':min(('tau0','tau0.5','tau1'),key=worst.get),
              'hybrid':min(('tau0','tau0_bias','tau0.5_bias','tau1_bias'),key=worst.get),
              'gptq':'tau0','bias_only':'tau0_bias','block192':'tau1'}
    assert worst[selected['weight_only']]<=0 and worst[selected['hybrid']]<=0
    return selected,gaps,worst


def main():
    out=Path('results/round7_holdout')
    if out.exists():raise FileExistsError(out)
    frozen={'selected':{},'development_gaps':{},'worst_domain_gap':{},'config_sha256':{},'checkpoint_sha256':{},
            'source_sha256':{f:sha(f) for f in ['evaluate_round7.py','run_round7.py','round7_safety.py','gpu_safety.py']},
            'rule':'Minimize worst development-domain mean CE gap to GPTQ; same choice for all three seeds.'}
    for layer in (12,18):
        runs=[]
        for seed in range(3):
            folder=Path(f'results/round7_l{layer}_s{seed}')
            m=json.loads((folder/'metrics.json').read_text());c=json.loads((folder/'config.json').read_text())
            assert m['_summary']['complete'] and m['_summary']['candidate_count']==6
            for file,digest in c['source_sha256'].items():assert sha(file)==digest
            for file,digest in c['parent_checkpoints_sha256'].items():assert sha(file)==digest
            frozen['config_sha256'][str(folder/'config.json')]=sha(folder/'config.json');runs.append(m)
        selected,gaps,worst=choose(runs)
        frozen['selected'][str(layer)]=selected;frozen['development_gaps'][str(layer)]=gaps;frozen['worst_domain_gap'][str(layer)]=worst
        for seed in range(3):
            for key in set(selected.values()):
                path=Path(f'results/round7_l{layer}_s{seed}/{key}.pt');frozen['checkpoint_sha256'][str(path)]=sha(path)
    data=Path('data/round7_holdout');frozen['holdout_manifest_sha256']=sha(data/'manifest.json')
    out.mkdir();(out/'selection.json').write_text(json.dumps(frozen,indent=2),encoding='utf-8')
    print('ALL CHOICES FROZEN',frozen['selected'],flush=True)
    manifest=json.loads((data/'manifest.json').read_text());datasets={};allhashes=set()
    for name,description in manifest['datasets'].items():
        path=data/(name+'.pt');assert sha(path)==description['tokens_sha256']
        batches=torch.load(path,weights_only=True);hashes=list(map(tensor_hash,batches))
        assert hashes==[w['window_sha256'] for w in description['windows']]
        assert len(batches)==len(set(hashes))==128 and not set(hashes)&allhashes
        for config_path in frozen['config_sha256']:
            c=json.loads(Path(config_path).read_text())
            old=set(c['calibration_window_hashes'])
            for windows in c['development_window_hashes'].values():old.update(windows)
            assert not old&set(hashes)
        allhashes.update(hashes);datasets[name]=batches
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    initialize();torch.set_num_threads(4);torch.manual_seed(0)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    tok=AutoTokenizer.from_pretrained(MODEL,revision=REVISION,local_files_only=True)
    wiki=sequences(read_texts('data/wikitext2_20260926_s0/validation.jsonl'),tok,64,256)
    model=AutoModelForCausalLM.from_pretrained(MODEL,revision=REVISION,local_files_only=True,
        dtype=torch.float32,attn_implementation='eager').cuda().eval().requires_grad_(False)
    model.config.use_cache=False;metrics={};start=time.perf_counter();calls=0
    def record(suffix):
        nonlocal calls
        for name,batches in datasets.items():
            key=name+'/'+suffix;metrics[key]=safety.evaluate(model,batches,'cuda');calls+=1
            print(key,metrics[key]['ppl'],flush=True)
            (out/'metrics.json').write_text(json.dumps(metrics,indent=2),encoding='utf-8')
            (out/'resources.json').write_text(json.dumps(safety.summary(),indent=2),encoding='utf-8')
    record('floating_point')
    for layer in (12,18):
        block=model.model.layers[layer];target=block.mlp.up_proj;original=target.weight.detach().clone()
        for seed in range(3):
            folder=Path(f'results/round7_l{layer}_s{seed}');config=json.loads((folder/'config.json').read_text())
            assert apply_prefix(model,'gptq',seed)==config['prefix_sha256']
            prior=json.loads(Path(f'results/round6_l{layer}_s{seed}/metrics.json').read_text())
            assert abs(safety.evaluate(model,wiki,'cuda')['ce']-prior['common_prefix']['ce'])<1e-6
            computed={}
            for family,key in frozen['selected'][str(layer)].items():
                suffix=f'l{layer}/s{seed}/{family}'
                if key in computed:
                    for dataset in datasets:metrics[dataset+'/'+suffix]=metrics[dataset+'/'+computed[key]].copy()
                    continue
                path=folder/(key+'.pt');assert sha(path)==frozen['checkpoint_sha256'][str(path)]
                payload=torch.load(path,weights_only=True);assert payload['target']==config['target']
                with torch.no_grad():target.weight.copy_(payload['weight'])
                bias=payload['bias'];hook=None
                if bias is not None:
                    bias=bias.cuda();hook=block.register_forward_hook(lambda m,args,z:z+bias.view(1,1,-1))
                try:record(suffix)
                finally:
                    if hook is not None:hook.remove()
                computed[key]=suffix
            with torch.no_grad():target.weight.copy_(original)
            safety.pause()
    metrics['_summary']={'complete':True,'evaluation_seconds':time.perf_counter()-start,'actual_dataset_evaluations':calls,
        'unique_candidate_contexts':sum(len(set(v.values()))*3 for v in frozen['selected'].values()),
        'all_choices_frozen_before_evaluation':True,'prefix_checks':6,'datasets':list(datasets)}
    (out/'metrics.json').write_text(json.dumps(metrics,indent=2),encoding='utf-8')
    (out/'resources.json').write_text(json.dumps(safety.summary(),indent=2),encoding='utf-8')
    print('DONE',out,flush=True)


if __name__=='__main__':main()
