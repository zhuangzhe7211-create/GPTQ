"""Freeze all development choices, then score the new external corpus once."""
import json,statistics,time
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM,AutoTokenizer
import gpu_safety as safety
from evaluate_refinement import sha
from prepare_round3_holdout import MODEL,REVISION
from run_pilot import read_texts,sequences,tensor_hash
from run_round5 import apply_prefix
from run_round6 import FAMILIES


def main():
    out=Path('results/round6_holdout')
    if out.exists():raise FileExistsError(out)
    frozen={'selected':{},'development_ce':{},'checkpoint_sha256':{},'config_sha256':{},
            'source_sha256':{f:sha(f) for f in ['evaluate_round6.py','run_round6.py','gpu_safety.py']},
            'rule':'Mean development CE across three seeds; never use external scores to select steps.'}
    for layer in (12,18):
        runs=[]
        for seed in range(3):
            folder=Path(f'results/round6_l{layer}_s{seed}')
            m=json.loads((folder/'metrics.json').read_text());c=json.loads((folder/'config.json').read_text())
            assert m['_summary']['complete'] and m['_summary']['candidate_count']==10
            assert c['layer']==layer and c['seed']==seed and c['prefix_blocks']==12
            for name,digest in c['source_sha256'].items():assert sha(name)==digest
            frozen['config_sha256'][str(folder/'config.json')]=sha(folder/'config.json');runs.append(m)
        scores={k:statistics.mean(m[k]['ce'] for m in runs) for k in runs[0] if k not in ('_summary','common_prefix')}
        selected={family:min((k for k in scores if k.startswith(family+'_')),key=scores.get) for family in FAMILIES}
        selected.update({'gptq':'gptq','bias':'bias','block192':'block_s192','block576':'block_s576'})
        frozen['selected'][str(layer)]=selected;frozen['development_ce'][str(layer)]=scores
        for seed in range(3):
            for key in set(selected.values()):
                path=Path(f'results/round6_l{layer}_s{seed}/{key}.pt')
                frozen['checkpoint_sha256'][str(path)]=sha(path)
    hold=Path('data/round6_holdout');frozen['holdout_manifest_sha256']=sha(hold/'manifest.json')
    out.mkdir();(out/'selection.json').write_text(json.dumps(frozen,indent=2),encoding='utf-8')
    print('ALL CHOICES FROZEN',frozen['selected'],flush=True)
    manifest=json.loads((hold/'manifest.json').read_text());assert sha(hold/'tokens.pt')==manifest['tokens_sha256']
    batches=torch.load(hold/'tokens.pt',weights_only=True)
    hashes=list(map(tensor_hash,batches));assert hashes==[w['window_sha256'] for w in manifest['windows']]
    assert len(batches)==128 and len({w['source_block'] for w in manifest['windows']})==32
    for path in frozen['config_sha256']:
        c=json.loads(Path(path).read_text());assert not set(hashes)&set(c['calibration_window_hashes']+c['validation_window_hashes'])
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    safety.initialize();torch.set_num_threads(4);torch.manual_seed(0)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    tok=AutoTokenizer.from_pretrained(MODEL,revision=REVISION,local_files_only=True)
    val=sequences(read_texts('data/wikitext2_20260926_s0/validation.jsonl'),tok,64,256)
    model=AutoModelForCausalLM.from_pretrained(MODEL,revision=REVISION,local_files_only=True,
        dtype=torch.float32,attn_implementation='eager').cuda().eval().requires_grad_(False)
    model.config.use_cache=False;results={};start=time.perf_counter();unique=0
    def record(name):
        nonlocal unique
        results[name]=safety.evaluate(model,batches,'cuda');unique+=1
        (out/'metrics.json').write_text(json.dumps(results,indent=2),encoding='utf-8')
        (out/'resources.json').write_text(json.dumps(safety.summary(),indent=2),encoding='utf-8')
        print(name,results[name]['ppl'],flush=True)
    record('floating_point')
    for layer in (12,18):
        block=model.model.layers[layer];target=block.mlp.up_proj;original=target.weight.detach().clone()
        for seed in range(3):
            folder=Path(f'results/round6_l{layer}_s{seed}')
            config=json.loads((folder/'config.json').read_text())
            assert apply_prefix(model,'gptq',seed)==config['prefix_sha256']
            assert list(map(tensor_hash,val))==config['validation_window_hashes']
            prior=json.loads((folder/'metrics.json').read_text())
            assert abs(safety.evaluate(model,val,'cuda')['ce']-prior['common_prefix']['ce'])<1e-6
            record(f'l{layer}/s{seed}/prefix');computed={}
            for family,key in frozen['selected'][str(layer)].items():
                name=f'l{layer}/s{seed}/{family}'
                if key in computed:results[name]=results[computed[key]].copy();continue
                path=folder/(key+'.pt');assert sha(path)==frozen['checkpoint_sha256'][str(path)]
                payload=torch.load(path,weights_only=True);assert payload['target']==config['target']
                with torch.no_grad():target.weight.copy_(payload['weight'])
                bias=payload['bias'];hook=None
                if bias is not None:
                    bias=bias.cuda();hook=block.register_forward_hook(lambda module,args,output:output+bias.view(1,1,-1))
                try:record(name)
                finally:
                    if hook is not None:hook.remove()
                computed[key]=name
            with torch.no_grad():target.weight.copy_(original)
            safety.pause()
    results['_summary']={'complete':True,'unique_forward_evaluations':unique,'evaluation_seconds':time.perf_counter()-start,
        'prefix_checks':6,'all_choices_frozen_before_evaluation':True,'windows':128,'bootstrap_groups':32}
    (out/'metrics.json').write_text(json.dumps(results,indent=2),encoding='utf-8')
    (out/'resources.json').write_text(json.dumps(safety.summary(),indent=2),encoding='utf-8')
    print('DONE',out,flush=True)


if __name__=='__main__':main()
