"""Freeze all round-5 choices, then evaluate unseen articles under resource limits."""
import json
from pathlib import Path
import statistics as st
import time

import torch
from transformers import AutoModelForCausalLM,AutoTokenizer
import gpu_safety as safety
from evaluate_refinement import sha
from prepare_round3_holdout import MODEL,REVISION
from run_pilot import read_texts,sequences,tensor_hash
from run_round5 import TARGET,apply_prefix

FAMILIES=('rescomp','gptaq','weak1','branch','product','mlp')


def main():
    out=Path('results/round5_holdout')
    if out.exists():raise FileExistsError(out)
    frozen={'selected':{},'validation_ce':{},'checkpoint_sha256':{},'config_sha256':{},
            'source_sha256':sha(__file__),'safety_source_sha256':sha('gpu_safety.py'),
            'rule':'Mean development CE across three calibration seeds; all contexts frozen together.'}
    configs={}
    for prefix in ('rtn','gptq'):
        runs=[]
        for seed in range(3):
            folder=Path(f'results/round5_{prefix}_s{seed}')
            m=json.loads((folder/'metrics.json').read_text());c=json.loads((folder/'config.json').read_text())
            assert m['_summary']['complete'] and m['_summary']['candidate_count']==35
            assert c['prefix']==prefix and c['seed']==seed and c['target']==TARGET
            assert c['samples']==64 and c['bits']==4 and c['dtype']=='float32'
            for name,digest in c['source_sha256'].items():assert sha(name)==digest
            frozen['config_sha256'][str(folder/'config.json')]=sha(folder/'config.json')
            configs[(prefix,seed)]=c;runs.append(m)
        keys=[k for k in runs[0] if k not in ('_summary','common_prefix')]
        scores={k:st.mean(m[k]['ce'] for m in runs) for k in keys}
        selected={f:min((k for k in keys if k.startswith(f+'_')),key=scores.get) for f in FAMILIES}
        selected.update({'rtn_target':'rtn_target','gptq_reference':'gptq_reference'})
        selected['control']=min((k for k in keys if not k.startswith(('product_','mlp_'))),key=scores.get)
        frozen['selected'][prefix]=selected;frozen['validation_ce'][prefix]=scores
        for seed in range(3):
            for key in set(selected.values()):
                path=Path(f'results/round5_{prefix}_s{seed}/{key}.pt')
                frozen['checkpoint_sha256'][str(path)]=sha(path)
    hold=Path('data/round5_holdout');frozen['holdout_manifest_sha256']=sha(hold/'manifest.json')
    out.mkdir();(out/'selection.json').write_text(json.dumps(frozen,indent=2),encoding='utf-8')
    print('ALL CHOICES FROZEN',json.dumps(frozen['selected']),flush=True)
    manifest=json.loads((hold/'manifest.json').read_text())
    assert sha(hold/'tokens.pt')==manifest['tokens_sha256']
    batches=torch.load(hold/'tokens.pt',weights_only=True)
    hashes=list(map(tensor_hash,batches));articles={w['article_heading_row'] for w in manifest['windows']}
    assert hashes==[w['window_sha256'] for w in manifest['windows']]
    assert 64<=len(batches)==len(set(hashes))==len(articles)<=128
    assert not articles.intersection(manifest['excluded_calibration_article_ids']+manifest['excluded_round3_and_round4_article_ids'])
    for c in configs.values():assert not set(hashes).intersection(c['calibration_window_hashes']+c['validation_window_hashes'])
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    safety.initialize();torch.set_num_threads(4);torch.manual_seed(0)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    tok=AutoTokenizer.from_pretrained(MODEL,revision=REVISION,local_files_only=True)
    val=sequences(read_texts('data/wikitext2_20260926_s0/validation.jsonl'),tok,64,256)
    assert list(map(tensor_hash,val))==configs[('rtn',0)]['validation_window_hashes']
    model=AutoModelForCausalLM.from_pretrained(MODEL,revision=REVISION,local_files_only=True,
          dtype=torch.float32,attn_implementation='eager').cuda().eval().requires_grad_(False)
    model.config.use_cache=False;target=model.get_submodule(TARGET);original=target.weight.detach().clone()
    results={};start=time.perf_counter()
    def record(name):
        results[name]=safety.evaluate(model,batches,'cuda')
        (out/'metrics.json').write_text(json.dumps(results,indent=2,allow_nan=False),encoding='utf-8')
        (out/'resources.json').write_text(json.dumps(safety.summary(),indent=2),encoding='utf-8')
        print(name,results[name]['ppl'],flush=True)
    record('floating_point')
    for prefix in ('rtn','gptq'):
        for seed in range(3):
            # RTN is identical across seeds; never requantize an already quantized grid.
            if prefix=='gptq' or seed==0:
                observed_hash=apply_prefix(model,prefix,seed)
                assert observed_hash==configs[(prefix,seed)]['prefix_sha256']
            expected=json.loads(Path(f'results/round5_{prefix}_s{seed}/metrics.json').read_text())['common_prefix']['ce']
            observed=safety.evaluate(model,val,'cuda')['ce']
            assert abs(expected-observed)<1e-6,'Prefix reproduction mismatch'
            record(f'{prefix}/s{seed}/prefix');computed={}
            for family,key in frozen['selected'][prefix].items():
                name=f'{prefix}/s{seed}/{family}'
                if key in computed:
                    results[name]=results[computed[key]].copy()
                    continue
                path=Path(f'results/round5_{prefix}_s{seed}/{key}.pt')
                assert sha(path)==frozen['checkpoint_sha256'][str(path)]
                payload=torch.load(path,weights_only=True);assert payload['target']==TARGET
                with torch.no_grad():target.weight.copy_(payload['weight'])
                record(name);computed[key]=name
            with torch.no_grad():target.weight.copy_(original)
            torch.cuda.empty_cache();safety.pause()
    results['_summary']={'complete':True,'all_choices_frozen_before_evaluation':True,'prefix_checks':'6 passed',
                          'evaluation_seconds':time.perf_counter()-start,'articles':len(batches),
                          'note':'control is a development-selected alias, not an additional independent experiment'}
    (out/'metrics.json').write_text(json.dumps(results,indent=2,allow_nan=False),encoding='utf-8')
    (out/'resources.json').write_text(json.dumps(safety.summary(),indent=2),encoding='utf-8')
    print('DONE',out,flush=True)


if __name__=='__main__':main()
