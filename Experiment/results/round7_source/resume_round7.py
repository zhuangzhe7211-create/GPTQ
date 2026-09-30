"""Complete only missing frozen evaluations after an output-file write failure."""
import json,os,shutil,time
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM,AutoTokenizer
from prepare_round3_holdout import MODEL,REVISION
from evaluate_refinement import sha
from run_round5 import apply_prefix
from run_pilot import sequences,read_texts
import gpu_safety as safety
from round7_safety import initialize


def save(path,value):
    temporary=path.with_suffix('.pending.json')
    temporary.write_text(json.dumps(value,indent=2,allow_nan=False),encoding='utf-8')
    for attempt in range(5):
        try:os.replace(temporary,path);return
        except OSError:
            if attempt==4:raise
            time.sleep(.5)


def main():
    out=Path('results/round7_holdout');archive=Path('results/round7_interrupted_holdout_write')
    if archive.exists():raise FileExistsError(archive)
    metrics=json.loads((out/'metrics.json').read_text());frozen=json.loads((out/'selection.json').read_text())
    old_resources=json.loads((out/'resources.json').read_text())
    assert '_summary' not in metrics and len(metrics)==71
    archive.mkdir()
    for path in out.iterdir():
        if path.is_file():shutil.copy2(path,archive/path.name)
    shutil.copy2('results/round7_holdout.log',archive/'interrupted.log')
    shutil.copy2('evaluate_round7.py',archive/'evaluate_round7.py')
    prior_elapsed=(archive/'metrics.json').stat().st_mtime-old_resources['samples'][0]['time']
    for field in ('source_sha256','config_sha256','checkpoint_sha256'):
        for path,digest in frozen[field].items():assert sha(path)==digest
    assert sha('data/round7_holdout/manifest.json')==frozen['holdout_manifest_sha256']
    manifest=json.loads(Path('data/round7_holdout/manifest.json').read_text());datasets={}
    for name,d in manifest['datasets'].items():
        path=Path('data/round7_holdout')/(name+'.pt');assert sha(path)==d['tokens_sha256']
        datasets[name]=torch.load(path,weights_only=True)
    missing=[f'{d}/l{l}/s{s}/{f}' for d in datasets for l,fs in frozen['selected'].items()
             for s in range(3) for f in fs if f'{d}/l{l}/s{s}/{f}' not in metrics]
    assert set(missing)=={'ptb_valid/l18/s2/block192','shakespeare/l18/s2/bias_only','shakespeare/l18/s2/block192'}
    provenance={'reason':'OSError Errno 22 while saving resources.json; no method/selection changes',
                'preserved_entries':len(metrics),'missing':missing,'resume_source_sha256':sha(__file__),
                'original_selection_sha256':sha(out/'selection.json'),'interrupted_elapsed_estimate_seconds':prior_elapsed,
                'interrupted_metrics_sha256':sha(archive/'metrics.json')}
    save(out/'resume_manifest.json',provenance)
    start=time.perf_counter();initialize();torch.set_num_threads(4);torch.manual_seed(0)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    tok=AutoTokenizer.from_pretrained(MODEL,revision=REVISION,local_files_only=True)
    wiki=sequences(read_texts('data/wikitext2_20260926_s0/validation.jsonl'),tok,64,256)
    model=AutoModelForCausalLM.from_pretrained(MODEL,revision=REVISION,local_files_only=True,
        dtype=torch.float32,attn_implementation='eager').cuda().eval().requires_grad_(False)
    model.config.use_cache=False
    config=json.loads(Path('results/round7_l18_s2/config.json').read_text())
    assert apply_prefix(model,'gptq',2)==config['prefix_sha256']
    prior=json.loads(Path('results/round6_l18_s2/metrics.json').read_text())
    assert abs(safety.evaluate(model,wiki,'cuda')['ce']-prior['common_prefix']['ce'])<1e-6
    block=model.model.layers[18];calls=0
    for name in missing:
        dataset,layer,seed,family=name.split('/');key=frozen['selected']['18'][family]
        path=Path('results/round7_l18_s2')/(key+'.pt')
        assert sha(path)==frozen['checkpoint_sha256'][str(path)]
        payload=torch.load(path,weights_only=True)
        with torch.no_grad():block.mlp.up_proj.weight.copy_(payload['weight'])
        bias=payload['bias'];hook=None
        if bias is not None:
            bias=bias.cuda();hook=block.register_forward_hook(lambda m,args,z:z+bias.view(1,1,-1))
        try:metrics[name]=safety.evaluate(model,datasets[dataset],'cuda');calls+=1
        finally:
            if hook is not None:hook.remove()
        save(out/'metrics.json',metrics);print(name,metrics[name]['ppl'],flush=True)
    original=json.loads((archive/'metrics.json').read_text())
    assert all(metrics[k]==v for k,v in original.items()) and len(metrics)==74
    resume_seconds=time.perf_counter()-start
    metrics['_summary']={'complete':True,'evaluation_seconds':prior_elapsed+resume_seconds,
        'timing_note':'Interrupted phase estimated from initial telemetry to last metrics write; includes initialization. Resume precisely timed. Excludes idle gap.',
        'resume_seconds':resume_seconds,'resumed_dataset_evaluations':calls,'actual_dataset_evaluations':62,
        'unique_candidate_contexts':30,'all_choices_frozen_before_evaluation':True,'prefix_checks':6,
        'resume_prefix_checks':1,'datasets':list(datasets),'preserved_metric_entries':71}
    resources=safety.summary()
    for key in ('peak_allocated_gib','peak_reserved_gib'):resources[key]=max(resources[key],old_resources[key])
    resources['samples']=old_resources['samples']+resources['samples']
    resources['note']='Combined successful samples before write failure and resume; approximately last 31 seconds before failure had no saved telemetry.'
    save(out/'resources.json',resources);save(out/'metrics.json',metrics)
    print('DONE frozen confirmation; 71 prior entries unchanged, 3 missing entries computed.',flush=True)


if __name__=='__main__':main()
