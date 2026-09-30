"""Shrink discrete weight changes and separately correct block-output means."""
import argparse,json,time
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM,AutoTokenizer
import gpu_safety as safety
from round7_safety import initialize
from prepare_round3_holdout import MODEL,REVISION
from evaluate_refinement import sha
from run_pilot import read_texts,sequences,tensor_hash
from run_round5 import apply_prefix


def shrink(q0,q1,scale,tau):
    codes=((1-tau)*(q0/scale).round()+tau*(q1/scale).round()).round().clamp(-8,7)
    return codes*scale


def main():
    p=argparse.ArgumentParser();p.add_argument('--layer',type=int,choices=[12,18],required=True)
    p.add_argument('--seed',type=int,choices=[0,1,2],required=True);a=p.parse_args()
    out=Path(f'results/round7_l{a.layer}_s{a.seed}')
    if out.exists():raise FileExistsError(out)
    out.mkdir();initialize();torch.set_num_threads(4);torch.manual_seed(a.seed)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    start=time.perf_counter();parent=Path(f'results/round6_l{a.layer}_s{a.seed}')
    pc=json.loads((parent/'config.json').read_text());prior=json.loads((parent/'metrics.json').read_text())
    frozen=json.loads(Path('results/round6_holdout/selection.json').read_text())
    prior_external=json.loads(Path('results/round6_holdout/metrics.json').read_text())
    assert prior['_summary']['complete'] and prior_external['_summary']['complete']
    for key in ('gptq','bias','block_s192'):
        path=parent/(key+'.pt');assert sha(path)==frozen['checkpoint_sha256'][str(path)]
    tok=AutoTokenizer.from_pretrained(MODEL,revision=REVISION,local_files_only=True)
    cal=sequences(read_texts(f'data/wikitext2_20260926_s{a.seed}/calibration.jsonl'),tok,64,256)
    wiki=sequences(read_texts('data/wikitext2_20260926_s0/validation.jsonl'),tok,64,256)
    ptb=torch.load('data/round6_holdout/tokens.pt',weights_only=True)
    pm=json.loads(Path('data/round6_holdout/manifest.json').read_text())
    assert sha('data/round6_holdout/tokens.pt')==pm['tokens_sha256']
    assert list(map(tensor_hash,cal))==pc['calibration_window_hashes']
    assert list(map(tensor_hash,wiki))==pc['validation_window_hashes']
    model=AutoModelForCausalLM.from_pretrained(MODEL,revision=REVISION,local_files_only=True,
        dtype=torch.float32,attn_implementation='eager').cuda().eval().requires_grad_(False)
    model.config.use_cache=False;block=model.model.layers[a.layer];target=block.mlp.up_proj
    unused,bf,_=safety.capture(model,block,cal,'cuda',False);del unused
    prefix_hash=apply_prefix(model,'gptq',a.seed);assert prefix_hash==pc['prefix_sha256']
    assert abs(safety.evaluate(model,wiki,'cuda')['ce']-prior['common_prefix']['ce'])<1e-6
    p0=torch.load(parent/'gptq.pt',weights_only=True);p1=torch.load(parent/'block_s192.pt',weights_only=True)
    q0=p0['weight'].cuda();q1=p1['weight'].cuda();scale=torch.load(parent/'grid.pt',weights_only=True)['scale'].cuda()
    config={'layer':a.layer,'seed':a.seed,'target':pc['target'],'prefix_sha256':prefix_hash,'taus':[0,.5,1],
        'calibration_window_hashes':pc['calibration_window_hashes'],
        'development_window_hashes':{'wiki':pc['validation_window_hashes'],'ptb_seen':list(map(tensor_hash,ptb))},
        'parent_config_sha256':sha(parent/'config.json'),
        'parent_checkpoints_sha256':{str(parent/(k+'.pt')):sha(parent/(k+'.pt')) for k in ('gptq','bias','block_s192')},
        'source_sha256':{f:sha(f) for f in ['run_round7.py','round7_safety.py','run_round5.py','gpu_safety.py','run_pilot.py']},
        'selection_rule':'Minimize worst of two development-domain mean CE gaps to initial GPTQ, across seeds.',
        'bias_fit_only_on':'WikiText calibration','memory_cap_gib':4.,'pause_seconds':safety.PAUSE}
    (out/'config.json').write_text(json.dumps(config,indent=2),encoding='utf-8');torch.save({'scale':scale.cpu()},out/'grid.pt')
    metrics={}
    # Reuse exact old weights and old scores, with explicit provenance.
    for name,key,family in [('tau0','gptq','gptq'),('tau0_bias','bias','bias'),('tau1','block_s192','block192')]:
        payload=torch.load(parent/(key+'.pt'),weights_only=True)
        torch.save(payload,out/(name+'.pt'))
        metrics[name]={'wiki':prior[key],'ptb_seen':prior_external[f'l{a.layer}/s{a.seed}/{family}'],
                       'reused_from_round6':True,'fit_seconds':0.}
    with torch.no_grad():target.weight.copy_(q0)
    for domain,batches in [('wiki',wiki),('ptb_seen',ptb)]:
        assert abs(safety.evaluate(model,batches,'cuda')['ce']-metrics['tau0'][domain]['ce'])<1e-6
    for tau,name in [(.5,'tau0.5'),(1.,'tau1')]:
        tick=time.perf_counter()
        q=shrink(q0,q1,scale,tau)
        torch.testing.assert_close(q,(q/scale).round().clamp(-8,7)*scale,rtol=0,atol=0)
        with torch.no_grad():target.weight.copy_(q)
        unused,bq,_=safety.capture(model,block,cal,'cuda',False);del unused
        bias=(bf-bq).mean(1).cuda();fit=time.perf_counter()-tick;del bq
        for use_bias in (False,True):
            key=name+('_bias' if use_bias else '')
            if key in metrics:continue
            hook=block.register_forward_hook(lambda m,args,z:z+bias.view(1,1,-1)) if use_bias else None
            try:
                value={domain:safety.evaluate(model,batches,'cuda') for domain,batches in [('wiki',wiki),('ptb_seen',ptb)]}
            finally:
                if hook is not None:hook.remove()
            value.update({'reused_from_round6':False,'fit_seconds':fit,'changed_codes_vs_initial':int(((q/scale).round()!=(q0/scale).round()).sum())})
            torch.save({'target':pc['target'],'weight':q.cpu(),'bias':bias.cpu() if use_bias else None,'quantized':True},out/(key+'.pt'))
            metrics[key]=value
            (out/'metrics.json').write_text(json.dumps(metrics,indent=2,allow_nan=False),encoding='utf-8')
            (out/'resources.json').write_text(json.dumps(safety.summary(),indent=2),encoding='utf-8')
            print(key,'Wiki PPL',value['wiki']['ppl'],'seen PTB PPL',value['ptb_seen']['ppl'],flush=True)
    metrics['_summary']={'complete':True,'candidate_count':6,'new_candidates':3,'reused_candidates':3,
                         'initial_CE_reproduction_checks':2,'total_seconds':time.perf_counter()-start}
    (out/'metrics.json').write_text(json.dumps(metrics,indent=2),encoding='utf-8')
    (out/'resources.json').write_text(json.dumps(safety.summary(),indent=2),encoding='utf-8')
    print('DONE',out,flush=True)


if __name__=='__main__':main()
