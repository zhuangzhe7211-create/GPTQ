"""Exploratory follow-up: preserve the full decoder block, including residual drift."""
import argparse,json,time
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM,AutoTokenizer
import gpu_safety as safety
from evaluate_refinement import sha
from prepare_round3_holdout import MODEL,REVISION
from run_pilot import read_texts,sequences,tensor_hash
from run_round5 import TARGET,LRS,STEPS,apply_prefix,fake_weight,prediction
from quant_core import RowQuantizer


def capture_residual(model,batches):
    residual=[]
    handle=model.model.layers[12].post_attention_layernorm.register_forward_pre_hook(
        lambda module,args:residual.append(args[0].detach().cpu().reshape(-1,args[0].shape[-1]).T))
    try:
        x,unused,_=safety.capture(model,model.get_submodule(TARGET),batches,'cuda',False)
        del unused
    finally:handle.remove()
    return x,torch.cat(residual,1)


def main():
    p=argparse.ArgumentParser();p.add_argument('--prefix',choices=['rtn','gptq'],required=True)
    p.add_argument('--seed',type=int,required=True);a=p.parse_args()
    out=Path(f'results/round5_block_{a.prefix}_s{a.seed}')
    if out.exists():raise FileExistsError(out)
    out.mkdir();safety.initialize();torch.set_num_threads(4);torch.manual_seed(a.seed)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    start=time.perf_counter()
    tok=AutoTokenizer.from_pretrained(MODEL,revision=REVISION,local_files_only=True)
    cal=sequences(read_texts(f'data/wikitext2_20260926_s{a.seed}/calibration.jsonl'),tok,64,256)
    val=sequences(read_texts('data/wikitext2_20260926_s0/validation.jsonl'),tok,64,256)
    original_run=Path(f'results/round5_{a.prefix}_s{a.seed}')
    previous=json.loads((original_run/'config.json').read_text())
    assert previous['calibration_window_hashes']==list(map(tensor_hash,cal))
    model=AutoModelForCausalLM.from_pretrained(MODEL,revision=REVISION,local_files_only=True,
        dtype=torch.float32,attn_implementation='eager').cuda().eval().requires_grad_(False)
    model.config.use_cache=False;block=model.model.layers[12];mlp=block.mlp
    target=model.get_submodule(TARGET);w0=target.weight.detach().clone();down=mlp.down_proj.weight.detach()
    unused,bf,_=safety.capture(model,block,cal,'cuda',False);del unused
    unused,bv,_=safety.capture(model,block,val,'cuda',False);del unused
    prefix_hash=apply_prefix(model,a.prefix,a.seed)
    prefix=safety.evaluate(model,val,'cuda')
    prior_metrics=json.loads((original_run/'metrics.json').read_text())
    assert abs(prefix['ce']-prior_metrics['common_prefix']['ce'])<1e-6
    xq,rq=capture_residual(model,cal);vq,rv=capture_residual(model,val)
    gc=mlp.gate_proj.weight.detach().cpu()
    with torch.no_grad():aq=mlp.act_fn(gc@xq);av=mlp.act_fn(gc@vq)
    teacher=bf-rq;vteacher=bv-rv
    del bf,bv,rq,rv
    initial=torch.load(original_run/'gptq_reference.pt',weights_only=True)['weight'].cuda()
    scale=RowQuantizer(w0,4).scale
    torch.testing.assert_close(fake_weight((initial/scale).round(),scale),initial,rtol=0,atol=0)
    generator=torch.Generator().manual_seed(20260930+a.seed)
    batches=torch.randint(xq.shape[1],(max(STEPS),256),generator=generator)
    assert tensor_hash(batches)==previous['optimization_token_indices_sha256']
    config={'prefix':a.prefix,'seed':a.seed,'target':TARGET,'prefix_sha256':prefix_hash,
            'calibration_window_hashes':previous['calibration_window_hashes'],
            'validation_window_hashes':previous['validation_window_hashes'],
            'source_sha256':{name:sha(name) for name in ['run_round5_block.py','run_round5.py','gpu_safety.py']},
            'initial_sha256':sha(original_run/'gptq_reference.pt'),
            'normalizer':'mean square of (teacher full block output - current pre-MLP residual)',
            'adaptive_followup':'Added after first-seed development results, before any round-5 holdout evaluation.'}
    (out/'config.json').write_text(json.dumps(config,indent=2),encoding='utf-8')
    results={'common_prefix':prefix};traces={}
    @torch.no_grad()
    def mse(weight,x,gate,y):
        total=0.
        for begin in range(0,x.shape[1],512):
            sl=slice(begin,begin+512)
            total+=float((prediction(weight,x[:,sl].cuda(),gate[:,sl].cuda(),down,'mlp')-y[:,sl].cuda()).square().sum())
            safety.check()
        return total/y.numel()
    initial_mse=mse(initial,xq,aq,teacher)
    for lr in LRS:
        model.cpu();torch.cuda.empty_cache()
        code=(initial/scale).round().detach().requires_grad_(True);optimizer=torch.optim.Adam([code],lr=lr)
        normalizer=max(float(teacher.square().mean()),1e-12);seconds=0.;trace=[]
        for step,index in enumerate(batches,1):
            tick=time.perf_counter();optimizer.zero_grad(set_to_none=True);value=0.
            for micro in index.split(64):
                q=fake_weight(code,scale)
                loss=(prediction(q,xq[:,micro].cuda(),aq[:,micro].cuda(),down,'mlp')-teacher[:,micro].cuda()).square().mean()/normalizer/4
                if not torch.isfinite(loss):raise RuntimeError('Nonfinite block reconstruction')
                loss.backward();value+=float(loss.detach());del q,loss
            optimizer.step()
            with torch.no_grad():code.clamp_(-8,7)
            torch.cuda.synchronize();seconds+=time.perf_counter()-tick;trace.append(value);safety.pause()
            if step in STEPS:
                name=f'block_lr{lr:g}_s{step}';q=fake_weight(code.detach(),scale)
                model.cuda();target.weight.data.copy_(q)
                result=safety.evaluate(model,val,'cuda')
                result.update({'optimization_seconds':seconds,'calibration_block_mse':mse(q,xq,aq,teacher),
                  'validation_block_mse':mse(q,vq,av,vteacher),'initial_calibration_block_mse':initial_mse,
                  'changed_codes_vs_gptq':int((torch.round(q/scale)!=torch.round(initial/scale)).sum())})
                torch.testing.assert_close(q,torch.round(q/scale).clamp(-8,7)*scale,rtol=0,atol=0)
                torch.save({'weight':q.cpu(),'target':TARGET},out/f'{name}.pt');results[name]=result
                (out/'metrics.json').write_text(json.dumps(results,indent=2,allow_nan=False),encoding='utf-8')
                (out/'resources.json').write_text(json.dumps(safety.summary(),indent=2),encoding='utf-8')
                print(name,'PPL',result['ppl'],'block MSE',result['calibration_block_mse'],flush=True)
                model.cpu();torch.cuda.empty_cache();del q
        traces[f'block_lr{lr:g}']=trace;del code,optimizer
    results['_summary']={'complete':True,'candidate_count':6,'total_seconds':time.perf_counter()-start}
    (out/'metrics.json').write_text(json.dumps(results,indent=2,allow_nan=False),encoding='utf-8')
    (out/'optimization_trace.json').write_text(json.dumps(traces),encoding='utf-8')
    (out/'resources.json').write_text(json.dumps(safety.summary(),indent=2),encoding='utf-8')
    print('DONE',out,flush=True)


if __name__=='__main__':main()
