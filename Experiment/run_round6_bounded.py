"""Adaptive follow-up: cap directional anisotropy at 10 instead of 897."""
import argparse,json,time
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM,AutoTokenizer
import gpu_safety as safety
from evaluate_refinement import sha
from prepare_round3_holdout import MODEL,REVISION
from run_pilot import read_texts,sequences,tensor_hash
from run_round5 import apply_prefix,fake_weight,prediction
from quant_core import RowQuantizer,reference_nd,make_weights

from run_round6 import capture_residual

STEPS=(192,576)
FAMILIES=('bounded',)

def objective(error,family,score,direction):
    return error.square().mean()+9/error.shape[0]*(error*direction).sum(0).square().mean()


def main():
    p=argparse.ArgumentParser();p.add_argument('--layer',type=int,choices=[12,18],required=True)
    p.add_argument('--seed',type=int,choices=[0,1,2],required=True);a=p.parse_args()
    out=Path(f'results/round6_bounded_l{a.layer}_s{a.seed}')
    if out.exists():raise FileExistsError(out)
    out.mkdir();safety.initialize();torch.set_num_threads(4);torch.manual_seed(a.seed)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    start=time.perf_counter()
    tok=AutoTokenizer.from_pretrained(MODEL,revision=REVISION,local_files_only=True)
    cal=sequences(read_texts(f'data/wikitext2_20260926_s{a.seed}/calibration.jsonl'),tok,64,256)
    val=sequences(read_texts('data/wikitext2_20260926_s0/validation.jsonl'),tok,64,256)
    model=AutoModelForCausalLM.from_pretrained(MODEL,revision=REVISION,local_files_only=True,
        dtype=torch.float32,attn_implementation='eager').cuda().eval().requires_grad_(False)
    model.config.use_cache=False;block=model.model.layers[a.layer];mlp=block.mlp
    target=mlp.up_proj;target_name=f'model.layers.{a.layer}.mlp.up_proj'
    w0=target.weight.detach().clone();down=mlp.down_proj.weight.detach()
    unused,bf,gradient=safety.capture(model,block,cal,'cuda',True);del unused
    unused,bv,_=safety.capture(model,block,val,'cuda',False);del unused
    prefix_hash=apply_prefix(model,'gptq',a.seed)
    prefix=safety.evaluate(model,val,'cuda')
    prior=json.loads(Path(f'results/round5_gptq_s{a.seed}/metrics.json').read_text())
    assert abs(prefix['ce']-prior['common_prefix']['ce'])<1e-6
    xq,rq=capture_residual(model,a.layer,cal);vq,rv=capture_residual(model,a.layer,val)
    gc=mlp.gate_proj.weight.detach().cpu()
    with torch.no_grad():aq=mlp.act_fn(gc@xq);av=mlp.act_fn(gc@vq)
    teacher=bf-rq;vteacher=bv-rv;del bf,bv,rq,rv
    score=make_weights(gradient,groups=1,rho=.1,clip=10.,power=1.)[0][1]
    direction=gradient/gradient.norm(dim=0,keepdim=True).clamp_min(1e-12);del gradient
    parent=Path(f'results/round6_l{a.layer}_s{a.seed}')
    initial=torch.load(parent/'gptq.pt',weights_only=True)['weight'].cuda()
    parent_config=json.loads((parent/'config.json').read_text())
    assert tensor_hash(initial.cpu())==parent_config['initial_weight_sha256']
    scale=RowQuantizer(w0,4).scale
    torch.testing.assert_close(initial,(initial/scale).round()*scale,rtol=0,atol=0)
    torch.save({'scale':scale.cpu()},out/'grid.pt')
    indices=torch.randint(xq.shape[1],(max(STEPS),256),generator=torch.Generator().manual_seed(20260930+a.seed))
    config={'layer':a.layer,'seed':a.seed,'target':target_name,'prefix_sha256':prefix_hash,
        'model':MODEL,'revision':REVISION,'dtype':'float32','bits':4,'lr':.03,'steps':STEPS,
        'calibration_window_hashes':list(map(tensor_hash,cal)),'validation_window_hashes':list(map(tensor_hash,val)),
        'token_indices_sha256':tensor_hash(indices),'initial_weight_sha256':tensor_hash(initial.cpu()),
        'source_sha256':{name:sha(name) for name in ['run_round6_bounded.py','run_round6.py','run_round5.py','quant_core.py','gpu_safety.py','run_pilot.py']},
        'normalizer':float(teacher.square().mean()),'prefix_blocks':12,'token_rho':.1,'direction_lambda':9/teacher.shape[0],'max_per_token_error_curvature_ratio':10.,'adaptive_followup_before_external_test':True,
        'batch_tokens':256,'microbatch_tokens':64,'memory_cap_gib':safety.CAP_GIB}
    (out/'config.json').write_text(json.dumps(config,indent=2),encoding='utf-8')
    results={'common_prefix':prefix};traces={}
    @torch.no_grad()
    def errors(weight,x,gate,y,bias=None):
        totals=0.;summed=torch.zeros(y.shape[0],device='cuda')
        for begin in range(0,x.shape[1],512):
            sl=slice(begin,begin+512)
            e=prediction(weight,x[:,sl].cuda(),gate[:,sl].cuda(),down,'mlp')-y[:,sl].cuda()
            if bias is not None:e=e+bias[:,None]
            totals+=float(e.square().sum());summed+=e.sum(1);safety.pause()
        return totals/y.numel(),summed/x.shape[1]
    @torch.no_grad()
    def record(name,weight,seconds,bias=None,quantized=True):
        safety.check(True);model.cuda();target.weight.copy_(weight)
        if quantized:torch.testing.assert_close(weight,(weight/scale).round().clamp(-8,7)*scale,rtol=0,atol=0)
        hook=block.register_forward_hook(lambda module,args,output:output+bias.view(1,1,-1)) if bias is not None else None
        try:value=safety.evaluate(model,val,'cuda')
        finally:
            if hook is not None:hook.remove()
        value['calibration_block_mse']=errors(weight,xq,aq,teacher,bias)[0]
        value['validation_block_mse']=errors(weight,vq,av,vteacher,bias)[0]
        value.update({'optimization_seconds':seconds,'quantized':quantized,'extra_bias_parameters':0 if bias is None else bias.numel(),
                      'changed_codes_vs_gptq':int(((weight/scale).round()!=(initial/scale).round()).sum()) if quantized else None})
        if a.layer==12 and name=='gptq':assert abs(value['ce']-prior['gptq_reference']['ce'])<1e-6
        torch.save({'target':target_name,'weight':weight.cpu(),'bias':None if bias is None else bias.cpu(),'quantized':quantized},out/f'{name}.pt')
        results[name]=value
        (out/'metrics.json').write_text(json.dumps(results,indent=2,allow_nan=False),encoding='utf-8')
        (out/'resources.json').write_text(json.dumps(safety.summary(),indent=2),encoding='utf-8')
        print(name,'PPL',value['ppl'],'MSE',value['calibration_block_mse'],flush=True)
    for family in FAMILIES:
        model.cpu();torch.cuda.empty_cache()
        code=(initial/scale).round().detach().requires_grad_(True);optimizer=torch.optim.Adam([code],lr=.03)
        normalizer=max(config['normalizer'],1e-12);seconds=0.;trace=[]
        for step,index in enumerate(indices,1):
            tick=time.perf_counter();optimizer.zero_grad(set_to_none=True);value=0.
            for micro in index.split(64):
                q=code*scale if family=='continuous' else fake_weight(code,scale)
                e=prediction(q,xq[:,micro].cuda(),aq[:,micro].cuda(),down,'mlp')-teacher[:,micro].cuda()
                loss=objective(e,family,score[micro].cuda(),direction[:,micro].cuda())/normalizer/4
                if not torch.isfinite(loss):raise RuntimeError('Nonfinite reconstruction loss')
                loss.backward();value+=float(loss.detach());del q,e,loss
            optimizer.step()
            with torch.no_grad():code.clamp_(-8,7)
            torch.cuda.synchronize();seconds+=time.perf_counter()-tick;trace.append(value);safety.pause()
            if step in STEPS:
                q=code.detach()*scale if family=='continuous' else fake_weight(code.detach(),scale)
                record(f'{family}_s{step}',q,seconds,quantized=family!='continuous')
                model.cpu();torch.cuda.empty_cache();del q
        traces[family]=trace;del code,optimizer
    results['_summary']={'complete':True,'candidate_count':2,'total_seconds':time.perf_counter()-start}
    (out/'metrics.json').write_text(json.dumps(results,indent=2),encoding='utf-8')
    (out/'optimization_trace.json').write_text(json.dumps(traces),encoding='utf-8')
    (out/'resources.json').write_text(json.dumps(safety.summary(),indent=2),encoding='utf-8')
    print('DONE',out,flush=True)


if __name__=='__main__':main()
