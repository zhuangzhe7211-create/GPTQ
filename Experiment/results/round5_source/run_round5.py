"""Fixed-grid STE reconstruction: branch vs gated product vs full MLP output."""
import argparse
import json
from pathlib import Path
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from evaluate_refinement import sha
from prepare_round3_holdout import MODEL, REVISION
from quant_core import RowQuantizer, official_rescomp, reference_nd, make_weights
from run_pilot import read_texts, sequences, tensor_hash
import gpu_safety as safety
from gpu_safety import capture, evaluate

TARGET = 'model.layers.12.mlp.up_proj'
ALPHAS = (0., .0625, .125, .25, .5)
STEPS = (32,96,192)
LRS = (.01,.03)


def fake_weight(code, scale):
    """Forward is exact fixed-grid round; backward treats round as identity."""
    rounded = code.round().clamp(-8,7)
    return (code + (rounded-code).detach()) * scale


def prediction(weight,x,gate,down,objective):
    up = weight @ x
    if objective == 'branch':
        return up
    product = gate * up
    return product if objective == 'product' else down @ product


@torch.no_grad()
def apply_prefix(model,prefix,seed):
    if prefix == 'gptq':
        path = Path(f'results/round4_prefix_s{seed}')
        manifest = json.loads((path/'manifest.json').read_text())
        assert sha(path/'weights.pt') == manifest['weight_sha256']
        for name,weight in torch.load(path/'weights.pt',weights_only=True).items():
            model.get_submodule(name).weight.copy_(weight)
        return manifest['weight_sha256']
    for block in model.model.layers[:12]:
        for module in block.modules():
            if isinstance(module,torch.nn.Linear):
                module.weight.copy_(RowQuantizer(module.weight,4).quantize(module.weight))
    return 'CUDA RTN original row grid'


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--prefix',choices=['rtn','gptq'],required=True)
    p.add_argument('--seed',type=int,required=True)
    a=p.parse_args()
    out=Path(f'results/round5_{a.prefix}_s{a.seed}')
    if out.exists(): raise FileExistsError(out)
    out.mkdir()
    safety.initialize()
    torch.set_num_threads(4);torch.manual_seed(a.seed)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    start=time.perf_counter()
    tok=AutoTokenizer.from_pretrained(MODEL,revision=REVISION,local_files_only=True)
    cal=sequences(read_texts(f'data/wikitext2_20260926_s{a.seed}/calibration.jsonl'),tok,64,256)
    val=sequences(read_texts('data/wikitext2_20260926_s0/validation.jsonl'),tok,64,256)
    previous=Path(f'results/round4_{a.prefix}_mlp_s{a.seed}')
    pc=json.loads((previous/'config.json').read_text());pm=json.loads((previous/'metrics.json').read_text())
    assert list(map(tensor_hash,cal))==pc['calibration_window_hashes']
    assert list(map(tensor_hash,val))==pc['validation_window_hashes']
    assert not set(map(tensor_hash,cal)) & set(map(tensor_hash,val))
    model=AutoModelForCausalLM.from_pretrained(MODEL,revision=REVISION,local_files_only=True,
          dtype=torch.float32,attn_implementation='eager').cuda().eval().requires_grad_(False)
    model.config.use_cache=False
    target=model.get_submodule(TARGET);mlp=model.model.layers[12].mlp
    w0=target.weight.detach().clone();down=mlp.down_proj.weight.detach()
    xf,unused,gradient=capture(model,target,cal,'cuda',True);del unused
    vf,unused,_=capture(model,target,val,'cuda',False);del unused
    prefix_hash=apply_prefix(model,a.prefix,a.seed)
    prefix_result=evaluate(model,val,'cuda')
    assert abs(prefix_result['ce']-pm['common_rtn_prefix']['ce'])<1e-6
    xq,unused,_=capture(model,target,cal,'cuda',False);del unused
    vq,unused,_=capture(model,target,val,'cuda',False);del unused
    # Large activation/gradient caches stay on CPU; only bounded slices enter CUDA.
    wc=w0.cpu();gc=mlp.gate_proj.weight.detach().cpu();dc=down.cpu()
    with torch.no_grad():
        af=mlp.act_fn(gc@xf)
        aq=mlp.act_fn(gc@xq)
        uf=wc@xf
        product=af*uf
        teachers={'branch':uf,'product':product,'mlp':dc@product}
        av=mlp.act_fn(gc@vq)
        vteacher=dc@(mlp.act_fn(gc@vf)*(wc@vf))
        del af,vf,uf
    # Verify the cached algebra agrees with the actual MLP for one validation sequence.
    with torch.no_grad():
        actual=mlp(vq[:,:256].cuda().T).T
        torch.testing.assert_close(actual,prediction(w0,vq[:,:256].cuda(),av[:,:256].cuda(),down,'mlp'),rtol=1e-4,atol=2e-6)
        del actual
    init_path=previous/'gptq_reference.pt'
    initial=torch.load(init_path,weights_only=True)['weight'].cuda()
    scale=RowQuantizer(w0,4).scale
    torch.testing.assert_close(fake_weight((initial/scale).round(),scale),initial,rtol=0,atol=0)
    config={'prefix':a.prefix,'seed':a.seed,'target':TARGET,'model':MODEL,'revision':REVISION,
            'bits':4,'dtype':'float32','samples':64,'seq_len':256,'eval_samples':64,
            'calibration_window_hashes':list(map(tensor_hash,cal)),
            'validation_window_hashes':list(map(tensor_hash,val)),
            'prefix_sha256':prefix_hash,'initial_sha256':sha(init_path),
            'source_sha256':{name:sha(name) for name in ['run_round5.py','quant_core.py','run_pilot.py','gpu_safety.py']},
            'learning_rates':LRS,'steps':STEPS,'alphas':ALPHAS,'batch_tokens':256,
            'microbatch_tokens':64,'gradient_accumulation':4,'cache_device':'cpu',
            'memory_cap_gib':safety.CAP_GIB,'teacher_cache_arithmetic':'CPU float32',
            'warning':'Only up_proj optimized; gate/down fixed. STE calibration reconstruction, not full-model quantization.'}
    (out/'config.json').write_text(json.dumps(config,indent=2),encoding='utf-8')
    results={'common_prefix':prefix_result};traces={}
    @torch.no_grad()
    def record(name,weight,seconds):
        safety.check(True)
        model.cuda()
        assert torch.isfinite(weight).all()
        # Check fixed grid using integer-code reconstruction, not a newly fitted quantizer.
        torch.testing.assert_close(weight,torch.round(weight/scale).clamp(-8,7)*scale,rtol=0,atol=0)
        target.weight.copy_(weight)
        value=evaluate(model,val,'cuda')
        value['optimization_seconds']=seconds
        errors={k:0. for k in teachers}
        for start in range(0,xq.shape[1],512):
            sl=slice(start,start+512)
            up=weight@xq[:,sl].cuda();product=aq[:,sl].cuda()*up
            preds={'branch':up,'product':product,'mlp':down@product}
            for objective,pred in preds.items():
                errors[objective]+=float((pred-teachers[objective][:,sl].cuda()).square().sum())
            safety.check()
        for objective in teachers:
            value[f'calibration_{objective}_mse']=errors[objective]/teachers[objective].numel()
        error=0.
        for start in range(0,vq.shape[1],512):
            sl=slice(start,start+512)
            error+=float((prediction(weight,vq[:,sl].cuda(),av[:,sl].cuda(),down,'mlp')-vteacher[:,sl].cuda()).square().sum())
            safety.check()
        value['validation_mlp_mse']=error/vteacher.numel()
        value['changed_codes_vs_gptq']=int((torch.round(weight/scale)!=torch.round(initial/scale)).sum())
        results[name]=value
        torch.save({'target':TARGET,'weight':weight.cpu()},out/f'{name}.pt')
        (out/'metrics.json').write_text(json.dumps(results,indent=2,allow_nan=False),encoding='utf-8')
        (out/'resources.json').write_text(json.dumps(safety.summary(),indent=2),encoding='utf-8')
        print(name,'PPL',value['ppl'],'seconds',round(seconds,3),flush=True)
        torch.cuda.empty_cache();safety.pause()
    record('gptq_reference',initial,pm['gptq_reference']['quantization_seconds'])
    assert abs(results['gptq_reference']['ce']-pm['gptq_reference']['ce'])<1e-6
    record('rtn_target',RowQuantizer(w0,4).quantize(w0),0.)
    one=torch.ones(xq.shape[1],device='cuda')
    score=make_weights(gradient,1,.1)[0][1].cuda()
    del gradient
    xf_gpu,xq_gpu=xf.cuda(),xq.cuda()
    for alpha in ALPHAS:
        for family in ('rescomp','gptaq','weak1'):
            tick=time.perf_counter()
            if family=='rescomp': candidate=official_rescomp(w0,xf_gpu,xq_gpu,one,alpha=alpha)
            elif family=='gptaq': candidate=reference_nd(w0,xf_gpu,xq_gpu,one,alpha=alpha,beta=0.)
            else: candidate=official_rescomp(w0,xf_gpu,xq_gpu,score,alpha=alpha)
            torch.cuda.synchronize()
            record(f'{family}_a{alpha:g}',candidate,time.perf_counter()-tick)
    del candidate,xf,xf_gpu,xq_gpu
    generator=torch.Generator().manual_seed(20260930+a.seed)
    batches=torch.randint(xq.shape[1],(max(STEPS),256),generator=generator)
    config['optimization_token_indices_sha256']=tensor_hash(batches.cpu())
    (out/'config.json').write_text(json.dumps(config,indent=2),encoding='utf-8')
    for objective in ('branch','product','mlp'):
        for lr in LRS:
            model.cpu();torch.cuda.empty_cache();safety.check(True)
            code=(initial/scale).round().detach().requires_grad_(True)
            optimizer=torch.optim.Adam([code],lr=lr)
            normalizer=max(float(teachers[objective].square().mean()),1e-12)
            seconds=0.;trace=[]
            for step,index in enumerate(batches,1):
                tick=time.perf_counter()
                optimizer.zero_grad(set_to_none=True)
                loss_value=0.
                for micro in index.split(64):
                    q=fake_weight(code,scale)
                    estimate=prediction(q,xq[:,micro].cuda(),aq[:,micro].cuda(),down,objective)
                    loss=(estimate-teachers[objective][:,micro].cuda()).square().mean()/normalizer/4
                    if not torch.isfinite(loss):raise RuntimeError('Nonfinite reconstruction objective')
                    loss.backward();loss_value+=float(loss.detach())
                    del q,estimate,loss
                optimizer.step()
                with torch.no_grad():code.clamp_(-8,7)
                torch.cuda.synchronize();seconds+=time.perf_counter()-tick
                trace.append(loss_value)
                safety.pause()
                if step in STEPS:
                    record(f'{objective}_lr{lr:g}_s{step}',fake_weight(code.detach(),scale),seconds)
                    model.cpu();torch.cuda.empty_cache()
            traces[f'{objective}_lr{lr:g}']=trace
            del optimizer,code
    results['_summary']={'complete':True,'candidate_count':35,'prefix_and_initial_regression':'passed',
        'cached_mlp_forward_check':'passed','total_seconds':time.perf_counter()-start,
        'peak_cuda_bytes':torch.cuda.max_memory_allocated()}
    assert len(results)-2==35
    (out/'metrics.json').write_text(json.dumps(results,indent=2,allow_nan=False),encoding='utf-8')
    (out/'optimization_trace.json').write_text(json.dumps(traces),encoding='utf-8')
    (out/'resources.json').write_text(json.dumps(safety.summary(),indent=2),encoding='utf-8')
    print('DONE',out,flush=True)


if __name__=='__main__':main()
