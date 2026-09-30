"""Fixed-grid STE reconstruction: branch vs gated product vs full MLP output."""
import argparse
import json
from pathlib import Path
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from evaluate_refinement import sha
from prepare_round3_holdout import MODEL, REVISION
from quant_core import RowQuantizer, official_rescomp, reference_nd, grouped_official
from run_pilot import capture, evaluate, read_texts, sequences, tensor_hash

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
    xf,xq,gradient=xf.cuda(),xq.cuda(),gradient.cuda()
    vf,vq=vf.cuda(),vq.cuda()
    with torch.no_grad():
        af=mlp.act_fn(mlp.gate_proj(xf.T)).T
        aq=mlp.act_fn(mlp.gate_proj(xq.T)).T
        uf=w0@xf
        teachers={'branch':uf,'product':af*uf,'mlp':down@(af*uf)}
        av=mlp.act_fn(mlp.gate_proj(vq.T)).T
        vteacher=down@(mlp.act_fn(mlp.gate_proj(vf.T)).T*(w0@vf))
        del af,vf,uf
    # Verify the cached algebra agrees with the actual MLP for one validation sequence.
    with torch.no_grad():
        actual=mlp(vq[:,:256].T).T
        torch.testing.assert_close(actual,prediction(w0,vq[:,:256],av[:,:256],down,'mlp'),rtol=1e-5,atol=1e-6)
    init_path=previous/'gptq_reference.pt'
    initial=torch.load(init_path,weights_only=True)['weight'].cuda()
    scale=RowQuantizer(w0,4).scale
    torch.testing.assert_close(fake_weight((initial/scale).round(),scale),initial,rtol=0,atol=0)
    config={'prefix':a.prefix,'seed':a.seed,'target':TARGET,'model':MODEL,'revision':REVISION,
            'bits':4,'dtype':'float32','samples':64,'seq_len':256,'eval_samples':64,
            'calibration_window_hashes':list(map(tensor_hash,cal)),
            'validation_window_hashes':list(map(tensor_hash,val)),
            'prefix_sha256':prefix_hash,'initial_sha256':sha(init_path),
            'source_sha256':{name:sha(name) for name in ['run_round5.py','quant_core.py','run_pilot.py']},
            'learning_rates':LRS,'steps':STEPS,'alphas':ALPHAS,'batch_tokens':256,
            'warning':'Only up_proj optimized; gate/down fixed. STE calibration reconstruction, not full-model quantization.'}
    (out/'config.json').write_text(json.dumps(config,indent=2),encoding='utf-8')
    results={'common_prefix':prefix_result};traces={}
    @torch.no_grad()
    def record(name,weight,seconds):
        assert torch.isfinite(weight).all()
        # Check fixed grid using integer-code reconstruction, not a newly fitted quantizer.
        torch.testing.assert_close(weight,torch.round(weight/scale).clamp(-8,7)*scale,rtol=0,atol=0)
        target.weight.copy_(weight)
        value=evaluate(model,val,'cuda')
        value['optimization_seconds']=seconds
        value['validation_mlp_mse']=float((prediction(weight,vq,av,down,'mlp')-vteacher).square().mean())
        for objective in ('branch','product','mlp'):
            value[f'calibration_{objective}_mse']=float((prediction(weight,xq,aq,down,objective)-teachers[objective]).square().mean())
        value['changed_codes_vs_gptq']=int((torch.round(weight/scale)!=torch.round(initial/scale)).sum())
        results[name]=value
        torch.save({'target':TARGET,'weight':weight.cpu()},out/f'{name}.pt')
        (out/'metrics.json').write_text(json.dumps(results,indent=2,allow_nan=False),encoding='utf-8')
        print(name,'PPL',value['ppl'],'seconds',round(seconds,3),flush=True)
    record('gptq_reference',initial,pm['gptq_reference']['quantization_seconds'])
    assert abs(results['gptq_reference']['ce']-pm['gptq_reference']['ce'])<1e-6
    record('rtn_target',RowQuantizer(w0,4).quantize(w0),0.)
    one=torch.ones(xq.shape[1],device='cuda')
    for alpha in ALPHAS:
        for family in ('rescomp','gptaq','weak1'):
            tick=time.perf_counter()
            if family=='rescomp': candidate=official_rescomp(w0,xf,xq,one,alpha=alpha)
            elif family=='gptaq': candidate=reference_nd(w0,xf,xq,one,alpha=alpha,beta=0.)
            else: candidate=grouped_official(w0,xf,xq,gradient,1,.1,4,.01,alpha,128)
            torch.cuda.synchronize()
            record(f'{family}_a{alpha:g}',candidate,time.perf_counter()-tick)
    del candidate,gradient,xf
    generator=torch.Generator(device='cuda').manual_seed(20260930+a.seed)
    batches=torch.randint(xq.shape[1],(max(STEPS),256),generator=generator,device='cuda')
    config['optimization_token_indices_sha256']=tensor_hash(batches.cpu())
    (out/'config.json').write_text(json.dumps(config,indent=2),encoding='utf-8')
    for objective in ('branch','product','mlp'):
        for lr in LRS:
            code=(initial/scale).round().detach().requires_grad_(True)
            optimizer=torch.optim.Adam([code],lr=lr)
            normalizer=teachers[objective].square().mean().clamp_min(1e-12)
            seconds=0.;trace=[]
            for step,index in enumerate(batches,1):
                tick=time.perf_counter()
                optimizer.zero_grad(set_to_none=True)
                q=fake_weight(code,scale)
                estimate=prediction(q,xq[:,index],aq[:,index],down,objective)
                loss=(estimate-teachers[objective][:,index]).square().mean()/normalizer
                if not torch.isfinite(loss):raise RuntimeError('Nonfinite reconstruction objective')
                loss.backward();optimizer.step()
                with torch.no_grad():code.clamp_(-8,7)
                torch.cuda.synchronize();seconds+=time.perf_counter()-tick
                trace.append(float(loss.detach()))
                if step in STEPS:
                    record(f'{objective}_lr{lr:g}_s{step}',fake_weight(code.detach(),scale),seconds)
            traces[f'{objective}_lr{lr:g}']=trace
            del optimizer,code,q,estimate,loss
    results['_summary']={'complete':True,'candidate_count':35,'prefix_and_initial_regression':'passed',
        'cached_mlp_forward_check':'passed','total_seconds':time.perf_counter()-start,
        'peak_cuda_bytes':torch.cuda.max_memory_allocated()}
    assert len(results)-2==35
    (out/'metrics.json').write_text(json.dumps(results,indent=2,allow_nan=False),encoding='utf-8')
    (out/'optimization_trace.json').write_text(json.dumps(traces),encoding='utf-8')
    print('DONE',out,flush=True)


if __name__=='__main__':main()
