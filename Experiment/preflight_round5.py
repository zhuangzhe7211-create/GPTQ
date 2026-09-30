"""Resource-only short run. Its scores are not scientific quality evidence."""
import json
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM,AutoTokenizer
import gpu_safety as safety
from prepare_round3_holdout import MODEL,REVISION
from quant_core import official_rescomp
from run_round5 import fake_weight,prediction
from run_pilot import read_texts,sequences

out=Path('results/round5_resource_preflight');out.mkdir(exist_ok=False)
safety.initialize();torch.set_num_threads(4)
torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
tok=AutoTokenizer.from_pretrained(MODEL,revision=REVISION,local_files_only=True)
batches=sequences(read_texts('data/wikitext2_20260926_s0/calibration.jsonl'),tok,2,256)
model=AutoModelForCausalLM.from_pretrained(MODEL,revision=REVISION,local_files_only=True,
    dtype=torch.float32,attn_implementation='eager').cuda().eval().requires_grad_(False)
model.config.use_cache=False
mlp=model.model.layers[12].mlp;w0=mlp.up_proj.weight.detach().clone();down=mlp.down_proj.weight.detach()
x,_,grad=safety.capture(model,mlp.up_proj,batches,'cuda',True)
xlarge=x.repeat(1,32).cuda()
q=official_rescomp(w0,xlarge,xlarge,torch.ones(xlarge.shape[1],device='cuda'))
del xlarge,grad
from quant_core import RowQuantizer
scale=RowQuantizer(w0,4).scale
micro=x[:,:64].cuda()
with torch.no_grad():gate=mlp.act_fn(mlp.gate_proj(micro.T)).T;target=down@(gate*(w0@micro))
model.cpu();torch.cuda.empty_cache()
code=(q/scale).round().detach().requires_grad_(True);opt=torch.optim.Adam([code],lr=.01)
for _ in range(3):
    opt.zero_grad(set_to_none=True)
    for _ in range(4):
        loss=(prediction(fake_weight(code,scale),micro,gate,down,'mlp')-target).square().mean()/4
        loss.backward()
    opt.step()
    with torch.no_grad():code.clamp_(-8,7)
    safety.pause()
model.cuda();model.model.layers[12].mlp.up_proj.weight.data.copy_(fake_weight(code.detach(),scale))
safety.evaluate(model,batches,'cuda')
result=safety.summary();result['quality_evidence']=False
assert result['peak_allocated_gib']<3.5
assert max(x['used_mib'] for x in result['samples'])<7*1024
(out/'resources.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
print('PREFLIGHT PASSED',json.dumps({k:v for k,v in result.items() if k!='samples'}))
