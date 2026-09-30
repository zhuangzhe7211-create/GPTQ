"""Check whether CPU/GPU RTN division changes exact quantization bins."""
import json
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from prepare_round3_holdout import MODEL, REVISION
from quant_core import RowQuantizer
from run_pilot import evaluate, sequences, read_texts

torch.set_num_threads(4)
torch.backends.cuda.matmul.allow_tf32=False
model=AutoModelForCausalLM.from_pretrained(MODEL,revision=REVISION,local_files_only=True,
         dtype=torch.float32,attn_implementation='eager').cuda().eval().requires_grad_(False)
result={}
cpu_weights={}
for i,block in enumerate(model.model.layers[:12]):
    for name,m in block.named_modules():
        if not isinstance(m,torch.nn.Linear): continue
        w=m.weight.detach();cpu=w.cpu()
        qa,qb=RowQuantizer(w,4),RowQuantizer(cpu,4)
        a=qa.quantize(w).cpu()
        b=qb.quantize(cpu)
        bins=int((torch.round(w/qa.scale).clamp(-8,7).cpu()!=torch.round(cpu/qb.scale).clamp(-8,7)).sum())
        n=int((a!=b).sum())
        key=f'model.layers.{i}.{name}'
        if n: result[key]={'different_weights':n,'different_bins':bins,'max_difference':float((a-b).abs().max())}
        cpu_weights[key]=b
        m.weight.copy_(a.to(w.device))
tok=AutoTokenizer.from_pretrained(MODEL,revision=REVISION,local_files_only=True)
val=sequences(read_texts('data/wikitext2_20260926_s0/validation.jsonl'),tok,64,256)
gpu_ce=evaluate(model,val,'cuda')['ce']
for key,w in cpu_weights.items(): model.get_submodule(key).weight.copy_(w)
cpu_ce=evaluate(model,val,'cuda')['ce']
out={'modules_differing':len(result),'weights_differing':sum(v['different_weights'] for v in result.values()),
     'different_bins':sum(v['different_bins'] for v in result.values()),
     'gpu_prefix_ce':gpu_ce,'cpu_prefix_ce':cpu_ce,
     'modules':result,'fix':'Reconstruct RTN on CUDA, matching the original scans; keep CE tolerance unchanged.'}
Path('results/round4_prefix_device_diagnosis.json').write_text(json.dumps(out,indent=2),encoding='utf-8')
print(json.dumps({k:v for k,v in out.items() if k!='modules'},indent=2))
