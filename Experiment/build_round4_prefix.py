"""Blockwise calibrated GPTQ prefix, fixed row grid, no act-order/group scales."""
import argparse
import json
from pathlib import Path
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from prepare_round3_holdout import MODEL, REVISION
from run_pilot import read_texts, sequences, tensor_hash, evaluate
from quant_core import RowQuantizer
from evaluate_refinement import sha


@torch.no_grad()
def gptq_h(weight, h, bits=4, damp=.01, blocksize=128):
    w = weight.float().clone()
    h = h.clone()
    h.diagonal().add_(damp * h.diag().mean().clamp_min(1e-8))
    upper = torch.linalg.cholesky(torch.cholesky_inverse(torch.linalg.cholesky(h)), upper=True)
    quantizer = RowQuantizer(weight, bits)
    q = torch.empty_like(w)
    for start in range(0, w.shape[1], blocksize):
        end = min(start + blocksize, w.shape[1])
        local = w[:, start:end].clone()
        errors = torch.zeros_like(local)
        for j in range(end-start):
            col = local[:, j].clone()
            value = quantizer.quantize(col[:, None]).flatten()
            q[:, start+j] = value
            error = (col-value) / upper[start+j, start+j]
            local[:, j:] -= error[:, None] * upper[start+j, start+j:end][None, :]
            errors[:, j] = error
        w[:, end:] -= errors @ upper[start:end, end:]
    if not torch.isfinite(q).all():
        raise RuntimeError('Nonfinite GPTQ prefix')
    return q


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--seed', type=int, required=True)
    a = p.parse_args()
    out = Path(f'results/round4_prefix_s{a.seed}')
    if out.exists():
        raise FileExistsError(out)
    out.mkdir()
    torch.set_num_threads(4)
    torch.manual_seed(a.seed)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    tok = AutoTokenizer.from_pretrained(MODEL, revision=REVISION, local_files_only=True)
    cal = sequences(read_texts(f'data/wikitext2_20260926_s{a.seed}/calibration.jsonl'), tok, 64, 256)
    val = sequences(read_texts('data/wikitext2_20260926_s0/validation.jsonl'), tok, 64, 256)
    model = AutoModelForCausalLM.from_pretrained(MODEL, revision=REVISION, local_files_only=True,
              dtype=torch.float32, attn_implementation='eager').cuda().eval().requires_grad_(False)
    model.config.use_cache = False
    hidden, call_kwargs = [], {}
    class Captured(Exception):
        pass
    def first_hook(module, args, kwargs):
        hidden.append(kwargs['hidden_states'].detach() if not args else args[0].detach())
        call_kwargs.update({k:v for k,v in kwargs.items() if k != 'hidden_states'})
        raise Captured()
    handle = model.model.layers[0].register_forward_pre_hook(first_hook, with_kwargs=True)
    with torch.no_grad():
        for ids in cal:
            try:
                model(input_ids=ids.cuda(), use_cache=False)
            except Captured:
                pass
    handle.remove()
    assert len(hidden) == 64
    # All windows have identical positions and length; kwargs contain no token-dependent state.
    assert call_kwargs.get('past_key_values') is None
    mapping = {'self_attn.q_proj':'self_attn.q_proj', 'self_attn.k_proj':'self_attn.q_proj',
               'self_attn.v_proj':'self_attn.q_proj', 'self_attn.o_proj':'self_attn.o_proj',
               'mlp.gate_proj':'mlp.up_proj', 'mlp.up_proj':'mlp.up_proj', 'mlp.down_proj':'mlp.down_proj'}
    start = time.perf_counter()
    weights = {}
    with torch.no_grad():
        for index, block in enumerate(model.model.layers[:12]):
            stats = {}
            handles = []
            for name in sorted(set(mapping.values())):
                def collect(module, args, output, key=name):
                    x = args[0].float().reshape(-1, args[0].shape[-1]).T
                    if key not in stats:
                        stats[key] = torch.zeros((x.shape[0], x.shape[0]), device=x.device)
                    stats[key].addmm_(x, x.T, alpha=1/(64*256))
                handles.append(block.get_submodule(name).register_forward_hook(collect))
            for x in hidden:
                block(x, **call_kwargs)
            for h in handles:
                h.remove()
            for name, key in mapping.items():
                module = block.get_submodule(name)
                module.weight.copy_(gptq_h(module.weight, stats[key]))
                weights[f'model.layers.{index}.{name}'] = module.weight.detach().cpu().clone()
            hidden = [block(x, **call_kwargs).detach() for x in hidden]
            del stats
            torch.cuda.synchronize()
            print('PREFIX', a.seed, 'block', index, 'seconds', round(time.perf_counter()-start,2), flush=True)
    elapsed = time.perf_counter()-start
    metrics = evaluate(model, val, 'cuda')
    torch.save(weights, out/'weights.pt')
    manifest = {'seed':a.seed, 'model':MODEL, 'revision':REVISION, 'bits':4, 'blocks':12,
                'kind':'blockwise GPTQ; H collected before within-block quantization, propagated between blocks',
                'damp':.01, 'blocksize':128, 'calibration_hashes':list(map(tensor_hash,cal)),
                'validation_hashes':list(map(tensor_hash,val)), 'validation':metrics,
                'build_seconds':elapsed, 'weight_sha256':sha(out/'weights.pt'), 'source_sha256':sha(__file__)}
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    print('DONE',out,'PPL',metrics['ppl'],flush=True)


if __name__ == '__main__':
    main()
