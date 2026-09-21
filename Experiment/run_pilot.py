"""Single-module pilot with frozen common RTN prefix; see TONIGHT.md.

Default --smoke is a random tiny Llama, NOT pretrained quality evidence.
Real experiments require --model plus independent calibration/validation JSONL.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import platform
import time

import torch
import torch.nn.functional as F
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer, LlamaConfig, LlamaForCausalLM, Qwen2Config, Qwen2ForCausalLM

from quant_core import RowQuantizer, grouped_official, make_weights, official_rescomp, reference_nd


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--smoke', action='store_true')
    p.add_argument('--smoke-arch', choices=['llama', 'qwen2'], default='llama')
    p.add_argument('--model')
    p.add_argument('--revision', default='main')
    p.add_argument('--allow-download', action='store_true')
    p.add_argument('--calibration')
    p.add_argument('--validation')
    p.add_argument('--device', default='cpu')
    p.add_argument('--dtype', choices=['float32', 'float16', 'bfloat16'], default='float32')
    p.add_argument('--target', default='model.layers.1.self_attn.o_proj')
    p.add_argument('--samples', type=int, default=8)
    p.add_argument('--eval-samples', type=int, default=8)
    p.add_argument('--seq-len', type=int, default=128)
    p.add_argument('--bits', type=int, default=4)
    p.add_argument('--groups', type=int, default=4)
    p.add_argument('--rho', type=float, default=.5)
    p.add_argument('--alpha', type=float, default=.25)
    p.add_argument('--damp', type=float, default=.01)
    p.add_argument('--blocksize', type=int, default=128)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--out', default='results/pilot')
    return p.parse_args()


def read_texts(path):
    entries = [json.loads(line)['text'] for line in Path(path).read_text(encoding='utf-8').splitlines() if line.strip()]
    if not entries or any(not isinstance(x, str) or not x.strip() for x in entries):
        raise ValueError('JSONL must contain nonempty text strings')
    return entries


def sequences(texts, tokenizer, count, length):
    batches = []
    for text in texts:
        ids = tokenizer.encode(text, add_special_tokens=False)
        # No padding. Keep contiguous fixed-length windows inside one document.
        for start in range(0, len(ids) - length + 1, length):
            batches.append(torch.tensor([ids[start:start + length]], dtype=torch.long))
            if len(batches) == count:
                return batches
    raise ValueError(f'Only {len(batches)} windows; need {count}. Supply more/longer text or lower sample count.')


def loss_for(model, ids):
    logits = model(input_ids=ids, use_cache=False).logits
    return F.cross_entropy(logits[:, :-1].float().reshape(-1, logits.shape[-1]),
                           ids[:, 1:].reshape(-1), reduction='mean')


@torch.no_grad()
def evaluate(model, batches, device):
    losses = [float(loss_for(model, ids.to(device))) for ids in batches]
    if not all(math.isfinite(v) for v in losses):
        raise RuntimeError('Non-finite evaluation loss')
    # Equal-length windows -> arithmetic mean equals token-weighted mean.
    ce = sum(losses) / len(losses)
    return {'ce': ce, 'ppl': math.exp(ce) if ce < 700 else None,
            'sequence_ce': losses, 'scored_tokens': len(batches) * (batches[0].shape[1] - 1)}


def capture(model, module, batches, device, gradients):
    xs, zs, gs = [], [], []
    for ids in batches:
        cache = {}

        def hook(_module, inputs, output):
            cache['x'] = inputs[0].detach().float().cpu().reshape(-1, inputs[0].shape[-1]).T
            cache['z'] = output.detach().requires_grad_(True) if gradients else output.detach()
            return cache['z']

        handle = module.register_forward_hook(hook)
        try:
            with torch.set_grad_enabled(gradients):
                loss = loss_for(model, ids.to(device))
                if gradients:
                    g, = torch.autograd.grad(loss, cache['z'])
                    gs.append(g.detach().float().cpu().reshape(-1, g.shape[-1]).T)
            xs.append(cache['x'])
            z = cache['z'].detach().float().cpu()
            zs.append(z.reshape(-1, z.shape[-1]).T)
        finally:
            handle.remove()
    return torch.cat(xs, dim=1), torch.cat(zs, dim=1), torch.cat(gs, dim=1) if gradients else None


def tensor_hash(tensor):
    # CPU token IDs only; avoids numpy dependency and saves exact reproducibility IDs.
    return hashlib.sha256(json.dumps(tensor.tolist()).encode()).hexdigest()


def main():
    args = parse_args()
    if args.samples < 1 or args.eval_samples < 1 or args.seq_len < 2 or args.damp <= 0:
        raise ValueError('positive sample counts/damping and seq_len >=2 required')
    if args.device.startswith('cuda') and not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable in this Python. Check CUDA-enabled PyTorch; do not silently run real model on CPU.')
    if args.smoke == bool(args.model):
        raise ValueError('Choose exactly one of --smoke or --model')
    out = Path(args.out)
    if (out / 'metrics.json').exists():
        raise FileExistsError('Use a new --out directory; do not overwrite results')
    out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    started = time.perf_counter()
    if args.smoke:
        config_class = LlamaConfig if args.smoke_arch == 'llama' else Qwen2Config
        model_class = LlamaForCausalLM if args.smoke_arch == 'llama' else Qwen2ForCausalLM
        config = config_class(vocab_size=128, hidden_size=32, intermediate_size=64,
                             num_hidden_layers=3, num_attention_heads=4, num_key_value_heads=2,
                             max_position_embeddings=max(256, args.seq_len))
        config._attn_implementation = 'eager'
        model = model_class(config)
        generator = torch.Generator().manual_seed(args.seed + 1000)
        cal = [torch.randint(3, 128, (1, args.seq_len), generator=generator) for _ in range(args.samples)]
        val = [torch.randint(3, 128, (1, args.seq_len), generator=generator) for _ in range(args.eval_samples)]
    else:
        if not args.calibration or not args.validation:
            raise ValueError('Real model requires --calibration and --validation JSONL')
        texts_cal, texts_val = read_texts(args.calibration), read_texts(args.validation)
        if set(texts_cal) & set(texts_val):
            raise ValueError('Calibration and validation documents overlap')
        tokenizer = AutoTokenizer.from_pretrained(args.model, revision=args.revision,
                                                  local_files_only=not args.allow_download, trust_remote_code=False)
        cal = sequences(texts_cal, tokenizer, args.samples, args.seq_len)
        val = sequences(texts_val, tokenizer, args.eval_samples, args.seq_len)
        model = AutoModelForCausalLM.from_pretrained(args.model, revision=args.revision,
                    local_files_only=not args.allow_download, trust_remote_code=False,
                    dtype=getattr(torch, args.dtype), attn_implementation='eager')
    if set(map(tensor_hash, cal)) & set(map(tensor_hash, val)):
        raise ValueError('Calibration and validation token windows overlap')
    model = model.to(device=args.device, dtype=getattr(torch, args.dtype)).eval()
    model.requires_grad_(False)
    model.config.use_cache = False
    target = model.get_submodule(args.target)
    if not isinstance(target, torch.nn.Linear) or not args.target.startswith('model.layers.'):
        raise ValueError('First pilot supports model.layers.N.* nn.Linear targets only')
    target_index = int(args.target.split('.')[2])
    if target_index < 1:
        raise ValueError('Choose target block >=1 to create a nonempty quantized prefix')
    original = target.weight.detach().float().clone()
    metadata = {'args': vars(args), 'python': platform.python_version(), 'torch': torch.__version__,
                'transformers': transformers.__version__, 'model_commit': getattr(model.config, '_commit_hash', None),
                'calibration_window_hashes': list(map(tensor_hash, cal)),
                'validation_window_hashes': list(map(tensor_hash, val)),
                'kind': 'random_model_smoke' if args.smoke else 'pretrained_single_module_pilot',
                'prefix': 'RTN all Linear modules in complete blocks before target block; target block otherwise FP',
                'scope': 'official ResComp module kernel + fixed row quantizer, not official full-model benchmark'}
    source_root = Path(__file__).resolve().parent
    metadata['source_sha256'] = {name: hashlib.sha256((source_root / name).read_bytes()).hexdigest()
                                 for name in ['run_pilot.py', 'quant_core.py']}
    metadata['official_core'] = json.loads((source_root / 'vendor/rescomp/manifest.json').read_text(encoding='utf-8'))
    (out / 'config.json').write_text(json.dumps(metadata, indent=2), encoding='utf-8')
    results = {}

    def save_result(name, value):
        results[name] = value
        (out / 'metrics.json').write_text(json.dumps(results, indent=2, allow_nan=False), encoding='utf-8')
        print(name, json.dumps({k: v for k, v in value.items() if k != 'sequence_ce'}), flush=True)

    save_result('floating_point', evaluate(model, val, args.device))
    print('Capturing teacher gradients on calibration only...', flush=True)
    xf, _, grad = capture(model, target, cal, args.device, gradients=True)
    if not torch.isfinite(grad).all() or grad.square().sum() == 0:
        raise RuntimeError('Teacher gradient is zero/nonfinite')
    # Validation reconstruction target without validation gradient fitting.
    _, y_val, _ = capture(model, target, val, args.device, gradients=False)
    prefix_count = 0
    with torch.no_grad():
        for index, block in enumerate(model.model.layers):
            if index >= target_index:
                break
            for module in block.modules():
                if isinstance(module, torch.nn.Linear):
                    module.weight.copy_(RowQuantizer(module.weight, args.bits).quantize(module.weight.float()))
                    prefix_count += 1
    save_result('common_rtn_prefix', evaluate(model, val, args.device))
    xq, _, _ = capture(model, target, cal, args.device, gradients=False)
    xq_val, _, _ = capture(model, target, val, args.device, gradients=False)
    y0 = original.cpu() @ xf
    bias = target.bias.detach().float().cpu()[:, None] if target.bias is not None else 0.
    device = args.device
    xf, xq, grad = xf.to(device), xq.to(device), grad.to(device)
    one = torch.ones(xq.shape[1], device=device)
    stats = {'prefix_linear_count': prefix_count, 'relative_input_error': float((xf-xq).norm()/xf.norm().clamp_min(1e-12)),
             'gradient_abs_quantiles': torch.quantile(grad.abs().flatten(), torch.tensor([0., .5, .9, .99, 1.], device=device)).tolist()}
    (out / 'diagnostics.json').write_text(json.dumps(stats, indent=2), encoding='utf-8')
    common = (args.bits, args.damp, args.alpha, args.blocksize)
    print('Checking all-one group regression...', flush=True)
    baseline_tick = time.perf_counter()
    baseline = official_rescomp(original, xf, xq, one, *common)
    if device.startswith('cuda'):
        torch.cuda.synchronize()
    baseline_seconds = time.perf_counter() - baseline_tick
    allone = grouped_official(original, xf, xq, grad, args.groups, 0., *common)
    torch.testing.assert_close(allone, baseline, rtol=1e-5, atol=1e-6)
    candidates = {
        'rtn_target': lambda: RowQuantizer(original, args.bits).quantize(original),
        'gptaq_nd_reference': lambda: reference_nd(original, xf, xq, one, args.bits, args.damp, args.alpha, 0.),
        'rescomp_official_core': lambda: baseline,
        'task_weighted_rescomp': lambda: grouped_official(original, xf, xq, grad, args.groups, args.rho, *common),
        'shuffled_weight_rescomp': lambda: grouped_official(original, xf, xq, grad, args.groups, args.rho, *common, shuffle=True, seed=args.seed),
    }
    for name, build in candidates.items():
        tick = time.perf_counter()
        with torch.no_grad():
            candidate = build()
            target.weight.copy_(candidate)
        if device.startswith('cuda'):
            torch.cuda.synchronize()
        elapsed = baseline_seconds if name == 'rescomp_official_core' else time.perf_counter() - tick
        # Evaluate actual stored weight after dtype cast, not only FP32 candidate.
        stored = target.weight.detach().float().cpu()
        value = evaluate(model, val, device)
        error = stored @ xq.cpu() - y0
        weighted = 0.
        for rows, score in make_weights(grad.cpu(), args.groups, args.rho):
            weighted += float((error[rows].square() * score).sum())
        value.update({'calibration_mse': float(error.square().mean()),
                      'calibration_task_mse': weighted/error.numel(),
                      'validation_mse': float((stored @ xq_val + bias - y_val).square().mean()),
                      'quantization_seconds': elapsed,
                      'delta_ce_vs_rescomp': None})
        save_result(name, value)
        torch.save({'weight': stored, 'target': args.target}, out / f'{name}.pt')
    base_ce = results['rescomp_official_core']['ce']
    for name in candidates:
        results[name]['delta_ce_vs_rescomp'] = results[name]['ce'] - base_ce
    results['_summary'] = {'all_one_regression': 'passed', 'total_seconds': time.perf_counter()-started,
                           'peak_cuda_bytes': torch.cuda.max_memory_allocated() if device.startswith('cuda') else 0,
                           'quality_evidence': not args.smoke,
                           'warning': 'Single-module validation pilot; no held-out test or full-model superiority claim.'}
    (out / 'metrics.json').write_text(json.dumps(results, indent=2, allow_nan=False), encoding='utf-8')
    print('DONE:', out.resolve(), flush=True)


if __name__ == '__main__':
    main()
