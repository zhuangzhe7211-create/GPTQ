"""Readable research kernel + adapter to the pinned official ResComp core.

Fake quantization only: no packed checkpoint, no inference speed claim.
"""
import ast
import hashlib
import json
import logging
from pathlib import Path
import types

import torch


class RowQuantizer:
    """Fixed per-output-row signed grid, shared across all compared methods.

    Intentionally no clipping search, activation ordering or input group scales.
    """
    def __init__(self, weight, bits):
        if bits not in (2, 3, 4, 8):
            raise ValueError('bits must be 2, 3, 4, or 8')
        self.bits, self.sym = bits, True
        self.maxq = 2 ** (bits - 1) - 1
        self.scale = weight.float().abs().amax(dim=1, keepdim=True).clamp_min(1e-8) / self.maxq

    def ready(self):
        return True

    def quantize(self, weight):
        return torch.round(weight / self.scale).clamp(-self.maxq - 1, self.maxq) * self.scale


def make_weights(gradient, groups=4, rho=.5, clip=10., shuffle=False, seed=0):
    """USER EDIT POINT: [out,tokens] -> list of (row indices, token weights)."""
    if not 0 <= rho <= 1 or not 1 <= groups <= gradient.shape[0] or clip <= 0:
        raise ValueError('invalid sensitivity configuration')
    result = []
    generator = torch.Generator(device=gradient.device).manual_seed(seed)
    for rows in torch.tensor_split(torch.arange(gradient.shape[0], device=gradient.device), groups):
        score = gradient[rows].double().square().mean(0)
        if score.mean() <= 0:
            score = torch.ones_like(score)
        else:
            score = (score / score.mean()).clamp(max=clip)
            score = score / score.mean()
        score = (1 - rho) + rho * score
        if shuffle:
            score = score[torch.randperm(score.numel(), generator=generator, device=score.device)]
        result.append((rows, score.float()))
    return result


def statistics(xf, xq, score):
    """USER EDIT POINT: H and C must use exactly the same weights."""
    if xf.shape != xq.shape or score.shape != (xq.shape[1],):
        raise ValueError('unaligned inputs/weights')
    if not torch.isfinite(score).all() or (score < 0).any() or score.sum() <= 0:
        raise ValueError('invalid token weights')
    weighted = xq * score.unsqueeze(0)
    normalizer = xq.shape[1]
    h = weighted @ xq.T / normalizer
    c = ((xf - xq) * score.unsqueeze(0)) @ xq.T / normalizer
    return h, c


def reference_nd(weight, xf, xq, score, bits=4, damp=.01, alpha=.25, beta=.25):
    """Readable neuron-decomposition kernel (not official block implementation).

    beta=0 -> GPTAQ-style; alpha=beta=0 -> GPTQ-style.
    Uses lower Cholesky L of H^-1: P=triu(C L,1) L^T.
    """
    w0 = weight.float().clone()
    w = w0.clone()
    h, c = statistics(xf.float(), xq.float(), score)
    t = h + c
    h = h + torch.eye(h.shape[0], device=h.device) * (damp * h.diag().mean().clamp_min(1e-8))
    inv = torch.cholesky_inverse(torch.linalg.cholesky(h))
    lower = torch.linalg.cholesky(inv)
    p = torch.triu(c @ lower, diagonal=1) @ lower.T
    p2 = torch.triu(t @ lower, diagonal=1) @ lower.T
    quantizer = RowQuantizer(w0, bits)
    result = torch.empty_like(w)
    for k in range(w.shape[1]):
        wk = w[:, k].clone()
        qk = quantizer.quantize(wk[:, None]).flatten()
        result[:, k] = qk
        tail = lower[k:, k] / lower[k, k]
        w[:, k:] += (qk - wk)[:, None] * tail[None, :]
        w[:, k:] += alpha * wk[:, None] * p[k, k:][None, :]
        w[:, k:] += beta * (w0[:, k] - wk)[:, None] * p2[k, k:][None, :]
    return result


class _TorchProxy:
    """Only adapt unconditional CUDA timing sync for CPU smoke; no global patch."""
    def __init__(self):
        self.cuda = types.SimpleNamespace(synchronize=lambda: None)

    def __getattr__(self, name):
        return getattr(torch, name)


def official_class():
    root = Path(__file__).resolve().parent / 'vendor' / 'rescomp'
    manifest = json.loads((root / 'manifest.json').read_text(encoding='utf-8'))
    path = root / 'gptaq_utils_r.py'
    data = path.read_bytes()
    if hashlib.sha256(data).hexdigest() != manifest['files'][path.name]['sha256']:
        raise RuntimeError('Official source hash mismatch')
    # Load only the reviewed class, not the repository model/data pipeline imports.
    parsed = ast.parse(data.decode('utf-8'))
    node = next(n for n in parsed.body if isinstance(n, ast.ClassDef) and n.name == 'GPTAQ')
    namespace = {'torch': torch if torch.cuda.is_available() else _TorchProxy(),
                 'logging': logging}
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), 'exec'), namespace)
    return namespace['GPTAQ']


@torch.no_grad()
def official_rescomp(weight, xf, xq, score, bits=4, damp=.01, alpha=.25, blocksize=128):
    """Use official fasterquant; supply weighted sufficient statistics.

    beta/alpha2 is hardcoded .25 by the pinned upstream implementation.
    Fresh layer and fixed original grid on every call; no actorder/groupsize.
    """
    layer = torch.nn.Linear(weight.shape[1], weight.shape[0], bias=False,
                            device=weight.device, dtype=torch.float32)
    layer.weight.copy_(weight.float())
    core = official_class()(layer)
    core.quantizer = RowQuantizer(weight, bits)
    core.H, core.dXXT = statistics(xf.float(), xq.float(), score)
    core.fasterquant(blocksize=blocksize, percdamp=damp, groupsize=-1,
                     actorder=False, static_groups=False, alpha=alpha)
    return layer.weight.detach().clone()


def grouped_official(weight, xf, xq, gradient, groups, rho, bits, damp, alpha, blocksize, shuffle=False, seed=0):
    output = torch.empty_like(weight, dtype=torch.float32)
    for rows, score in make_weights(gradient, groups, rho, shuffle=shuffle, seed=seed):
        output[rows] = official_rescomp(weight[rows], xf, xq, score,
                                      bits, damp, alpha, blocksize)
    return output
