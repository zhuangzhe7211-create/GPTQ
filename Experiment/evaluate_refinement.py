"""Freeze round-2 choices on validation, then evaluate held-out WikiText test."""
import argparse
import hashlib
import json
from pathlib import Path
import random
import re
import statistics
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from quant_core import RowQuantizer
from run_pilot import evaluate, read_texts, tensor_hash


def select_methods(metrics, families=('rescomp', 'gptaq', 'teacher', 'prefix', 'shared')):
    selected, scores = {}, {}
    for family in families:
        keys = [f'{family}_a{a:g}' for a in (.25, .5, 1.)]
        scores[family] = {k: statistics.mean(m[k]['ce'] for m in metrics) for k in keys}
        selected[family] = min(keys, key=scores[family].get)
    return selected, scores


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runs', nargs=3, default=[f'results/round2_s{s}' for s in range(3)])
    parser.add_argument('--out', default='results/round2_test')
    args = parser.parse_args()
    folders = [Path(p) for p in args.runs]
    out = Path(args.out)
    if out.exists():
        raise FileExistsError('Use a new output directory; test results and selection must not be overwritten')
    metrics = [json.loads((p / 'metrics.json').read_text()) for p in folders]
    configs = [json.loads((p / 'config.json').read_text()) for p in folders]
    for m, c in zip(metrics, configs):
        assert m['_summary']['all_one_regression'] == 'passed' and c['args']['refine']
        assert c['kind'] == 'pretrained_single_module_pilot'
        assert c['validation_window_hashes'] == configs[0]['validation_window_hashes']
        for key in ('model', 'revision', 'target', 'bits', 'dtype', 'seq_len', 'groups', 'rho', 'damp', 'blocksize'):
            assert c['args'][key] == configs[0]['args'][key], key
    selected, scores = select_methods(metrics)
    selected.update({'fixed_rescomp': 'rescomp_a0.25', 'fixed_teacher': 'teacher_a0.25'})
    frozen = {'rule': 'One alpha per family, chosen by mean validation CE across all three calibration seeds',
              'selected': selected, 'validation_ce': scores,
              'checkpoint_sha256': {str(p / f'{k}.pt'): sha(p / f'{k}.pt') for p in folders for k in set(selected.values())},
              'evaluation_source_sha256': sha(__file__), 'runs': [str(p.resolve()) for p in folders]}
    out.mkdir(parents=True)
    (out / 'selection.json').write_text(json.dumps(frozen, indent=2), encoding='utf-8')
    print('FROZEN BEFORE TEST:', json.dumps(selected), flush=True)

    # Test text is accessed only after parameter and checkpoint selection is saved.
    from datasets import load_dataset
    a = configs[0]['args']
    torch.set_num_threads(4)
    torch.manual_seed(0)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    device = a['device']
    if device.startswith('cuda') and not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable')
    tokenizer = AutoTokenizer.from_pretrained(a['model'], revision=a['revision'], local_files_only=True, trust_remote_code=False)
    excluded_text = set()
    excluded_windows = set()
    for c in configs:
        for key in ('calibration', 'validation'):
            excluded_text.update(read_texts(c['args'][key]))
        excluded_windows.update(c['calibration_window_hashes'])
        excluded_windows.update(c['validation_window_hashes'])
    dataset = load_dataset('Salesforce/wikitext', 'wikitext-2-raw-v1', split='test',
                           revision='b08601e04326c79dfdd32d625aee71d232d685c3')
    article_ids, article_id = [], -1
    for index, row in enumerate(dataset):
        if re.fullmatch(r'\s*= [^=\n]+ =\s*', row['text']):
            article_id = index
        article_ids.append(article_id)
    indices = list(range(len(dataset)))
    random.Random(20260926).shuffle(indices)
    batches, provenance = [], []
    seen_text, seen_windows = set(excluded_text), set(excluded_windows)
    for index in indices:
        text = dataset[index]['text']
        if text in seen_text or len(text.strip()) < 500:
            continue
        ids = tokenizer.encode(text, add_special_tokens=False)
        if len(ids) < a['seq_len']:
            continue
        batch = torch.tensor([ids[:a['seq_len']]], dtype=torch.long)
        digest = tensor_hash(batch)
        if digest in seen_windows:
            continue
        seen_text.add(text)
        seen_windows.add(digest)
        batches.append(batch)
        provenance.append({'source_row': index, 'article_heading_row': article_ids[index],
                           'text_sha256': hashlib.sha256(text.encode()).hexdigest(), 'window_sha256': digest})
        if len(batches) == 128:
            break
    if len(batches) != 128:
        raise ValueError('Not enough independent test rows')
    manifest = {'dataset': 'Salesforce/wikitext', 'revision': 'b08601e04326c79dfdd32d625aee71d232d685c3',
                'split': 'test', 'fingerprint': dataset._fingerprint, 'seed': 20260926,
                'policy': 'First fixed-length window per text row; exact text/window deduplication against calibration and validation',
                'windows': provenance}
    (out / 'test_manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    model = AutoModelForCausalLM.from_pretrained(a['model'], revision=a['revision'], local_files_only=True,
                trust_remote_code=False, dtype=getattr(torch, a['dtype']), attn_implementation='eager')
    model = model.to(device).eval().requires_grad_(False)
    model.config.use_cache = False
    results = {}

    def record(name):
        value = evaluate(model, batches, device)
        results[name] = value
        (out / 'metrics.json').write_text(json.dumps(results, indent=2, allow_nan=False), encoding='utf-8')
        print(name, value['ce'], value['ppl'], flush=True)
        return value

    started = time.perf_counter()
    record('floating_point')
    target_index = int(a['target'].split('.')[2])
    with torch.no_grad():
        for block in model.model.layers[:target_index]:
            for module in block.modules():
                if isinstance(module, torch.nn.Linear):
                    module.weight.copy_(RowQuantizer(module.weight, a['bits']).quantize(module.weight.float()))
    record('common_rtn_prefix')
    target = model.get_submodule(a['target'])
    for folder, config in zip(folders, configs):
        seed = config['args']['seed']
        cache = {}
        for family, checkpoint in selected.items():
            path = folder / f'{checkpoint}.pt'
            if sha(path) != frozen['checkpoint_sha256'][str(path)]:
                raise RuntimeError('Checkpoint changed after selection')
            if checkpoint not in cache:
                payload = torch.load(path, weights_only=True, map_location='cpu')
                assert payload['target'] == a['target']
                with torch.no_grad():
                    target.weight.copy_(payload['weight'])
                cache[checkpoint] = record(f's{seed}/{family}')
            else:
                results[f's{seed}/{family}'] = cache[checkpoint]
    results['_summary'] = {
        'families': {family: {metric: statistics.mean(results[f"s{c['args']['seed']}/{family}"][metric] for c in configs)
                              for metric in ('ce', 'ppl')} for family in selected},
        'total_seconds': time.perf_counter() - started,
        'peak_cuda_bytes': torch.cuda.max_memory_allocated() if device.startswith('cuda') else 0,
        'selection_frozen_before_test': True,
        'warning': 'Same test text across seeds. Single-module result; no further tuning on this test.'}
    (out / 'metrics.json').write_text(json.dumps(results, indent=2, allow_nan=False), encoding='utf-8')
    print('DONE:', out.resolve(), flush=True)


if __name__ == '__main__':
    main()
