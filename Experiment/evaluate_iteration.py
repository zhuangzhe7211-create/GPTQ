"""Freeze all layer-specific round-3 choices before evaluating new held-out articles."""
import json
from pathlib import Path
import statistics
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from evaluate_refinement import select_methods, sha
from quant_core import RowQuantizer
from run_pilot import evaluate, tensor_hash, ITERATION_FAMILIES, sequences, read_texts

LAYERS = (6, 12, 18)
FAMILIES = ('rescomp', 'gptaq', *ITERATION_FAMILIES)


def main():
    out = Path('results/round3_holdout')
    if out.exists():
        raise FileExistsError('Do not overwrite frozen selection or held-out results')
    frozen = {'selected': {}, 'validation_ce': {}, 'checkpoint_sha256': {}, 'config_sha256': {},
              'evaluation_source_sha256': sha(__file__),
              'rule': 'For each layer/family choose one alpha by mean validation CE across three calibration seeds'}
    sources = json.loads(Path('results/round3_source/source_manifest.json').read_text())
    allowed_core_versions = {sources['quant_core.py'], sources['quant_core_null_endpoint.py']}
    frozen['core_compatibility'] = {'allowed_sha256': sorted(allowed_core_versions),
        'reason': 'Only rho=0 endpoint dispatch changed; all scanned weighted candidates have rho>0. Positive-rho outputs regression-checked.'}
    config_by_layer = {}
    common = None
    for layer in LAYERS:
        folders = [Path(f'results/round3_l{layer}_s{s}') for s in range(3)]
        metrics = [json.loads((p/'metrics.json').read_text()) for p in folders]
        configs = [json.loads((p/'config.json').read_text()) for p in folders]
        for seed, (p, c, m) in enumerate(zip(folders, configs, metrics)):
            assert m['_summary']['all_one_regression'] == 'passed'
            assert c['kind'] == 'pretrained_single_module_pilot' and c['args']['iterate']
            assert c['args']['target'] == f'model.layers.{layer}.self_attn.o_proj'
            assert c['args']['seed'] == seed and c['args']['samples'] == 64
            assert c['args']['seq_len'] == 256
            if common is None:
                common = c
            for k in ('model', 'revision', 'dtype', 'bits', 'seq_len', 'damp', 'blocksize'):
                assert c['args'][k] == common['args'][k]
            assert c['validation_window_hashes'] == common['validation_window_hashes']
            assert c['source_sha256']['run_pilot.py'] == sources['run_pilot.py']
            assert c['source_sha256']['quant_core.py'] in allowed_core_versions
            frozen['config_sha256'][str(p/'config.json')] = sha(p/'config.json')
        selected, scores = select_methods(metrics, FAMILIES)
        frozen['selected'][str(layer)] = selected
        frozen['validation_ce'][str(layer)] = scores
        config_by_layer[layer] = configs
        for p in folders:
            for checkpoint in selected.values():
                path = p/f'{checkpoint}.pt'
                frozen['checkpoint_sha256'][str(path)] = sha(path)
    heldout_path = Path('data/round3_holdout')
    frozen['holdout_manifest_sha256'] = sha(heldout_path/'manifest.json')
    out.mkdir()
    (out/'selection.json').write_text(json.dumps(frozen, indent=2), encoding='utf-8')
    print('ALL LAYERS FROZEN:', json.dumps(frozen['selected']), flush=True)
    manifest = json.loads((heldout_path/'manifest.json').read_text())
    assert sha(heldout_path/'tokens.pt') == manifest['tokens_sha256']
    batches = torch.load(heldout_path/'tokens.pt', weights_only=True)
    hashes = list(map(tensor_hash, batches))
    assert hashes == [x['window_sha256'] for x in manifest['windows']]
    assert len(set(hashes)) == 128
    assert len({x['article_heading_row'] for x in manifest['windows']}) == 128
    for configs in config_by_layer.values():
        for c in configs:
            assert not set(hashes).intersection(c['calibration_window_hashes'] + c['validation_window_hashes'])
    (out/'manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    torch.set_num_threads(4)
    torch.manual_seed(0)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    a = common['args']
    device = a['device']
    if device.startswith('cuda') and not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable')
    tokenizer = AutoTokenizer.from_pretrained(a['model'], revision=a['revision'], local_files_only=True)
    validation_batches = sequences(read_texts(a['validation']), tokenizer, a['eval_samples'], a['seq_len'])
    assert list(map(tensor_hash, validation_batches)) == common['validation_window_hashes']
    model = AutoModelForCausalLM.from_pretrained(a['model'], revision=a['revision'], local_files_only=True,
                trust_remote_code=False, dtype=getattr(torch, a['dtype']), attn_implementation='eager')
    model = model.to(device).eval().requires_grad_(False)
    model.config.use_cache = False
    results = {}

    def record(name):
        value = evaluate(model, batches, device)
        results[name] = value
        (out/'metrics.json').write_text(json.dumps(results, indent=2, allow_nan=False), encoding='utf-8')
        print(name, 'CE', value['ce'], 'PPL', value['ppl'], flush=True)

    start = time.perf_counter()
    record('floating_point')
    # Advance the identical RTN prefix monotonically, restoring each target first.
    prefix_end = 0
    for layer in LAYERS:
        with torch.no_grad():
            for block in model.model.layers[prefix_end:layer]:
                for module in block.modules():
                    if isinstance(module, torch.nn.Linear):
                        module.weight.copy_(RowQuantizer(module.weight, a['bits']).quantize(module.weight.float()))
        prefix_end = layer
        expected = json.loads(Path(f'results/round3_l{layer}_s0/metrics.json').read_text())['common_rtn_prefix']['ce']
        observed = evaluate(model, validation_batches, device)['ce']
        assert abs(expected-observed) < 1e-6, 'Incremental RTN prefix differs from the original isolated scan'
        record(f'l{layer}/common_rtn_prefix')
        target = model.get_submodule(f'model.layers.{layer}.self_attn.o_proj')
        original = target.weight.detach().clone()
        for seed in range(3):
            for family, checkpoint in frozen['selected'][str(layer)].items():
                path = Path(f'results/round3_l{layer}_s{seed}')/f'{checkpoint}.pt'
                assert sha(path) == frozen['checkpoint_sha256'][str(path)]
                payload = torch.load(path, weights_only=True, map_location='cpu')
                assert payload['target'] == f'model.layers.{layer}.self_attn.o_proj'
                with torch.no_grad():
                    target.weight.copy_(payload['weight'])
                record(f'l{layer}/s{seed}/{family}')
        with torch.no_grad():
            target.weight.copy_(original)
    results['_summary'] = {
        'layers': {str(layer): {family: {metric: statistics.mean(results[f'l{layer}/s{s}/{family}'][metric] for s in range(3))
                                       for metric in ('ce', 'ppl')} for family in FAMILIES} for layer in LAYERS},
        'evaluation_seconds': time.perf_counter()-start,
        'peak_cuda_bytes': torch.cuda.max_memory_allocated() if device.startswith('cuda') else 0,
        'all_selections_frozen_before_evaluation': True,
        'incremental_prefix_regression': 'passed for every layer on identical validation windows',
        'warning': 'New unused train articles, not official test. Multiple single-module contexts, not full-model quantization.'}
    (out/'metrics.json').write_text(json.dumps(results, indent=2, allow_nan=False), encoding='utf-8')
    print('DONE:', out.resolve(), flush=True)


if __name__ == '__main__':
    main()
