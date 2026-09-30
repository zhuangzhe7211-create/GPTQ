"""Freeze round-4 development choices, then open previously unused article holdout."""
import json
from pathlib import Path
import statistics
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from evaluate_refinement import sha
from prepare_round3_holdout import MODEL, REVISION
from quant_core import RowQuantizer
from run_pilot import evaluate, read_texts, sequences, tensor_hash

FAMILIES = ('rescomp', 'gptaq', 'weak1', 'rms1')
TARGETS = {'attn':'model.layers.12.self_attn.o_proj', 'mlp':'model.layers.12.mlp.up_proj'}


def main():
    out = Path('results/round4_holdout')
    if out.exists():
        raise FileExistsError(out)
    frozen = {'selected':{}, 'validation_ce':{}, 'checkpoint_sha256':{}, 'config_sha256':{},
              'prefix_sha256':{}, 'evaluation_source_sha256':sha(__file__),
              'rule':'Choose one alpha per prefix/module/family by three-seed mean development CE'}
    configs = {}
    sources = json.loads(Path('results/round4_source/source_manifest.json').read_text())
    allowed_pilot = {sources['run_pilot.py'], sources['run_pilot_large_diagnostics.py']}
    frozen['pilot_compatibility'] = {'allowed_sha256':sorted(allowed_pilot),
        'reason':'Only gradient quantile diagnostics changed to deterministic subsampling; fitting uses all gradients.'}
    common_validation = None
    for prefix in ('rtn','gptq'):
        for module, target in TARGETS.items():
            context = f'{prefix}_{module}'
            runs = []
            for seed in range(3):
                path = Path(f'results/round4_{context}_s{seed}')
                config = json.loads((path/'config.json').read_text())
                metrics = json.loads((path/'metrics.json').read_text())
                assert metrics['_summary']['all_one_regression'] == 'passed'
                a = config['args']
                assert a['round4'] and a['target'] == target and a['seed'] == seed
                assert a['samples'] == a['eval_samples'] == 64 and a['seq_len'] == 256
                assert a['model'] == MODEL and a['revision'] == REVISION and a['dtype'] == 'float32'
                assert a['bits'] == 4 and a['damp'] == .01 and a['blocksize'] == 128
                assert bool(a['prefix_checkpoint']) == (prefix == 'gptq')
                for name, digest in config['source_sha256'].items():
                    assert digest in allowed_pilot if name == 'run_pilot.py' else sha(name) == digest, f'Source changed: {name}'
                if common_validation is None:
                    common_validation = config['validation_window_hashes']
                assert common_validation == config['validation_window_hashes']
                configs[(context,seed)] = config
                frozen['config_sha256'][str(path/'config.json')] = sha(path/'config.json')
                runs.append(metrics)
            scores = {f'{f}_a{a:g}':statistics.mean(r[f'{f}_a{a:g}']['ce'] for r in runs)
                      for f in FAMILIES for a in (.25,.5,1.,1.5)}
            selected = {f:min((f'{f}_a{a:g}' for a in (.25,.5,1.,1.5)), key=scores.get) for f in FAMILIES}
            selected.update({'rtn_target':'rtn_target', 'gptq_reference':'gptq_reference'})
            frozen['selected'][context] = selected
            frozen['validation_ce'][context] = scores
            for seed in range(3):
                for checkpoint in selected.values():
                    path = Path(f'results/round4_{context}_s{seed}/{checkpoint}.pt')
                    frozen['checkpoint_sha256'][str(path)] = sha(path)
    for seed in range(3):
        path = Path(f'results/round4_prefix_s{seed}')
        manifest = json.loads((path/'manifest.json').read_text())
        assert manifest['weight_sha256'] == sha(path/'weights.pt')
        assert manifest['calibration_hashes'] == configs[('gptq_attn',seed)]['calibration_window_hashes']
        frozen['prefix_sha256'][str(path/'weights.pt')] = manifest['weight_sha256']
    holdout = Path('data/round4_holdout')
    frozen['holdout_manifest_sha256'] = sha(holdout/'manifest.json')
    prior = Path('results/round4_failed_holdout_prefix_device/selection.json')
    if prior.exists():
        before = json.loads(prior.read_text())
        for key in ('selected','validation_ce','checkpoint_sha256','config_sha256','prefix_sha256','holdout_manifest_sha256'):
            assert frozen[key] == before[key], 'Previously frozen choices must not change after evaluator repair'
        frozen['prior_frozen_selection_sha256'] = sha(prior)
        frozen['repair'] = 'Reconstruct RTN prefix on CUDA as in original scans; no selection changes or tolerance relaxation.'
    out.mkdir()
    (out/'selection.json').write_text(json.dumps(frozen,indent=2),encoding='utf-8')
    print('ALL CHOICES FROZEN',json.dumps(frozen['selected']),flush=True)
    manifest = json.loads((holdout/'manifest.json').read_text())
    assert sha(holdout/'tokens.pt') == manifest['tokens_sha256']
    batches = torch.load(holdout/'tokens.pt',weights_only=True)
    hashes = list(map(tensor_hash,batches))
    articles = {r['article_heading_row'] for r in manifest['windows']}
    assert hashes == [r['window_sha256'] for r in manifest['windows']]
    assert len(articles) == len(set(hashes)) == 128
    assert not articles.intersection(manifest['excluded_calibration_article_ids']+manifest['excluded_round3_article_ids'])
    for config in configs.values():
        assert not set(hashes).intersection(config['calibration_window_hashes']+config['validation_window_hashes'])
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    torch.set_num_threads(4)
    torch.manual_seed(0)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    tok = AutoTokenizer.from_pretrained(MODEL,revision=REVISION,local_files_only=True)
    val = sequences(read_texts('data/wikitext2_20260926_s0/validation.jsonl'),tok,64,256)
    assert list(map(tensor_hash,val)) == common_validation
    model = AutoModelForCausalLM.from_pretrained(MODEL,revision=REVISION,local_files_only=True,
                dtype=torch.float32,attn_implementation='eager').cuda().eval().requires_grad_(False)
    model.config.use_cache = False
    original = {f'model.layers.{i}.{name}':m.weight.detach().cpu().clone()
                for i,block in enumerate(model.model.layers[:12]) for name,m in block.named_modules()
                if isinstance(m,torch.nn.Linear)}
    results = {}
    def record(name):
        results[name] = evaluate(model,batches,'cuda')
        (out/'metrics.json').write_text(json.dumps(results,indent=2,allow_nan=False),encoding='utf-8')
        print(name,results[name]['ppl'],flush=True)
    start = time.perf_counter()
    record('floating_point')
    for prefix in ('rtn','gptq'):
        for seed in range(3):
            path = Path(f'results/round4_prefix_s{seed}/weights.pt')
            if prefix == 'gptq':
                assert sha(path) == frozen['prefix_sha256'][str(path)]
                weights = torch.load(path,weights_only=True)
            else:
                weights = original
            with torch.no_grad():
                for name,w in weights.items():
                    w = w.to(model.get_submodule(name).weight.device)
                    value = w if prefix == 'gptq' else RowQuantizer(w,4).quantize(w)
                    model.get_submodule(name).weight.copy_(value)
            del weights
            observed = evaluate(model,val,'cuda')['ce']
            for module in TARGETS:
                expected = json.loads(Path(f'results/round4_{prefix}_{module}_s{seed}/metrics.json').read_text())['common_rtn_prefix']['ce']
                assert abs(observed-expected) < 1e-6, 'Prefix reproduction mismatch'
            record(f'{prefix}/s{seed}/prefix')
            for module,target_name in TARGETS.items():
                context = f'{prefix}_{module}'
                target = model.get_submodule(target_name)
                target_original = target.weight.detach().clone()
                for family,checkpoint in frozen['selected'][context].items():
                    path = Path(f'results/round4_{context}_s{seed}/{checkpoint}.pt')
                    assert sha(path) == frozen['checkpoint_sha256'][str(path)]
                    payload = torch.load(path,weights_only=True)
                    assert payload['target'] == target_name
                    with torch.no_grad():
                        target.weight.copy_(payload['weight'])
                    record(f'{context}/s{seed}/{family}')
                with torch.no_grad():
                    target.weight.copy_(target_original)
    results['_summary'] = {'all_selections_frozen_before_evaluation':True,
                           'prefix_validation_regression':'passed in all 12 scans',
                           'evaluation_seconds':time.perf_counter()-start,
                           'warning':'Independent single-module contexts, not a full W4 quantized model'}
    (out/'metrics.json').write_text(json.dumps(results,indent=2,allow_nan=False),encoding='utf-8')
    print('DONE',out,flush=True)


if __name__ == '__main__':
    main()
