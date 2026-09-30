"""CPU-only audit of completed artifacts; does not retune or reevaluate holdout."""
import hashlib,json,math,statistics
from pathlib import Path
import torch
from safetensors import safe_open

base=Path(__file__).resolve().parent.parent
torch.set_num_threads(4)
def read(p):return json.loads((base/p).read_text(encoding='utf-8'))
def sha(p):return hashlib.sha256((base/p).read_bytes()).hexdigest()
frozen=read('results/round5_holdout/selection.json')
held=read('results/round5_holdout/metrics.json')
assert held['_summary']['complete'] and len(held)==68
for field in ('checkpoint_sha256','config_sha256'):
    for name,digest in frozen[field].items():assert sha(name)==digest,name
assert sha('evaluate_round5.py')==frozen['source_sha256']
assert sha('gpu_safety.py')==frozen['safety_source_sha256']
assert sha('data/round5_holdout/manifest.json')==frozen['holdout_manifest_sha256']
target='model.layers.12.mlp.up_proj'
model=base/'.hf_cache/hub/models--Qwen--Qwen2.5-0.5B/snapshots/060db6499f32faf8b98477b0a26969ef7d8b9987/model.safetensors'
with safe_open(model,framework='pt',device='cpu') as f:w0=f.get_tensor(target+'.weight').float()
analytic_scale=w0.abs().amax(1,keepdim=True).clamp_min(1e-8)/7
# CUDA scalar division may differ from CPU division by one ULP. Recover the
# actual frozen scale from code +/-1 in the original GPTQ checkpoint, then
# require exact membership on that same scale for every candidate.
reference=torch.load(base/'results/round5_rtn_s0/gptq_reference.pt',weights_only=True,map_location='cpu')['weight']
ones=(reference/analytic_scale).round().abs()==1
assert ones.any(1).all()
scale=reference.gather(1,ones.int().argmax(1,keepdim=True)).abs()
torch.testing.assert_close(scale,analytic_scale,rtol=1.2e-7,atol=0)
count=0;details=[]
for prefix,selected in frozen['selected'].items():
    runs=[]
    for seed in range(3):
        merged={}
        for stage,n in [('round5',35),('round5_block',6)]:
            folder=Path(f'results/{stage}_{prefix}_s{seed}')
            metrics=read(folder/'metrics.json');config=read(folder/'config.json')
            assert metrics['_summary']['complete'] and metrics['_summary']['candidate_count']==n
            for name,digest in config['source_sha256'].items():assert sha(name)==digest
            candidates={k:v for k,v in metrics.items() if k not in ('_summary','common_prefix')}
            assert len(candidates)==n
            for key,v in candidates.items():
                weight=torch.load(base/folder/(key+'.pt'),weights_only=True,map_location='cpu')['weight']
                codes=(weight/scale).round()
                assert torch.isfinite(weight).all() and codes.min()>=-8 and codes.max()<=7
                torch.testing.assert_close(weight,codes*scale,rtol=0,atol=0)
                assert len(v['sequence_ce'])==64
                assert abs(statistics.mean(v['sequence_ce'])-v['ce'])<1e-12
                count+=1
            merged.update(candidates)
        runs.append(merged)
    scores={k:statistics.mean(r[k]['ce'] for r in runs) for k in runs[0]}
    assert scores==frozen['validation_ce'][prefix]
    for family,key in selected.items():
        if family in ('rtn_target','gptq_reference'):continue
        choices=[k for k in scores if not k.startswith(('product_','mlp_','block_'))] if family=='control' else [k for k in scores if k.startswith(family+'_')]
        assert key==min(choices,key=scores.get)
    for seed in range(3):
        a=held[f'{prefix}/s{seed}/block'];b=held[f'{prefix}/s{seed}/control']
        details.append({'prefix':prefix,'seed':seed,'block_ppl':a['ppl'],'control_ppl':b['ppl'],'delta_ce':a['ce']-b['ce']})
for key,v in held.items():
    if key=='_summary':continue
    assert len(v['sequence_ce'])==73 and v['scored_tokens']==18615
    assert abs(statistics.mean(v['sequence_ce'])-v['ce'])<1e-12
    assert abs(math.exp(v['ce'])-v['ppl'])<1e-12
m=read('data/round5_holdout/manifest.json')
articles={w['article_heading_row'] for w in m['windows']}
assert len(articles)==73 and not articles.intersection(m['excluded_calibration_article_ids']+m['excluded_round3_and_round4_article_ids'])
result={'complete':True,'cpu_only':True,'candidate_checkpoints_on_original_fixed_grid':count,
        'grid_check':'Exact reconstruction using original GPTQ code-1 scales; CPU absmax/7 differs by <=1.2e-7 relative due to CPU/CUDA scalar division.',
        'holdout_records':len(held)-1,'selection_recomputed_from_development':True,
        'frozen_hashes_unchanged':True,'unused_article_ids_checked':len(articles),'per_seed_block_vs_control':details}
(base/'results/round5_verification.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
print(json.dumps(result,indent=2))
