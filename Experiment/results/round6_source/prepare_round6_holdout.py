"""Prepare a pinned external corpus; never evaluate a model here."""
import json,random,hashlib
from pathlib import Path
import torch
from transformers import AutoTokenizer
from prepare_round3_holdout import MODEL,REVISION
from run_pilot import tensor_hash

out=Path('data/round6_holdout')
if out.exists():raise FileExistsError(out)
source=Path('data/round6_ptb_source');metadata=json.loads((source/'source.json').read_text())
raw=(source/'ptb.test.txt').read_bytes()
assert hashlib.sha256(raw).hexdigest()==metadata['sha256']
tok=AutoTokenizer.from_pretrained(MODEL,revision=REVISION,local_files_only=True)
ids=tok.encode(raw.decode('utf-8'),add_special_tokens=False)
excluded=set()
for p in Path('results').glob('*/config.json'):
    c=json.loads(p.read_text())
    excluded.update(c.get('calibration_window_hashes',[]));excluded.update(c.get('validation_window_hashes',[]))
for p in [Path('results/round2_test/test_manifest.json'),*[Path(f'data/round{r}_holdout/manifest.json') for r in (3,4,5)]]:
    excluded.update(w['window_sha256'] for w in json.loads(p.read_text())['windows'])
groups=list(range(len(ids)//1024));random.Random(20261001).shuffle(groups)
records=[];batches=[]
for group in groups[:32]:
    for j in range(4):
        begin=group*1024+j*256;batch=torch.tensor([ids[begin:begin+256]])
        digest=tensor_hash(batch);assert digest not in excluded
        batches.append(batch);records.append({'source_block':group,'token_start':begin,'window_sha256':digest})
assert len(batches)==128 and len({r['window_sha256'] for r in records})==128
out.mkdir();torch.save(batches,out/'tokens.pt')
manifest={'source':metadata,'model':MODEL,'revision':REVISION,'seed':20261001,
          'source_tokens':len(ids),'windows':records,'bootstrap_groups':32,'tokens_per_window':256,
          'exact_overlap_with_historical_windows':0,'excluded_window_hash_count':len(excluded),
          'tokens_sha256':hashlib.sha256((out/'tokens.pt').read_bytes()).hexdigest(),
          'warning':'Qwen-tokenized preprocessed PTB; contiguous groups, not article identities or standard word-level PTB PPL.'}
(out/'manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
print('Prepared 128 external windows in 32 contiguous blocks; no model evaluation.')
