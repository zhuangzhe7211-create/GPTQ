"""Prepare two fresh confirmation sets without computing any model scores."""
import json,random,hashlib
from pathlib import Path
import torch
from transformers import AutoTokenizer
from prepare_round3_holdout import MODEL,REVISION
from run_pilot import tensor_hash

out=Path('data/round7_holdout')
if out.exists():raise FileExistsError(out)
tok=AutoTokenizer.from_pretrained(MODEL,revision=REVISION,local_files_only=True)
sources=json.loads(Path('data/round7_sources/sources.json').read_text())
excluded=set()
for p in Path('results').glob('*/config.json'):
    c=json.loads(p.read_text());excluded.update(c.get('calibration_window_hashes',[]));excluded.update(c.get('validation_window_hashes',[]))
for p in [Path('results/round2_test/test_manifest.json'),*[Path(f'data/round{r}_holdout/manifest.json') for r in (3,4,5,6)]]:
    excluded.update(w['window_sha256'] for w in json.loads(p.read_text())['windows'])
out.mkdir();allsets={}
for offset,(name,source) in enumerate(sources.items()):
    raw=Path(f'data/round7_sources/{name}.txt').read_bytes();assert hashlib.sha256(raw).hexdigest()==source['sha256']
    ids=tok.encode(raw.decode('utf-8'),add_special_tokens=False)
    groups=list(range(len(ids)//1024));random.Random(20261003+offset).shuffle(groups)
    batches=[];records=[]
    for group in groups:
        items=[]
        for j in range(4):
            start=1024*group+256*j;batch=torch.tensor([ids[start:start+256]])
            items.append((batch,{'source_block':group,'token_start':start,'window_sha256':tensor_hash(batch)}))
        if any(r['window_sha256'] in excluded for b,r in items):continue
        for batch,record in items:batches.append(batch);records.append(record);excluded.add(record['window_sha256'])
        if len(batches)==128:break
    assert len(batches)==128
    torch.save(batches,out/(name+'.pt'))
    allsets[name]={'source':source,'source_tokens':len(ids),'seed':20261003+offset,'windows':records,
                  'tokens_sha256':hashlib.sha256((out/(name+'.pt')).read_bytes()).hexdigest()}
manifest={'model':MODEL,'revision':REVISION,'datasets':allsets,'windows_per_dataset':128,'bootstrap_groups_per_dataset':32,
          'warning':'PTB official valid is a fresh confirmation split for this project; Shakespeare is a new corpus. Contiguous groups are not independent articles.'}
(out/'manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
print('Prepared two new 128-window sets; no model evaluated.')
