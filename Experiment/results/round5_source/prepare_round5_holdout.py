"""Reserve 128 unused train articles; never compute any model losses here."""
import hashlib
import json
from pathlib import Path
import random
import re

from datasets import load_dataset
import torch
from transformers import AutoTokenizer

from run_pilot import tensor_hash, sequences, read_texts

MODEL = 'Qwen/Qwen2.5-0.5B'
REVISION = '060db6499f32faf8b98477b0a26969ef7d8b9987'
DATA_REVISION = 'b08601e04326c79dfdd32d625aee71d232d685c3'


def main():
    out = Path('data/round5_holdout')
    if out.exists():
        raise FileExistsError(out)
    tokenizer = AutoTokenizer.from_pretrained(MODEL, revision=REVISION, local_files_only=True)
    dataset = load_dataset('Salesforce/wikitext', 'wikitext-2-raw-v1', split='train', revision=DATA_REVISION)
    article_ids, article = [], -1
    for index, row in enumerate(dataset):
        if re.fullmatch(r'\s*= [^=\n]+ =\s*', row['text']):
            article = index
        article_ids.append(article)
    excluded_articles, excluded_texts, excluded_windows = set(), set(), set()
    used_rows = set()
    for seed in range(3):
        path = Path(f'data/wikitext2_20260926_s{seed}/calibration.jsonl')
        count = 0
        for line in path.read_text(encoding='utf-8').splitlines():
            row = json.loads(line)
            ids = tokenizer.encode(row['text'], add_special_tokens=False)
            for start in range(0, len(ids) - 256 + 1, 256):
                excluded_articles.add(article_ids[row['source_row']])
                excluded_texts.add(row['text'])
                used_rows.add(row['source_row'])
                excluded_windows.add(tensor_hash(torch.tensor([ids[start:start+256]])))
                count += 1
                if count == 64:
                    break
            if count == 64:
                break
        assert count == 64
    validation = read_texts('data/wikitext2_20260926_s0/validation.jsonl')
    excluded_texts.update(validation)
    excluded_windows.update(map(tensor_hash, sequences(validation, tokenizer, 64, 256)))
    # Explicitly prove that the historical calibration windows are excluded too.
    for seed in range(3):
        config = json.loads(Path(f'results/round2_s{seed}/config.json').read_text())
        assert set(config['calibration_window_hashes']).issubset(excluded_windows)
    previous = {'windows': []}
    for prior_round in (3,4):
        previous['windows'].extend(json.loads(Path(f'data/round{prior_round}_holdout/manifest.json').read_text())['windows'])
    prior_articles = {r['article_heading_row'] for r in previous['windows']}
    excluded_articles.update(prior_articles)
    excluded_windows.update(r['window_sha256'] for r in previous['windows'])
    prior_texts = {r['text_sha256'] for r in previous['windows']}
    excluded_texts.update(row['text'] for row in dataset if hashlib.sha256(row['text'].encode()).hexdigest() in prior_texts)
    historical_test = json.loads(Path('results/round2_test/test_manifest.json').read_text())['windows']
    excluded_windows.update(r['window_sha256'] for r in historical_test)
    historical_texts = {r['text_sha256'] for r in historical_test}
    excluded_texts.update(row['text'] for row in dataset if hashlib.sha256(row['text'].encode()).hexdigest() in historical_texts)
    options = {}
    for index, row in enumerate(dataset):
        article = article_ids[index]
        text = row['text']
        if article < 0 or article in excluded_articles or text in excluded_texts or len(text.strip()) < 500:
            continue
        ids = tokenizer.encode(text, add_special_tokens=False)
        if len(ids) < 256:
            continue
        batch = torch.tensor([ids[:256]], dtype=torch.long)
        digest = tensor_hash(batch)
        if digest in excluded_windows:
            continue
        options.setdefault(article, []).append((index, text, batch, digest))
    rng = random.Random(20260929)
    articles = sorted(options)
    rng.shuffle(articles)
    batches, records = [], []
    for article in articles:
        choices = options[article]
        rng.shuffle(choices)
        for index, text, batch, digest in choices:
            if digest in excluded_windows or text in excluded_texts:
                continue
            batches.append(batch)
            records.append({'source_row': index, 'article_heading_row': article,
                            'text_sha256': hashlib.sha256(text.encode()).hexdigest(), 'window_sha256': digest})
            excluded_windows.add(digest)
            excluded_texts.add(text)
            break
        if len(batches) == 128:
            break
    assert len(batches) >= 64, f'Only {len(batches)} eligible unused articles'
    assert len({x['article_heading_row'] for x in records}) == len(batches)
    assert not {x['article_heading_row'] for x in records} & excluded_articles
    out.mkdir()
    torch.save(batches, out / 'tokens.pt')
    manifest = {'dataset': 'Salesforce/wikitext', 'revision': DATA_REVISION, 'source_split': 'train',
                'kind': 'New held-out unused articles, not official test split', 'seed': 20260929,
                'model': MODEL, 'model_revision': REVISION, 'windows': records,
                'excluded_calibration_article_ids': sorted(excluded_articles-prior_articles),
                'excluded_round3_and_round4_article_ids': sorted(prior_articles),
                'excluded_calibration_source_rows': sorted(used_rows),
                'tokens_sha256': hashlib.sha256((out/'tokens.pt').read_bytes()).hexdigest()}
    (out / 'manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    print('Reserved',len(records),'articles; excluded',len(excluded_articles),'calibration/previous articles; no model evaluation.')


if __name__ == '__main__':
    main()
