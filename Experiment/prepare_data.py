"""Prepare independent pilot JSONL from official WikiText-2 train/validation splits.

Explicit invocation downloads the dataset. Test split is untouched.
"""
import argparse
import hashlib
import json
from pathlib import Path
import random


def main():
    from datasets import load_dataset
    p = argparse.ArgumentParser()
    p.add_argument('--out', default='data/wikitext2')
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--revision', default='main')
    args = p.parse_args()
    dest = Path(args.out)
    dest.mkdir(parents=True, exist_ok=True)
    if any((dest / n).exists() for n in ('calibration.jsonl', 'validation.jsonl')):
        raise FileExistsError('Use a new output directory')
    meta = {'dataset': 'Salesforce/wikitext', 'config': 'wikitext-2-raw-v1',
            'revision_requested': args.revision, 'seed': args.seed, 'splits': {}}
    seen = set()
    for split, filename in [('train', 'calibration.jsonl'), ('validation', 'validation.jsonl')]:
        dataset = load_dataset('Salesforce/wikitext', 'wikitext-2-raw-v1', split=split, revision=args.revision)
        entries = [(i, row['text']) for i, row in enumerate(dataset) if len(row['text'].strip()) >= 500]
        random.Random(args.seed).shuffle(entries)
        written = []
        for i, text in entries:
            digest = hashlib.sha256(text.encode()).hexdigest()
            if digest in seen:
                continue
            seen.add(digest)
            written.append({'text': text, 'source_split': split, 'source_row': i, 'sha256': digest})
        payload = ''.join(json.dumps(x, ensure_ascii=False) + '\n' for x in written)
        (dest / filename).write_text(payload, encoding='utf-8')
        meta['splits'][split] = {'fingerprint': dataset._fingerprint, 'documents': len(written),
                                 'file_sha256': hashlib.sha256(payload.encode()).hexdigest()}
    (dest / 'manifest.json').write_text(json.dumps(meta, indent=2), encoding='utf-8')
    print(dest.resolve())


if __name__ == '__main__':
    main()
