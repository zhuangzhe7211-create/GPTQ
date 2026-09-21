"""Download a pinned official core, without executing downloaded code."""
import hashlib
import json
from pathlib import Path
from urllib.request import urlopen
from urllib.error import HTTPError

COMMIT = '354a96255e1738e4063212599402503e43e2013d'
ROOT = Path(__file__).resolve().parent / 'vendor' / 'rescomp'


def main():
    ROOT.mkdir(parents=True, exist_ok=True)
    manifest = {'repository': 'https://github.com/list0830/ResComp', 'commit': COMMIT, 'files': {}}
    for relative in ['fake_quant/gptaq_utils_r.py', 'LICENSE']:
        url = f'https://raw.githubusercontent.com/list0830/ResComp/{COMMIT}/{relative}'
        try:
            data = urlopen(url, timeout=40).read()
        except HTTPError as error:
            if relative == 'LICENSE' and error.code == 404:
                manifest['license_note'] = 'No LICENSE at repository root in pinned commit; source retained for local research with provenance.'
                continue
            raise
        path = ROOT / Path(relative).name
        if path.exists() and path.read_bytes() != data:
            raise RuntimeError(f'Refusing to overwrite different source: {path}')
        path.write_bytes(data)
        manifest['files'][path.name] = {'url': url, 'sha256': hashlib.sha256(data).hexdigest()}
    (ROOT / 'manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    print(json.dumps(manifest, indent=2))


if __name__ == '__main__':
    main()
