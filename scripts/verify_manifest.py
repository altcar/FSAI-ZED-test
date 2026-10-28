"""Check that all transferred kit files still match MANIFEST_SHA256.json."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    manifest = json.loads((ROOT / 'MANIFEST_SHA256.json').read_text(encoding='utf-8'))
    failed = []
    for name, expected in manifest.items():
        path = ROOT / name
        if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            failed.append(name)
    if failed:
        raise SystemExit('Missing/changed files: ' + ', '.join(failed))
    print(f'OK: {len(manifest)} original kit files verified. Outputs and .venv are not checked.')


if __name__ == '__main__':
    main()
