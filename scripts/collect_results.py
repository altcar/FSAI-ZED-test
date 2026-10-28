from datetime import datetime, timezone
from pathlib import Path
import zipfile

ROOT = Path(__file__).resolve().parents[1]

if __name__ == '__main__':
    if not (ROOT / 'outputs').is_dir():
        raise SystemExit('No outputs yet. Run a comparison first.')
    name = ROOT / ('FSAI_results_' + datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ') + '.zip')
    files = [p for p in (ROOT / 'outputs').rglob('*') if p.is_file()]
    for leaf in ['installed_requirements.txt', 'MANIFEST_SHA256.json']:
        if (ROOT / leaf).is_file():
            files.append(ROOT / leaf)
    with zipfile.ZipFile(name, 'x', zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(files):
            archive.write(path, path.relative_to(ROOT))
    print('Send this results archive back:', name)
