"""Create FSAI_STEREO_RESULTS.ipynb with the full combined Markdown content."""
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
md = (ROOT / 'FSAI_STEREO_DEPTH_REPORT.md').read_text(encoding='utf-8')
parts = re.split(r'(?m)^(?=## )', md)
cells = []
for part in parts:
    text = part.strip('\n')
    if not text:
        continue
    cells.append({
        'cell_type': 'markdown',
        'metadata': {},
        'source': text.splitlines(keepends=True),
    })
nb = {
    'cells': cells,
    'metadata': {
        'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'},
        'language_info': {'name': 'python'},
    },
    'nbformat': 4,
    'nbformat_minor': 5,
}
out = ROOT / 'FSAI_STEREO_RESULTS.ipynb'
out.write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding='utf-8')
print('Wrote', out, 'cells:', len(cells))
