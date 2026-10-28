import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
md = ROOT / 'FSAI_STEREO_DEPTH_REPORT.md'
text = md.read_text(encoding='utf-8')
blocks, cur = [], []
for line in text.splitlines():
    if line.strip().startswith('|'):
        cur.append(line)
    elif cur:
        blocks.append(cur)
        cur = []
if cur:
    blocks.append(cur)
bad = [(i, {len(r.strip().strip('|').split('|')) for r in b})
       for i, b in enumerate(blocks)
       if len({len(r.strip().strip('|').split('|')) for r in b}) != 1]
links = re.findall(r'!\[[^\]]*\]\(([^)]+)\)', text)
missing = [l for l in links if not (ROOT / l).is_file()]
nb = json.loads((ROOT / 'FSAI_STEREO_RESULTS.ipynb').read_text(encoding='utf-8'))
print(f'tables={len(blocks)} bad={bad} images={len(links)} missing={missing} notebook_cells={len(nb["cells"])}')
print('notebook_first=', ''.join(nb['cells'][0]['source']).splitlines()[0])
