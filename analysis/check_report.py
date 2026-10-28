"""Sanity-check the generated Markdown tables and image links."""
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / 'STEREO_DEPTH_VALIDATION.md'
text = REPORT.read_text(encoding='utf-8')
lines = text.splitlines()

blocks = []
cur = []
for line in lines:
    if line.strip().startswith('|'):
        cur.append(line)
    else:
        if cur:
            blocks.append(cur)
            cur = []
if cur:
    blocks.append(cur)

bad = 0
for i, block in enumerate(blocks):
    counts = {len(line.strip().strip('|').split('|')) for line in block}
    if len(counts) != 1:
        bad += 1
        print(f'BAD TABLE {i}: cell counts {counts}')
        for line in block:
            print('  ', line)

links = re.findall(r'!\[[^\]]*\]\(([^)]+)\)', text)
missing = [l for l in links if not (ROOT / l).is_file()]
print(f'tables={len(blocks)} bad={bad} images={len(links)} missing={len(missing)}')
for m in missing:
    print('MISSING', m)
print('placeholders=', text.count('—'), 'lines=', len(lines),
      'chars=', len(text))
