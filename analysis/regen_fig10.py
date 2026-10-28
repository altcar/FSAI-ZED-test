"""Regenerate the process diagram from the saved results (no inference)."""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'analysis'))
from stereo_depth_validation import fig_pipeline  # noqa: E402

r = json.loads((ROOT / 'outputs' / 'stereo_validation' / 'results.json').read_text(
    encoding='utf-8'))
meta = dict(r['geometry'])
meta['sgbm_ms'] = r['timings']['sgbm_ms']
fig_pipeline(meta, dict(r['inference']))
print('regenerated fig10_pipeline.png')
