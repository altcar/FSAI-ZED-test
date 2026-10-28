"""Create and execute FSAI_STEREO_RESULTS.ipynb from the final result files."""
import json
from pathlib import Path

import nbformat
from nbclient import NotebookClient

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'FSAI_STEREO_RESULTS.ipynb'


def md(text):
    return nbformat.v4.new_markdown_cell(text)


def code(src):
    return nbformat.v4.new_code_cell(src)


cells = [
    md("""# FSAI Stereo Depth — Results Notebook

**Run:** 2026-10-28 · **GPU:** RTX 3070 Laptop 8 GB · **Driver:** 621.01
**SDK:** ZED SDK 5.5.1 / pyzed 5.5.1 · **Mode:** PERFORMANCE (H.264 3000 kbps)

This notebook collects the final stereo, DepthNet, SGBM and ZED SDK results. It
reads the JSON/PNG outputs only; it does not rerun inference or need a GPU.

The two main reports are:

- `STEREO_DEPTH_VALIDATION.md`
- `ZED_SDK_SIMULATION_RESULT.md`
"""),
    code("""from pathlib import Path
import json
from IPython.display import Markdown, Image, display

ROOT = Path.cwd()
stereo = json.loads((ROOT / 'outputs/stereo_validation/results.json').read_text())
sdk = json.loads((ROOT / 'outputs/sdk_three_way/results.json').read_text())
env = json.loads((ROOT / 'outputs/environment.json').read_text())

def table(headers, rows):
    lines = ['| ' + ' | '.join(headers) + ' |',
             '| ' + ' | '.join('---' for _ in headers) + ' |']
    lines += ['| ' + ' | '.join(str(c) for c in r) + ' |' for r in rows]
    return '\\n'.join(lines)

def f(v, n=3):
    return '—' if v is None else f'{v:.{n}f}'

print('stereo run:', stereo['run_utc'])
print('stereo device:', stereo['inference'].get('device'), '| torch', env['torch'])
print('SDK capture:', sdk['capture']['sdk_version'], sdk['capture']['sdk_depth_mode'])
"""),
    md("## 1. Stereo geometry and SGBM (clean prepared pair)"),
    code("""main = stereo['main']; pw = main['pairwise']; cm = main['common_pixel_metrics']
rows = []
for name in pw:
    p, c = pw[name], cm[name]
    rows.append([name, f"{p['pixels']:,}", f"{p['coverage_of_gt']*100:.2f}%",
                 f(p['mae_m']), f(p['rmse_m']), f(p['bias_m'], 3),
                 f(c['mae_m']), f(c['median_abs_error_m']), f(c['p90_abs_error_m'])])
display(Markdown('### Main metrics\\n' + table(
    ['Method', 'Valid px', 'Coverage', 'MAE', 'RMSE', 'Bias', 'MAE common',
     'Median common', 'P90 common'], rows)))
d = stereo['disparity']
display(Markdown(f\"\"\"
**Geometry:** f = {stereo['geometry']['K'][0][0]:.3f} px,
B = {stereo['geometry']['baseline_m']*1000:.4f} mm;
epipolar median |dy| {stereo['epipolar']['median_abs_dy_px']:.3f} px;
SGBM median |Δd| {d['median_abs_disparity_px']:.3f} px,
{d['within_1px_pct']*100:.1f}% within 1 px.
\"\"\"))
"""),
    md("""## 2. ZED SDK three-way result (PERFORMANCE mode)

The SDK capture is **PERFORMANCE**, not NEURAL, and the stream is H.264
3000 kbps. DepthNet was evaluated on the R/B-corrected SDK left image
(`sdk_left_corrected.png`); the raw `sdk_left.png` had a channel swap that
inflated DepthNet to 9.34 m and is not used for the fair comparison.
"""),
    code("""s = sdk['sdk']; d = sdk['depthnet_same_image']; sc = sdk['sdk_common']; dc = sdk['depthnet_common']
rows = [
    ['ZED SDK', f"{s['pixels']:,}", f"{s['coverage']*100:.2f}%", f(s['mae_m']), f(s['rmse_m']),
     f(s['bias_m'], 3), f(s['median_abs_error_m']), f(s['p90_abs_error_m']), f"{s['bad_3m']*100:.1f}%"],
    ['DepthNet', f"{d['pixels']:,}", f"{d['coverage']*100:.2f}%", f(d['mae_m']), f(d['rmse_m']),
     f(d['bias_m'], 3), f(d['median_abs_error_m']), f(d['p90_abs_error_m']), f"{d['bad_3m']*100:.1f}%"],
]
display(Markdown('### SDK vs DepthNet on the same image\\n' + table(
    ['Method', 'Valid px', 'Coverage', 'MAE', 'RMSE', 'Bias', 'Median', 'P90', '≥3 m'], rows)))
display(Markdown(f\"\"\"
On the {sdk['common_pixels']:,} common pixels: SDK MAE **{sc['mae_m']:.3f} m**,
DepthNet MAE **{dc['mae_m']:.3f} m**. The SDK has the lower median
({sc['median_abs_error_m']:.3f} vs {dc['median_abs_error_m']:.3f} m) but the
heavier tail (RMSE {sc['rmse_m']:.3f} vs {dc['rmse_m']:.3f} m). This is a mixed
result, not a single-method win.
\"\"\"))
"""),
    md("## 3. Per-depth-bin SDK table"),
    code("""rows = []
for b in sdk['bins']:
    for name in ('ZED SDK', 'DepthNet'):
        x = b[name]
        rows.append([name, f"{b['lo']}–{b['hi']}", f"{b['common']:,}",
                     f"{x['coverage']*100:.1f}%", f(x['mae_m']), f(x['rmse_m']),
                     f(x['bias_m'], 3), f(x['p90_abs_error_m'])])
display(Markdown(table(['Method', 'Truth bin (m)', 'Common px', 'Coverage',
                        'MAE', 'RMSE', 'Bias', 'P90'], rows)))
"""),
    md("## 4. Figures"),
    code("""for name in ['fig01_depth_panels.png', 'fig02_scatter.png', 'fig03_bins.png',
             'fig04_thresholds.png', 'fig05_headline.png', 'fig06_process.png']:
    p = ROOT / 'outputs/sdk_three_way/figures' / name
    if p.is_file():
        display(Markdown(f'**{name}**'))
        display(Image(filename=str(p), width=1000))
"""),
    code("""for name in ['fig01_input_pair.png', 'fig02_depth_comparison.png',
             'fig03_scatter.png', 'fig04_error_vs_depth.png',
             'fig05_error_distribution.png', 'fig06_headline_metrics.png',
             'fig07_cone_regions.png', 'fig08_disparity_consistency.png',
             'fig09_epipolar_check.png', 'fig10_pipeline.png']:
    p = ROOT / 'outputs/stereo_validation/figures' / name
    if p.is_file() and name in ('fig02_depth_comparison.png', 'fig04_error_vs_depth.png',
                                'fig08_disparity_consistency.png', 'fig09_epipolar_check.png'):
        display(Markdown(f'**{name}**'))
        display(Image(filename=str(p), width=1000))
"""),
    md("""## 5. Caveats and reproduction

- **PERFORMANCE**, not NEURAL: TensorRT DLLs were not on the consumer search path.
- **H.264 3000 kbps**, chunk 1200, transport BOTH, extended `StreamingParameters`
  target fields; first `ENCODING_FAILED` tolerated.
- **Metadata mismatch:** 5.2.0 runtime sends 1,216 bytes; 5.5.1 consumer expects
  2,904 and skips camera settings. K and baseline passed.
- **Static frame:** duplicate-frame warnings / IDR requests; one capture.
- **Run-to-run variation:** two captures gave MAE 2.800 and 3.117 m; this
  notebook uses the final capture (3.117 m).
- The stock `04_run_zed_three_way_EXPERIMENTAL.cmd` fails; the fixed producer is
  documented in `ZED_SDK_SIMULATION_RESULT.md` and preserved in `analysis/`.

See `RUN_SUMMARY.md` and `AI_HANDOFF_NOTES.md` for the full command/test matrix
and the do-not-repeat diagnosis notes.
"""),
]

nb = nbformat.v4.new_notebook(
    cells=cells,
    metadata={
        'kernelspec': {'name': 'fsai', 'display_name': 'FSAI Results', 'language': 'python'},
        'language_info': {'name': 'python'},
    })
OUT.write_text(nbformat.writes(nb), encoding='utf-8')
client = NotebookClient(nb, timeout=120, kernel_name='fsai',
                        resources={'metadata': {'path': str(ROOT)}})
client.execute()
nbformat.write(nb, OUT)
print('Wrote and executed', OUT)
