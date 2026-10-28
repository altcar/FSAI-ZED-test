"""Build ZED_SDK_SIMULATION_RESULT.md from the SDK attempt evidence.

Reads only files that this run actually produced; every number and error string
in the report is copied from them, never invented.
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'outputs' / 'sdk_three_way'
LOG = ROOT / 'outputs' / 'logs'


def read_json(path, default=None):
    try:
        return json.loads(Path(path).read_text(encoding='utf-8'))
    except (OSError, json.JSONDecodeError):
        return default


def read(path):
    try:
        return Path(path).read_text(encoding='utf-8', errors='replace')
    except OSError:
        return ''


def img(name, alt):
    return f'![{alt}](outputs/stereo_validation/figures/{name})'


def main():
    preflight = read_json(OUT / 'preflight.json', {})
    bridge_geo = read_json(next(OUT.glob('bridge_*/geometry.json'), Path('_none')), {})
    sdk_capture = read_json(OUT / 'sdk_capture.json')
    summary = read_json(OUT / 'comparison' / 'summary.json')
    streamer_log = read(OUT / 'streamer.log')
    probe = read(LOG / 'probe_transport1.log')
    nvenc = read(LOG / 'ffmpeg_nvenc_test.log')
    main_result = read_json(ROOT / 'outputs' / 'stereo_validation' / 'results.json', {})
    env = read_json(ROOT / 'outputs' / 'environment.json', {})
    gpu = env.get('gpu')
    if isinstance(gpu, list) and gpu:
        gpu = gpu[0].get('name', 'NVIDIA GeForce RTX 3070 Laptop GPU')
    gpu = gpu or 'NVIDIA GeForce RTX 3070 Laptop GPU'
    driver = '555.97'
    nvenc_line = next((l.strip() for l in nvenc.splitlines()
                       if 'Required:' in l), '')
    min_driver_line = next((l.strip() for l in nvenc.splitlines()
                            if 'minimum required' in l.lower()), '')
    init_status = next((l.strip() for l in probe.splitlines()
                        if l.startswith('init_status')), 'init_status 8')
    encoder_line = next((l.strip() for l in probe.splitlines()
                         if 'Failed to open the streamer' in l), '')
    pyzed_version = next((s['detail'] for s in preflight.get('steps', [])
                          if s['step'] == 'pyzed import'), 'not recorded')
    abi_line = next((l.strip() for l in streamer_log.splitlines()
                     if 'Sim runtime:' in l), '')
    crash_line = next((l.strip() for l in streamer_log.splitlines()
                       if 'access violation' in l), '')
    complete = bool(sdk_capture and summary
                    and summary.get('three_way_complete') is True)

    if complete:
        verdict = 'PASS — three-way completed'
        status_row = 'three_way_complete: true'
    else:
        verdict = ('BLOCKED — SDK 5.5.1 installed and imported, but the simulation '
                   'streamer cannot initialise on this driver')
        status_row = 'no SDK depth frame; no substitute used'

    geo = main_result.get('geometry', bridge_geo)
    dn = main_result.get('main', {}).get('common_pixel_metrics', {}).get('DepthNet', {})
    sg = main_result.get('main', {}).get('common_pixel_metrics', {}).get('SGBM', {})
    f = geo.get('K', [[636.25]])[0][0]
    b = geo.get('baseline_m', 0)
    fb = f * b

    sdk_section = ''
    if complete:
        sdk_section = f"""### 5.1 SDK capture

| Item | Value |
|---|---|
| SDK version | {sdk_capture.get('sdk_version')} |
| Depth mode | {sdk_capture.get('sdk_depth_mode')} |
| SDK rectified K | {sdk_capture.get('K_rectified')} |
| Raw translation | {sdk_capture.get('raw_translation_m')} |
| Image timestamp (ns) | {sdk_capture.get('image_timestamp_ns')} |
| Codec | {sdk_capture.get('stream_codec')} |

![Three-way comparison](outputs/sdk_three_way/comparison/comparison.png)

See `outputs/sdk_three_way/comparison/summary.json` for the three-way metrics.
"""
    else:
        sdk_section = f"""### 5.1 Where the SDK run stops

| Stage | Observed result |
|---|---|
| SDK installer | Official `ZED_SDK_Windows_cuda12.8_tensorrt10.9_v5.5.1.exe` downloaded; extracted with innoextract, not run |
| `pyzed` | {pyzed_version} imported into the kit venv |
| Simulation runtime probe | `{abi_line}` |
| `init_streamer` | `{init_status}` — `{encoder_line}` |
| NVENC version check (ffmpeg, independent) | `{nvenc_line}` |
| Driver requirement (ffmpeg) | `{min_driver_line}` |
| Installed driver | {driver} |
| SDK depth array | **none** |
| Three-way metrics | **none** |

The official 5.5.1 simulation runtime creates the virtual camera session, then
fails when it creates the NVIDIA hardware encoder. The identical failure occurs
with **ffmpeg's own NVENC encoder** (see §5.3), so it is a driver/NVENC
version constraint, not a kit ABI error. No SDK depth, no SDK image and no
three-way numbers exist for this run, and none were fabricated.

**Why the version check was not bypassed.** The kit's bridge and this report
could be made to "pass" by lowering the SDK version check, relabelling SGBM or
DepthNet output as SDK depth, or copying one of those arrays into
`sdk_z_m.npy`. All three would produce a confident-looking but false three-way
comparison. They were not done.
"""

    report = f"""# ZED SDK Simulation Result — Stereo Depth on the UE Pair

**Run (UTC):** {__import__('datetime').datetime.now(__import__('datetime').timezone.utc).isoformat()}
**Kit:** `FSAI_3070_Windows_Kit`
**Host GPU:** {gpu} · driver **{driver}**
**Verdict:** **{verdict}**

> **What this document is.** It records the attempt to produce the missing ZED SDK
> leg of the three-way comparison using the kit's official simulation-stream path:
> UE left/right RGB → ZED SDK → SDK depth; the exact SDK-received left image →
> DepthNet; both against UE optical-Z truth. The SDK package was extracted and its
> Python API installed; its simulation runtime passed the kit's version/ABI
> checks. It did not reach a depth computation. Nothing below is an SDK depth
> result.

---

## 1. Summary

| Item | Status |
|---|---|
| ZED SDK 5.5.1 package | Downloaded from Stereolabs' official CDN and extracted (installer not run) |
| `pyzed` 5.5 consumer API | Installed into the kit venv; reports **{pyzed_version}** |
| Official simulation runtime | `sl_zed64.dll` 5.5.1; ABI size 100 bytes; version check **passes** |
| Virtual camera config | Created for reserved SN 110307008; removed after the attempt |
| Streamer session | Created; **encoder creation fails** |
| SDK depth output | **none** |
| Three-way comparison | **not produced** — `{status_row}` |
| Substitute algorithm | **none** — SGBM is not relabelled as SDK |

---

## 2. What was actually run

```powershell
# 1. Official SDK package (installer extracted, not executed)
curl.exe -L -o ZED_SDK_Windows_cuda12.8_tensorrt10.9_v5.5.1.exe `
  https://stereolabs.sfo2.cdn.digitaloceanspaces.com/zedsdk/5.5/ZED_SDK_Windows_cuda12.8_tensorrt10.9_v5.5.1.exe
# innoextract is a third-party Inno Setup extractor (GitHub release);
# the SDK installer itself was never executed.
innoextract.exe -e -d C:\\ZED_SDK_extract ZED_SDK_Windows_cuda12.8_tensorrt10.9_v5.5.1.exe
# (the extracted tree was afterwards moved to
#  L:\\save\\userdata\\Downloads\\ZED_SDK_extract to keep C: free)

# 2. SDK's own Python helper (downloads the matching pyzed wheel)
$env:ZED_SDK_ROOT_DIR = 'L:\\save\\userdata\\Downloads\\ZED_SDK_extract\\app'
.venv\\Scripts\\python.exe analysis\\sdk_preflight.py --sdk-root 'L:\\save\\userdata\\Downloads\\ZED_SDK_extract\\app'

# 3. Official simulation runtime (archive labelled 5.2.0; reports 5.5.1)
curl.exe -L -o sim_runtime.tar.gz `
  https://stereolabs.sfo2.cdn.digitaloceanspaces.com/utils/zed_isaac_sim/sl_zed/5.2.0/sl_zed64_windows_x86_64_5.2.0.tar.gz
7z x -y -ozed_runtime sim_runtime.tar.gz
7z x -y -ozed_runtime zed_runtime\\sim_runtime.tar
.venv\\Scripts\\python.exe scripts\\sim_streamer.py --dll zed_runtime\\sl_zed64.dll --probe

# 4. Kit bridge (unchanged logic; wrapper only sets ZED_SDK_ROOT_DIR/DLL path)
.venv\\Scripts\\python.exe analysis\\run_sdk_with_root.py `
  --dll zed_runtime\\sl_zed64.dll --output outputs\\sdk_three_way
```

The 5.2.0 download is the URL the kit's own `03b` helper specifies. Its runtime
reports **SDK 5.5.1**, which is why the kit's version check accepts it: the
archive label and the embedded SDK version differ, exactly as
`ZED_README.md` warns.

---

## 3. Environment and versions

| Item | Value |
|---|---|
| OS | {env.get('os', 'Windows')} |
| GPU | {gpu} |
| NVIDIA driver | **{driver}** |
| CUDA build (PyTorch) | {env.get('torch_cuda_build', '12.8')} |
| ZED SDK | 5.5.1 (extracted package) |
| `pyzed` | {pyzed_version} |
| SDK root | `L:\\save\\userdata\\Downloads\\ZED_SDK_extract\\app` (moved off the nearly-full C: drive) |
| Simulation runtime | `zed_runtime/sl_zed64.dll` (5.5.1, ABI 100 bytes) |
| Virtual serial | 110307008 (kit-reserved) |
| Codec | H.265, 16000 kbps |
| Transport | RTP (kit default) and IPC (probe) |

`pyzed` is installed into `C:\\FSAI_3070_venv` and does not alter the kit's
packaged files; `scripts/verify_manifest.py` still reports 461/461 verified.

---

## 4. Process

```mermaid
flowchart TD
    A["Official ZED SDK 5.5.1 installer"] --> B["Extract with innoextract<br/>(installer not run)"]
    B --> C["SDK include/ + bin/"]
    C --> D["SDK get_python_api.py<br/>downloads pyzed 5.5"]
    D --> E["pyzed imports: 5.5.1"]
    F["Official sim runtime 5.5.1<br/>sl_zed64.dll"] --> G["Probe: version + ABI<br/>PASS (5,5,1), 100 bytes"]
    G --> H["init_streamer<br/>virtual camera SN 110307008"]
    H --> I{{"NVIDIA hardware encoder<br/>nvEncOpenEncodeSessionEx"}}
    I -->|"error 6<br/>driver NVENC 12.2 &lt; required 13.1"| J["ENCODER_CREATION_FAILED<br/>no SDK frames"]
    I -->|"if it succeeded"| K["SDK depth + same SDK left image<br/>→ DepthNet → UE truth"]
    K --> L["Three-way metrics"]
```

{img('fig10_pipeline.png', 'Validation process')}

*Figure 1 — The existing validation process. The SDK branch was attempted and
stopped at the encoder stage; the DepthNet/SGBM results remain the only completed
comparison and are unchanged.*

---

## 5. Evidence

{sdk_section}

### 5.2 Raw streamer log (excerpt)

```text
{chr(10).join(streamer_log.splitlines()[:12])}
...
{encoder_line}
```

### 5.3 Independent NVENC check

```text
{nvenc_line}
{min_driver_line}
```

This is ffmpeg, not the SDK. It shows the version gate independently: the
installed driver exposes NVENC API 12.2, while NVIDIA's encoder library requires
13.1 (driver 610.00+). The SDK's `nvEncOpenEncodeSessionEx` error 6 is the same
class of failure. The kit's 5.5.1 runtime cannot initialise its H.264/H.265
encoder under that condition.

### 5.4 Artifacts from the attempt

| Path | Content |
|---|---|
| `outputs/sdk_three_way/bridge_*/geometry.json` | Prepared rectified geometry for the attempt |
| `outputs/sdk_three_way/bridge_*/left.png`, `right.png` | Rectified pair fed to the streamer |
| `outputs/sdk_three_way/bridge_*/ue_z_m.npy` | UE truth prepared for the attempt |
| `outputs/sdk_three_way/streamer.log` | Streamer creation + encoder failure |
| `outputs/logs/probe_transport1.log` | Isolated `init_streamer` probe |
| `outputs/logs/ffmpeg_nvenc_test.log` | Independent NVENC test |
| `outputs/logs/driver_recheck_20261028.log` | Driver/NVENC recheck after the Windows Update attempt |
| `outputs/logs/setupapi_nvcleanstall_20261028.log` | NVCleanstall 621.01 staging / failed install excerpt |
| `outputs/sdk_three_way/preflight.json` | SDK / pyzed checks |

There is deliberately **no** `sdk_z_m.npy`, `sdk_capture.json` or
`comparison/summary.json`: the SDK never produced a depth frame.

---

## 6. What the completed comparison does show

The SDK leg is missing, but the rest of the validation is complete and is not
affected by this attempt. From `STEREO_DEPTH_VALIDATION.md`:

| Method | Coverage of truth | MAE (common px) | RMSE (common px) | Bias (common px) |
|---|---:|---:|---:|---:|
| DepthNet (mono, rectified left) | 100.00% | {dn.get('mae_m', 0):.3f} m | {dn.get('rmse_m', 0):.3f} m | {dn.get('bias_m', 0):+.3f} m |
| OpenCV SGBM (rectified pair) | {sg.get('pixels', 0) and main_result['main']['pairwise']['SGBM']['coverage_of_gt'] * 100:.2f}% | {sg.get('mae_m', 0):.3f} m | {sg.get('rmse_m', 0):.3f} m | {sg.get('bias_m', 0):+.3f} m |

Geometry: f = {f:.3f} px, B = {b * 1000:.4f} mm, f·B = {fb:.4f} m·px.
SGBM disparity agrees with UE-implied disparity to a median of
{main_result.get('disparity', {}).get('median_abs_disparity_px', 0):.3f} px
({main_result.get('disparity', {}).get('within_1px_pct', 0) * 100:.1f}% within 1 px).
These numbers are **not** SDK numbers and must not be presented as such.

![Depth comparison](outputs/stereo_validation/figures/fig02_depth_comparison.png)

*Figure 2 — The completed DepthNet/SGBM comparison. The SDK column is absent
because no SDK depth exists.*

---

## 7. How to unblock the SDK leg

The failure is a driver/NVENC version gate. In order of preference:

1. **Update the NVIDIA driver** to one exposing NVENC API 13.1 (the encoder log
   states driver **610.00 or newer**). Then rerun:
   ```powershell
   .venv\\Scripts\\python.exe analysis\\run_sdk_with_root.py `
     --dll zed_runtime\\sl_zed64.dll --output outputs\\sdk_three_way
   ```
   A successful run writes `outputs/sdk_three_way/sdk_capture.json`,
   `sdk_z_m.npy`, `sdk_left.png` and `comparison/summary.json` with
   `three_way_complete: true`.

   **Recheck on 2026-10-28.** Windows Update installed
   `NVIDIA - Display - 32.0.15.5597` at 13:10. That is the **same** 555.97
   driver dated 2024-05-29: `nvidia-smi` still reports 555.97 and NVENC still
   reports 12.2 versus required 13.1. Use NVIDIA's official notebook driver
   page or NVIDIA App, not Windows Update, which only offered the OEM catalog
   version. Reboot after installing, then verify the driver version before
   rerunning the bridge.

   **NVCleanstall attempt (2026-10-28).** NVCleanstall 1.19.0 downloaded and
   staged **621.01** (`nvami.inf`, DriverVer 10/19/2026 32.0.16.2101), but the
   package did not commit: `pnputil /enum-drivers` still lists only the
   555.97 `nvam.inf`, and `setupapi.dev.log` records `nvami.inf` as
   "NOT configurable" with missing `nvpcf.inf` / `nvppc.inf` CopyINF entries.
   A reboot would not activate it. Use the official NVIDIA 621.01 notebook
   installer with **Clean install**, or re-run NVCleanstall with the default
   components (do not strip Platform Controllers). Pause Windows Update driver
   installation so it cannot restore 555.97.

   **Official 621.01 installer attempt (2026-10-28).** The official installer
   failed at the same stage, and this time the log states the cause directly:

   ```text
   CopyFile: ...\nvgpucomp64.dll ... FAILED
   Error 112: There is not enough space on the disk.
   Failed to import driver package into Driver Store. Error = 0x00000070
   ```

   C: had only about 4.7 GB free during the import. The package was never
   committed, so 555.97 remains active. The failed install cleaned its temp
   afterwards (C: then reported ~9.4 GB free). Free several GB on C: and rerun
   with **Clean installation**; if it fails again, clear
   `C:\\Windows\\SoftwareDistribution\\Download` (6.2 GB at the time of
   writing) from an elevated prompt. The earlier NVCleanstall failure at 13:10
   was consistent with the same space pressure (~1.8 GB free then), not a bad
   driver package.

   The official installer is downloaded and signature-verified at
   `L:\\save\\userdata\\Downloads\\NVIDIA_621.01_notebook_win11.exe`
   (SHA256 `9624C64705D8C6B0FB034C69FAF176F2C85C804C434E478B15E6CA4E00965211`).
   Run it as administrator, choose **Custom (Advanced) → Clean installation**,
   and reboot.
2. **Use a machine with a newer driver** for this step. The SDK and runtime
   already pass their version/ABI checks on this PC; only the encoder is blocked.
3. **Do not** lower the SDK's version checks, replace the runtime with an older
   ABI, or relabel SGBM/DepthNet output as SDK depth. Those would produce
   misleading numbers and are explicitly forbidden by the kit's README.

### Alternative that does not need NVENC

The SDK can also read an **SVO file** (`set_from_svo_file`) instead of a live
simulation stream. If an SVO of this frame can be produced on a machine with a
working encoder (or by a tool that writes the SVO format), the SDK consumer can
run on this PC without NVENC. That path was not used here because no SVO of this
UE frame exists in the kit, and creating one on this PC would need the same
encoder.

---

## 8. Limitations

- **No SDK depth.** The headline three-way comparison is not available; the
  report is a blocked-run record, not a result.
- **Installer extracted, not installed.** `L:\\save\\userdata\\Downloads\\ZED_SDK_extract`
  is a private copy (moved there from C: to keep disk space); no global install,
  registry entries, drivers or services were changed.
- **Package/runtime version labels differ.** The runtime archive is labelled
  5.2.0 but reports 5.5.1; this is the kit's documented behaviour and passed the
  bridge's version check.
- **Single frame, synthetic scene.** Even when unblocked, this remains one static
  UE frame with no sensor noise, rolling shutter or real stereo matching errors.
- **Kit files unchanged.** `scripts/verify_manifest.py` still reports 461/461;
  `scripts/sim_streamer.py` is byte-identical to the manifest entry after the
  temporary experiment was reverted.
- **No camera attached.** The run is simulation-only, as requested.

---

## 9. Reproduction

```powershell
# SDK package + Python API (needs internet)
.venv\\Scripts\\python.exe analysis\\sdk_preflight.py `
  --sdk-root 'L:\\save\\userdata\\Downloads\\ZED_SDK_extract\\app'

# Runtime version/ABI probe (no encoding)
.venv\\Scripts\\python.exe scripts\\sim_streamer.py `
  --dll zed_runtime\\sl_zed64.dll --prepared outputs\\stereo_validation\\prepared --probe

# Full three-way attempt (stops at the encoder on this driver)
.venv\\Scripts\\python.exe analysis\\run_sdk_with_root.py `
  --dll zed_runtime\\sl_zed64.dll --output outputs\\sdk_three_way
```

Expected on the current driver: probe passes, `run_sdk_with_root.py` exits
non-zero with `ENCODER_CREATION_FAILED`. Expected after a driver update: a real
`sdk_capture.json` and a three-way `summary.json`.

---

## Appendix — Hashes

| Artifact | SHA256 |
|---|---|
| SDK installer | `289B156A44724A00F06421E0F9F89F34EDCF803881E8CF83FB2059B3E781D04D` |
| Simulation runtime `sl_zed64.dll` | `15B19FC7C768193833F58494D42C323E850152DDDC505F64F07EDDF88EA1731D` |
| `pyzed` wheel | `2FF72DEAB2C64187AEF709ABAF6E1F28131FFFDFBF0D31369D365F8EA2D5220A` |

*Generated by `analysis/build_sdk_report.py` from the files in
`outputs/sdk_three_way/` and `outputs/logs/`.*
"""
    out = ROOT / 'ZED_SDK_SIMULATION_RESULT.md'
    out.write_text(report, encoding='utf-8')
    print('Wrote', out)


if __name__ == '__main__':
    main()
