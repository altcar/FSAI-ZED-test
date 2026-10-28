# Packaging checks (2026-10-27)

- Mac CPU fresh model inference: completed; prediction array bitwise equals previous reference.
- Mono replay: 419524 pixels, MAE 2.6131930395m, RMSE 2.7477118496m.
- Native cone instance mask: 5895 evaluated pixels, MAE 1.4218563511m.
- Prepared stereo pair: all target pixels have source image coverage; corrected optical Z used.
- Prepared stereo left Depth Net CPU run: 270876 pixels, MAE 2.4419448642m. This is not SDK output.
- Four geometry/metrics/ABI unit tests: passed.
- C++ static assertions against official header: size 100 and offsets 44/60/96 passed.
- Windows .cmd scripts: reviewed and saved with CRLF, not executed on Windows.
- CUDA, SDK DLL loading, simulator streaming and SDK depth: not tested on the Mac.
- No completed three-way SDK comparison is included.

Environment versions are in mac_cpu_environment.json. Output figures include metric depth and error color bars.
