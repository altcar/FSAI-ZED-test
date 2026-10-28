# Reproduce the Windows stereo-depth validation.
#
# Prerequisite: the kit venv (Python 3.11 + CUDA 12.8 PyTorch).  In this run the
# venv lives at C:\FSAI_3070_venv and is junctioned as .venv because the kit's
# L: drive copied the CUDA wheel too slowly.  Either path works.
$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root

$py = if (Test-Path 'C:\FSAI_3070_venv\Scripts\python.exe') {
    'C:\FSAI_3070_venv\Scripts\python.exe'
} else {
    '.venv\Scripts\python.exe'
}

New-Item -ItemType Directory -Force outputs\logs | Out-Null

function Invoke-Step {
    param([string]$Name, [string[]]$Arguments)
    $log = "outputs\logs\$Name.log"
    Write-Host "== $Name =="
    # Native tools such as unittest write progress to stderr.  With
    # ErrorActionPreference=Stop, Windows PowerShell 5.1 turns that into a
    # NativeCommandError even when the tool exits 0.  Let stderr flow to the
    # log, then decide success from the actual exit code.
    $ErrorActionPreference = 'Continue'
    & $py @Arguments 2>&1 | Tee-Object -FilePath $log
    $code = $LASTEXITCODE
    if ($code -ne 0) {
        throw "$Name failed with exit code $code; see $log"
    }
}

function Invoke-GpuStep {
    param([string]$Name, [string[]]$Arguments)
    # The laptop GPU is shared with desktop apps and occasionally reports
    # cudaErrorDevicesUnavailable between processes.  Retry once; a real code
    # failure will fail both attempts and still surface.
    for ($attempt = 1; $attempt -le 2; $attempt++) {
        try {
            Invoke-Step $Name $Arguments
            return
        } catch {
            if ($attempt -eq 2) { throw }
            Write-Host "  $Name failed (GPU may be busy); retrying once in 10 s..."
            Start-Sleep -Seconds 10
        }
    }
}

Invoke-Step 'environment' @('scripts\check_environment.py')
Invoke-Step 'manifest' @('scripts\verify_manifest.py')
Invoke-Step 'unit_tests' @('-m', 'unittest', 'discover', '-s', 'tests', '-v')

# Control: replay an existing prediction; no network, no GPU.
Invoke-Step 'replay' @(
    'scripts\compare.py',
    '--prediction', 'reference_results\single_depth_check.prediction.npy',
    '--instances', 'data\mono_demo\instances_pinhole.png',
    '--output', 'outputs\replay_windows'
)

# Fresh mono GPU inference on the kit's mono test image.
Invoke-GpuStep 'mono_gpu' @(
    'scripts\compare.py',
    '--device', '0',
    '--instances', 'data\mono_demo\instances_pinhole.png',
    '--output', 'outputs\mono_gpu'
)

# Rectify the stereo pair, run DepthNet and SGBM, write figures and results.
Invoke-GpuStep 'stereo_validation' @(
    'analysis\stereo_depth_validation.py', '--device', '0'
)

# Build STEREO_DEPTH_VALIDATION.md from outputs\stereo_validation\results.json.
Invoke-Step 'report' @('analysis\build_report.py')

Write-Host ''
Write-Host 'Done. Report: STEREO_DEPTH_VALIDATION.md'
Write-Host 'Figures: outputs\stereo_validation\figures'
Write-Host 'Raw results: outputs\stereo_validation\results.json'
