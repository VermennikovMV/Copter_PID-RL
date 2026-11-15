# User-invoked actual fresh reproduction via official metric-first preflight.
$ErrorActionPreference = 'Stop'
$recordingRoot = $PSScriptRoot
$recordingDrive = $recordingRoot.Substring(0,1).ToLowerInvariant()
$recordingLinuxRoot = '/mnt/' + $recordingDrive + $recordingRoot.Substring(2).Replace('\','/')
$recordingMode = @('--run')
if ($args.Count -gt 0) { $recordingMode = $args }
Write-Host 'Fresh PPO reproduction from the existing trained initial model.'
Write-Host 'A unique output is created; prior experiments are preserved.'
& wsl.exe -d Ubuntu-24.04 -u misha --cd $recordingLinuxRoot -- [external-home]/vkr_sim/venv-cu128/bin/python -B scripts/prepare_metric16_recording_v1.py @recordingMode
exit $LASTEXITCODE
