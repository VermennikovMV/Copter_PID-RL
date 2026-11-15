$ErrorActionPreference='Stop'
$memoryRoot=$PSScriptRoot
$memoryDrive=$memoryRoot.Substring(0,1).ToLowerInvariant()
$memoryLinuxRoot='/mnt/'+$memoryDrive+$memoryRoot.Substring(2).Replace('\','/')
$memoryMode=@('--run')
if($args.Count -gt 0){$memoryMode=$args}
Write-Host 'V21P1: новое дообучение уже обученных весов в PyBullet; Gazebo проверяется отдельно.'
Write-Host 'Старая V21 сохранена как INFRA_FAILED. Создаётся новый output, текущая модель не заменяется.'
& wsl.exe -d Ubuntu-24.04 -u misha --cd $memoryLinuxRoot -- [external-home]/vkr_sim/venv-cu128/bin/python -B scripts/joint_memory21p1_training_v1.py @memoryMode
exit $LASTEXITCODE
