$ErrorActionPreference='Stop'
$headingRoot=$PSScriptRoot
$headingDrive=$headingRoot.Substring(0,1).ToLowerInvariant()
$headingLinuxRoot='/mnt/'+$headingDrive+$headingRoot.Substring(2).Replace('\','/')
$headingMode=@('--run');if($args.Count -gt 0){$headingMode=$args}
Write-Host 'V23: новое PPO только B, начальные веса68 уже обучены. A и его auxiliary оценки переиспользуются.'
Write-Host 'Создаётся новый output; текущая проверенная модель Gazebo не заменяется.'
& wsl.exe -d Ubuntu-24.04 -u misha --cd $headingLinuxRoot -- [external-home]/vkr_sim/venv-cu128/bin/python -B train_joint_heading23.py @headingMode
exit $LASTEXITCODE
