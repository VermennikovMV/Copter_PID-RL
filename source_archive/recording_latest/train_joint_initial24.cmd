@echo off
setlocal
chcp 65001 >nul
if "%~1"=="" (
  wsl.exe -d Ubuntu-24.04 -u misha --cd "%~dp0." -- [external-home]/vkr_sim/venv-cu128/bin/python -B train_joint_initial24.py --run
) else (
  wsl.exe -d Ubuntu-24.04 -u misha --cd "%~dp0." -- [external-home]/vkr_sim/venv-cu128/bin/python -B train_joint_initial24.py %*
)
exit /b %errorlevel%
