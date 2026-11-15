@echo off
setlocal
powershell.exe -NoLogo -NoProfile -STA -ExecutionPolicy Bypass -File "%~dp0show_joint_training_native_v24.ps1" %*
exit /b %errorlevel%
