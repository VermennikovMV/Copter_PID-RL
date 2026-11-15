@echo off
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0show_joint_metric16.ps1" %*
exit /b %errorlevel%
