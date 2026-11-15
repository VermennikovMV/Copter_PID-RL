@echo off
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0show_joint_memory20.ps1" %*
exit /b %errorlevel%
