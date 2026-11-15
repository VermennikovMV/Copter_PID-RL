@echo off
chcp 65001 >nul
powershell.exe -NoLogo -NoProfile -ExecutionPolicy Bypass -File "%~dp0train_joint_heading23.ps1" %*
exit /b %errorlevel%
