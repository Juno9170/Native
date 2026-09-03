@echo off
rem Double-clickable launcher for dev.sh (Git Bash).
cd /d "%~dp0"
set "BASH=C:\Program Files\Git\bin\bash.exe"
if not exist "%BASH%" set "BASH=bash"
"%BASH%" ./dev.sh %*
if errorlevel 1 pause
