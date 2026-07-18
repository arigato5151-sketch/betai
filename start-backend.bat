@echo off
setlocal
cd /d "%~dp0"
if exist ".venv\Scripts\python.exe" (
  ".venv\Scripts\python.exe" run.py
) else if exist "backend\.venv\Scripts\python.exe" (
  "backend\.venv\Scripts\python.exe" run.py
) else (
  echo No virtual environment found. Run scripts\setup_venv.bat first.
)
pause
