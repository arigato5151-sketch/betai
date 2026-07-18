@echo off
setlocal
cd /d "%~dp0\.."
if not exist "C:\Python311\python.exe" if not exist "C:\Python312\python.exe" (
  echo Python 3.11 or 3.12 is required. Please install it first.
  exit /b 1
)

set PYTHON_COMMAND=python
where python >nul 2>&1
if errorlevel 1 (
  if exist "C:\Python311\python.exe" set PYTHON_COMMAND=C:\Python311\python.exe
  if exist "C:\Python312\python.exe" set PYTHON_COMMAND=C:\Python312\python.exe
)

%PYTHON_COMMAND% -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt

echo Virtual environment created and dependencies installed.
echo Activate with: .\.venv\Scripts\activate
pause
