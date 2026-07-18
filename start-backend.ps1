Set-Location "$PSScriptRoot"
$rootVenv = Test-Path ".venv\Scripts\python.exe"
$backendVenv = Test-Path "backend\.venv\Scripts\python.exe"

if ($rootVenv) {
    & ".\.venv\Scripts\python.exe" "run.py"
} elseif ($backendVenv) {
    & "backend\.venv\Scripts\python.exe" "run.py"
} else {
    Write-Error "No virtual environment found. Run scripts\setup_venv.ps1 first."
    exit 1
}
