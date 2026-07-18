#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

PYTHON_EXECUTABLE=$(command -v python3 || command -v python)
if [ -z "$PYTHON_EXECUTABLE" ]; then
  echo "Python is not installed. Install Python 3.11 or 3.12 first."
  exit 1
fi

$PYTHON_EXECUTABLE -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt

echo "Virtual environment created and dependencies installed."
echo "Activate with: source .venv/bin/activate"
