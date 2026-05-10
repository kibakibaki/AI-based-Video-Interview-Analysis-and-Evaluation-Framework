#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="$PROJECT_DIR/backend/venv311/bin/python"

if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "Virtual environment not found at backend/venv311."
  echo "Create it first:"
  echo "  python3.11 -m venv backend/venv311"
  echo "  # If python3.11 is missing on Apple Silicon macOS:"
  echo "  # brew install python@3.11"
  echo "  # /opt/homebrew/opt/python@3.11/bin/python3.11 -m venv backend/venv311"
  echo "  source backend/venv311/bin/activate"
  echo "  python -m pip install --upgrade pip"
  echo "  python -m pip install -r backend/requirements.txt"
  echo "  python -m pip install --no-deps -r backend/requirements-gaze.txt"
  echo ""
  echo "If backend/venv311 exists but is broken, remove it with:"
  echo "  rm -rf backend/venv311"
  echo "Then recreate it with the commands above."
  exit 1
fi

cd "$PROJECT_DIR"
exec "$PYTHON_BIN" backend/app.py
