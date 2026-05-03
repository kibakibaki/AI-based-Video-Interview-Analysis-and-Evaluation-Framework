#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="$PROJECT_DIR/backend/venv311/bin/python"

if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "Virtual environment not found at backend/venv311."
  echo "Create it first:"
  echo "  python3.11 -m venv backend/venv311"
  echo "  source backend/venv311/bin/activate"
  echo "  pip install -r backend/requirements.txt"
  exit 1
fi

cd "$PROJECT_DIR"
exec "$PYTHON_BIN" backend/app.py
