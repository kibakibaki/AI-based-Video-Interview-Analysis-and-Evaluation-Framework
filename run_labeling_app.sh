#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="$PROJECT_DIR/backend/venv311/bin/python"

if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "Virtual environment not found at backend/venv311."
  echo "Create it first using the Environment Setup section in README.md."
  exit 1
fi

cd "$PROJECT_DIR"

if ! "$PYTHON_BIN" -c "import tkinter" >/dev/null 2>&1; then
  echo "Tkinter is not available in this Python installation."
  echo "Install the macOS Tk support for Homebrew Python 3.11:"
  echo "  brew install python-tk@3.11"
  echo "Then run this command again:"
  echo "  ./run_labeling_app.sh"
  exit 1
fi

exec "$PYTHON_BIN" tools/labeling_app.py "$@"
