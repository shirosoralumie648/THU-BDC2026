#!/bin/sh

SCRIPT_DIR="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)"
PYTHON_BIN="${THU_BDC_PYTHON_BIN:-}"

if [ -z "$PYTHON_BIN" ]; then
  if [ -x "$SCRIPT_DIR/.venv/bin/python" ]; then
    PYTHON_BIN="$SCRIPT_DIR/.venv/bin/python"
  elif command -v python >/dev/null 2>&1; then
    PYTHON_BIN="$(command -v python)"
  elif command -v python3 >/dev/null 2>&1; then
    PYTHON_BIN="$(command -v python3)"
  else
    echo "python interpreter not found" >&2
    exit 127
  fi
fi

exec "$PYTHON_BIN" "$SCRIPT_DIR/code/src/predict.py" "$@"
