#!/bin/sh

case "$0" in
  */*) SCRIPT_DIR_INPUT=${0%/*} ;;
  *) SCRIPT_DIR_INPUT=. ;;
esac

SCRIPT_DIR="$(CDPATH= cd -- "$SCRIPT_DIR_INPUT" && pwd)"
PYTHON_BIN="${THU_BDC_PYTHON_BIN:-}"

if [ -z "$PYTHON_BIN" ]; then
  SEARCH_DIR="$SCRIPT_DIR"
  while [ -n "$SEARCH_DIR" ] && [ "$SEARCH_DIR" != "/" ]; do
    if [ -x "$SEARCH_DIR/.venv/bin/python" ]; then
      PYTHON_BIN="$SEARCH_DIR/.venv/bin/python"
      break
    fi
    NEXT_DIR="$(CDPATH= cd -- "$SEARCH_DIR/.." && pwd)"
    if [ "$NEXT_DIR" = "$SEARCH_DIR" ]; then
      break
    fi
    SEARCH_DIR="$NEXT_DIR"
  done
  if [ -z "$PYTHON_BIN" ] && [ -x "/.venv/bin/python" ]; then
    PYTHON_BIN="/.venv/bin/python"
  fi
  if [ -z "$PYTHON_BIN" ] && command -v python >/dev/null 2>&1; then
    PYTHON_BIN="$(command -v python)"
  elif [ -z "$PYTHON_BIN" ] && command -v python3 >/dev/null 2>&1; then
    PYTHON_BIN="$(command -v python3)"
  fi
  if [ -z "$PYTHON_BIN" ]; then
    echo "python interpreter not found" >&2
    exit 127
  fi
fi

exec "$PYTHON_BIN" "$SCRIPT_DIR/code/src/predict.py" "$@"
