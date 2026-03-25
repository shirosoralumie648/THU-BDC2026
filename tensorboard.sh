#!/usr/bin/env bash
set -euo pipefail

DEFAULT_LOGDIR="$(python - <<'PY'
import os
import sys
sys.path.insert(0, os.path.join(os.getcwd(), 'code/src'))
from config import config
print(os.path.join(config['output_dir'], 'log'))
PY
)"
LOGDIR="${1:-${DEFAULT_LOGDIR}}"
HOST="${TENSORBOARD_HOST:-127.0.0.1}"
PORT="${TENSORBOARD_PORT:-6006}"

echo "TensorBoard logdir: ${LOGDIR}"
echo "TensorBoard url: http://${HOST}:${PORT}"
python -m tensorboard.main --logdir "${LOGDIR}" --host "${HOST}" --port "${PORT}"
