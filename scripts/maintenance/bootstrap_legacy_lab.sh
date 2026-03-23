#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
VENV_DIR="$ROOT_DIR/.venv"

cd "$ROOT_DIR"

if [[ ! -x "$VENV_DIR/bin/python" ]]; then
  python3 -m venv "$VENV_DIR"
fi

"$VENV_DIR/bin/python" -m pip install --upgrade pip
"$VENV_DIR/bin/python" -m pip install -r requirements.txt

"$VENV_DIR/bin/python" -m pytest -q \
  tests/test_api_route_compatibility.py \
  tests/test_legacy_lab_smoke.py \
  tests/test_new_services.py \
  tests/test_geographic_simple.py \
  tests/test_bardcode_analyzer.py \
  tests/test_visualization.py

cd "$ROOT_DIR/frontend"

if [[ ! -d node_modules ]]; then
  npm install
fi

npm run build
