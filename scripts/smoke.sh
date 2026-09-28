#!/usr/bin/env bash
# Fast smoke checks for bicgo. Set SMOKE_FULL=1 to also run the full test suite.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="${PY:-/mnt/pfs/guoyuchong/guoyuchong/.venv-gnn-jax/bin/python}"
export XLA_PYTHON_CLIENT_MEM_FRACTION="${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.3}"
cd "$HERE"

echo "== fast smoke tests (configs/smoke.json) =="
"$PY" -m pytest tests/test_smoke.py -q

echo "== one tiny training run =="
"$PY" -m bicgo.train --config configs/smoke.json --out /tmp/bicgo_smoke

if [[ "${SMOKE_FULL:-0}" == "1" ]]; then
  echo "== full test suite =="
  "$PY" -m pytest tests/ -q
fi

echo "== smoke OK =="
