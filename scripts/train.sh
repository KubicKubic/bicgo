#!/usr/bin/env bash
# Launch a full bicgo training run on the local GPU.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="${PY:-/mnt/pfs/guoyuchong/guoyuchong/.venv-gnn-jax/bin/python}"
# Cap JAX's GPU reservation so the shared A100 is not monopolised. The default
# config peaks around 35 GiB; 0.6 of an 80GB card (48 GiB) leaves headroom.
export XLA_PYTHON_CLIENT_MEM_FRACTION="${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.6}"
cd "$HERE"
exec "$PY" -u -m bicgo.train --config configs/default.json "$@"
