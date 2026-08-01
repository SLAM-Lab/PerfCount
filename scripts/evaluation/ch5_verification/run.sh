#!/usr/bin/env bash
# Recompute and verify every data-backed number in Chapter 5. Seconds to run: it only
# re-aggregates dumped results (no model retraining). See README.md.
set -euo pipefail
cd "$(dirname "$0")"
PY=../../../.venv/bin/python3
[ -x "$PY" ] || PY=python3
exec "$PY" verify_ch5.py "$@"
