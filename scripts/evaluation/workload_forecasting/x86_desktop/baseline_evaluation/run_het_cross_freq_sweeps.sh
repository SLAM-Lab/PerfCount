#!/bin/bash
# Launch naive and translated cross-frequency het sweeps for P-core (cpu0) and E-core (cpu16).
# Each launcher runs its own internal thread pool; the four sweeps run sequentially
# (naive cpu0 → naive cpu16 → translated cpu0 → translated cpu16) so memory stays bounded.
#
# Usage:
#   ./run_het_cross_freq_sweeps.sh [--max_workers N] [--rescue] [--condense]
#
# All extra arguments are forwarded to the launcher.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LAUNCHER="$SCRIPT_DIR/parallel_launcher_het_cross_freq.py"
PYTHON="$(command -v python3)"

set -euo pipefail

echo "=== Naive cross-freq sweep — cpu0 (P-core) ==="
"$PYTHON" "$LAUNCHER" --cpu 0  --no_translate "$@"

echo ""
echo "=== Naive cross-freq sweep — cpu16 (E-core) ==="
"$PYTHON" "$LAUNCHER" --cpu 16 --no_translate "$@"

echo ""
echo "=== Translated cross-freq sweep — cpu0 (P-core) ==="
"$PYTHON" "$LAUNCHER" --cpu 0  "$@"

echo ""
echo "=== Translated cross-freq sweep — cpu16 (E-core) ==="
"$PYTHON" "$LAUNCHER" --cpu 16 "$@"

echo ""
echo "All sweeps complete."
