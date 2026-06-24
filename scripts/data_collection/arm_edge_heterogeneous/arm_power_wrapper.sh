#!/bin/bash
# arm_power_wrapper.sh <BASE> <CPU> <EVENTS> -- <command...>
# Writes <BASE>_sync.txt, <BASE>_power.csv, <BASE>.out
BASE="$1"
CPU="$2"
EVENTS="$3"
shift 3
# Consume the optional "--" separator
if [ "$1" = "--" ]; then shift; fi

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

python3 -c "import time; print(time.clock_gettime(time.CLOCK_MONOTONIC))" \
    > "${BASE}_sync.txt"

python3 "${SCRIPT_DIR}/arm_power_sampler.py" "${BASE}_power.csv" &
PWR_PID=$!

perf record -a -C "$CPU" -e "$EVENTS" -c 10000000 --no-buffering -o "${BASE}.out" -- "$@"
STATUS=$?

kill "$PWR_PID" 2>/dev/null
wait "$PWR_PID" 2>/dev/null
exit $STATUS
