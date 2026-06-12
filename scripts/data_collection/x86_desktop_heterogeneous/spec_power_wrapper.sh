#!/bin/bash
# spec_power_wrapper.sh <BASE> <events> -- <command...>
# Writes <BASE>_sync.txt, <BASE>_power.csv, <BASE>.out
BASE="$1"
EVENTS="$2"
shift 2

python3 -c "import time; print(time.clock_gettime(time.CLOCK_MONOTONIC))" \
    > "${BASE}_sync.txt"

perf stat -I 10 -x, -e power/energy-cores/ -o "${BASE}_power.csv" \
    -- sleep 99999 &
PWR_PID=$!

perf record -e "$EVENTS" -c 10000000 --no-buffering -o "${BASE}.out" -- "$@"
STATUS=$?

kill -INT "$PWR_PID" 2>/dev/null
wait "$PWR_PID" 2>/dev/null
exit $STATUS
