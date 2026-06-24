#!/usr/bin/env python3
"""Sample USB PD board power at ~100Hz and write a CSV."""
import signal
import sys
import time

VOLTAGE_PATH = "/sys/class/power_supply/usb/voltage_now"
CURRENT_PATH = "/sys/class/power_supply/usb/current_now"
INTERVAL = 0.01

def main():
    output = sys.argv[1]
    running = True

    def handle_signal(sig, frame):
        nonlocal running
        running = False

    signal.signal(signal.SIGTERM, handle_signal)
    signal.signal(signal.SIGINT, handle_signal)

    with open(output, "w") as f:
        f.write("timestamp,voltage_uv,current_ua,power_watts\n")
        while running:
            t = time.clock_gettime(time.CLOCK_MONOTONIC)
            try:
                with open(VOLTAGE_PATH) as vf:
                    v = int(vf.read().strip())
                with open(CURRENT_PATH) as cf:
                    c = int(cf.read().strip())
                w = (v * c) / 1e12
                f.write(f"{t:.6f},{v},{c},{w:.6f}\n")
                f.flush()
            except (IOError, ValueError):
                pass
            time.sleep(INTERVAL)

if __name__ == "__main__":
    main()
