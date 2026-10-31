"""IO-bound worker: mostly voluntary sleep (stand-in for blocking I/O),
short bursts of compute in between. Used to populate the io_bound class
with real, distinguishable scheduling behavior."""
import sys
import time

end_time = time.time() + float(sys.argv[1]) if len(sys.argv) > 1 else time.time() + 60
x = 0.0001
while time.time() < end_time:
    time.sleep(0.05)  # voluntary block, stand-in for disk/network wait
    for _ in range(500):  # tiny compute burst
        x = (x * 1.0000001) % 1e6
