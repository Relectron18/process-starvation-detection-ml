"""CPU-bound worker: a tight busy loop, no sleeping. Used to generate real
scheduler contention on the sandbox's single CPU core."""
import sys
import time

end_time = time.time() + float(sys.argv[1]) if len(sys.argv) > 1 else time.time() + 60
x = 0.0001
while time.time() < end_time:
    # pure CPU-bound work, no syscalls that would yield voluntarily
    for _ in range(20000):
        x = (x * 1.0000001) % 1e6
