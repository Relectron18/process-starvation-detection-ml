"""
Real-world data collection: spawns actual competing OS processes on this
sandbox's single CPU core (CPU-bound at normal priority, CPU-bound at low
niceness, I/O-bound, and a real SCHED_FIFO real-time hog that genuinely
starves everything else for a window) and samples /proc/[pid]/schedstat
directly -- no simulation, no synthetic generator -- to build a labeled
dataset of real scheduling behavior, including a real, verifiable
starvation episode.

Ground-truth starvation label is operational, not assumed: a CPU-bound
process (always runnable, never voluntarily sleeps) is labeled starved in
a given window if it received almost no CPU time while accumulating heavy
runqueue wait -- i.e. it wanted to run and the scheduler didn't let it.
"""
import csv
import json
import os
import platform
import re
import shutil
import sys
import subprocess
import time

from config import DATA_DIR, BASE_DIR

if platform.system() != "Linux" or not os.path.isdir("/proc"):
    sys.exit(
        "This script reads /proc/[pid]/schedstat, /proc/[pid]/stat, and "
        "/proc/[pid]/status directly and only runs on Linux. Detected "
        f"platform: {platform.system()}. See paper Section 4.2 - real-hardware "
        "data collection requires genuine Linux kernel scheduler accounting; "
        "there is no macOS/Windows equivalent to fall back to."
    )
_missing_bins = [b for b in ("chrt", "nice") if shutil.which(b) is None]
if _missing_bins:
    sys.exit(
        f"Required command(s) not found on PATH: {', '.join(_missing_bins)}. "
        "Install util-linux (chrt) / coreutils (nice), typically already "
        "present on most Linux distributions."
    )

WINDOW_S = 2.0
POLL_S = 0.25
TOTAL_S = 46.0
RT_START = 12.0
RT_DURATION = 16.0  # RT hog active from t=12 to t=28

SCENARIO = sys.argv[1] if len(sys.argv) > 1 else "rt"          # "rt" or "nice"
TRIAL = sys.argv[2] if len(sys.argv) > 2 else "1"
OUT_CSV = str(DATA_DIR / f"real_data_{SCENARIO}_trial{TRIAL}.csv")


def read_schedstat(pid):
    try:
        with open(f"/proc/{pid}/schedstat") as f:
            parts = f.read().split()
        return int(parts[0]), int(parts[1]), int(parts[2])  # run_ns, wait_ns, nr_switches
    except (FileNotFoundError, ProcessLookupError, ValueError, IndexError):
        return None


def read_stat_priority_nice(pid):
    try:
        with open(f"/proc/{pid}/stat") as f:
            raw = f.read()
        after = raw.rsplit(")", 1)[1].split()
        # fields after comm: state(0) ppid(1) pgrp(2) ... priority is index 15, nice is index 16
        priority = int(after[15])
        nice = int(after[16])
        return priority, nice
    except (FileNotFoundError, ProcessLookupError, ValueError, IndexError):
        return None, None


def read_ctxt_switches(pid):
    try:
        with open(f"/proc/{pid}/status") as f:
            text = f.read()
        vol = int(re.search(r"voluntary_ctxt_switches:\s*(\d+)", text).group(1))
        nonvol = int(re.search(r"nonvoluntary_ctxt_switches:\s*(\d+)", text).group(1))
        return vol, nonvol
    except (FileNotFoundError, ProcessLookupError, AttributeError, ValueError):
        return None, None


def read_mem_threads_state(pid):
    try:
        with open(f"/proc/{pid}/status") as f:
            text = f.read()
        rss = int(re.search(r"VmRSS:\s*(\d+)", text).group(1))
        threads = int(re.search(r"Threads:\s*(\d+)", text).group(1))
        state_char = re.search(r"State:\s*(\S)", text).group(1)
        return rss, threads, state_char
    except (FileNotFoundError, ProcessLookupError, AttributeError, ValueError):
        return 0, 1, "S"


def total_mem_kb():
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemTotal:"):
                    return int(line.split()[1])
    except FileNotFoundError:
        pass
    return 1


def runnable_queue_length():
    try:
        with open("/proc/loadavg") as f:
            parts = f.read().split()
        runnable, _total = parts[3].split("/")
        return int(runnable)
    except (FileNotFoundError, ValueError, IndexError):
        return 0


MEM_TOTAL_KB = total_mem_kb()


def phase_at(t):
    if t < RT_START:
        return "baseline"
    if t < RT_START + RT_DURATION:
        return "rt_contention"
    return "recovery"


def main():
    procs = {}  # label -> Popen

    procs["victim_normal"] = subprocess.Popen(
        ["python3", str(BASE_DIR / "workload_cpu.py"), str(TOTAL_S)]
    )
    procs["victim_lowprio"] = subprocess.Popen(
        ["nice", "-n", "19", "python3", str(BASE_DIR / "workload_cpu.py"), str(TOTAL_S)]
    )
    procs["io_bound_a"] = subprocess.Popen(
        ["python3", str(BASE_DIR / "workload_io.py"), str(TOTAL_S)]
    )
    procs["io_bound_b"] = subprocess.Popen(
        ["python3", str(BASE_DIR / "workload_io.py"), str(TOTAL_S)]
    )

    labels = {p.pid: name for name, p in procs.items()}
    process_type = {
        "victim_normal": "cpu_bound",
        "victim_lowprio": "cpu_bound",
        "io_bound_a": "io_bound",
        "io_bound_b": "io_bound",
        "hog_rt": "cpu_bound",
    }

    hog_proc = None
    hog_launched = False

    last_snapshot = {}  # pid -> (run_ns, wait_ns, nr_switches, vol, nonvol, t)
    window_start = {}   # pid -> snapshot at start of current 2s window
    rows = []

    t0 = time.time()
    next_window_boundary = WINDOW_S

    while True:
        now = time.time() - t0
        if now > TOTAL_S:
            break

        if not hog_launched and now >= RT_START:
            if SCENARIO == "rt":
                hog_proc = subprocess.Popen(
                    ["chrt", "-f", "50", "python3", str(BASE_DIR / "workload_cpu.py"), str(RT_DURATION)]
                )
                time.sleep(0.3)  # give chrt+exec a moment before checking
                rt_verified = False
                if hog_proc.poll() is not None:
                    print(
                        f"WARNING: 'chrt -f 50' exited immediately (code {hog_proc.returncode}). "
                        "This usually means the process lacks CAP_SYS_NICE / real-time "
                        "scheduling privileges, so NO genuine SCHED_FIFO preemption occurred "
                        "in this trial. Do not report this trial's rt_contention-phase rows "
                        "as a verified real-time starvation episode.",
                        file=sys.stderr,
                    )
                else:
                    try:
                        policy = os.sched_getscheduler(hog_proc.pid)
                        rt_verified = (policy == os.SCHED_FIFO)
                        if not rt_verified:
                            print(
                                f"WARNING: hog process pid={hog_proc.pid} has scheduling "
                                f"policy {policy}, not SCHED_FIFO ({os.SCHED_FIFO}). "
                                "chrt may not have applied as expected.",
                                file=sys.stderr,
                            )
                    except (ProcessLookupError, PermissionError, AttributeError) as e:
                        print(f"WARNING: could not verify scheduling policy: {e}", file=sys.stderr)
                with open(OUT_CSV.replace(".csv", "_rt_verification.json"), "w") as vf:
                    json.dump({"rt_policy_verified": rt_verified}, vf)
            else:  # "nice" scenario: strong CFS priority skew, no real-time preemption
                hog_proc = subprocess.Popen(
                    ["nice", "-n", "-15", "python3", str(BASE_DIR / "workload_cpu.py"), str(RT_DURATION)]
                )
            procs["hog_rt"] = hog_proc
            labels[hog_proc.pid] = "hog_rt"
            hog_launched = True

        live_pids = [p.pid for p in procs.values() if p.poll() is None]

        for pid in live_pids:
            ss = read_schedstat(pid)
            if ss is None:
                continue
            run_ns, wait_ns, nr_sw = ss
            vol, nonvol = read_ctxt_switches(pid)
            snap = (run_ns, wait_ns, nr_sw, vol or 0, nonvol or 0, now)
            last_snapshot[pid] = snap
            window_start.setdefault(pid, snap)

        if now >= next_window_boundary:
            for pid, start_snap in list(window_start.items()):
                end_snap = last_snapshot.get(pid)
                if end_snap is None:
                    continue
                run_delta_ns = end_snap[0] - start_snap[0]
                wait_delta_ns = end_snap[1] - start_snap[1]
                sw_delta = end_snap[2] - start_snap[2]
                vol_delta = end_snap[3] - start_snap[3]
                nonvol_delta = end_snap[4] - start_snap[4]
                window_dur_ms = (end_snap[5] - start_snap[5]) * 1000
                if window_dur_ms <= 0:
                    continue

                priority, nice = read_stat_priority_nice(pid)
                label = labels.get(pid, "unknown")
                rss_kb, threads, state_char = read_mem_threads_state(pid)
                queue_len = runnable_queue_length()

                burst_ms = run_delta_ns / 1e6
                wait_ms = wait_delta_ns / 1e6
                io_wait_ms = max(window_dur_ms - burst_ms - wait_ms, 0.0)

                is_starved = 0
                if process_type.get(label) == "cpu_bound":
                    if burst_ms < 0.05 * window_dur_ms and wait_ms > 0.5 * window_dur_ms:
                        is_starved = 1

                rows.append({
                    "t_window_end": round(now, 2),
                    "phase": phase_at(now),
                    "pid": pid,
                    "process_label": label,
                    "process_type": process_type.get(label, "unknown"),
                    "priority": priority,
                    "nice": nice,
                    "window_ms": round(window_dur_ms, 1),
                    "burst_time_ms": round(burst_ms, 2),
                    "waiting_time_ms": round(wait_ms, 2),
                    "io_wait_time_ms": round(io_wait_ms, 2),
                    "context_switches": sw_delta,
                    "voluntary_ctxt_switches": vol_delta,
                    "nonvoluntary_ctxt_switches": nonvol_delta,
                    "rss_kb": rss_kb,
                    "num_threads": threads,
                    "state_char": state_char,
                    "queue_length_at_arrival": queue_len,
                    "is_starved": is_starved,
                    "mem_total_kb": MEM_TOTAL_KB,
                })

            window_start = {pid: last_snapshot[pid] for pid in last_snapshot}
            next_window_boundary += WINDOW_S

        time.sleep(POLL_S)

    for p in procs.values():
        if p.poll() is None:
            p.terminate()
    for p in procs.values():
        try:
            p.wait(timeout=5)
        except Exception:
            p.kill()

    with open(OUT_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print(f"Collected {len(rows)} real windowed observations across "
          f"{len(set(r['pid'] for r in rows))} real processes.")
    print(f"Saved to {OUT_CSV}")


if __name__ == "__main__":
    main()
