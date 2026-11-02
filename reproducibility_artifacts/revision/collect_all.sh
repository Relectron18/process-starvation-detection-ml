#!/bin/bash
# usage: sudo run_all.sh <outdir>   (root needed for SCHED_FIFO, nice<0, cgroups)
set -u; OUT=$1; mkdir -p "$OUT"; K=$(dirname "$(readlink -f "$0")"); PY=${PY:-$K/venv/bin/python}
uname -a > "$OUT/env.txt"; nproc >> "$OUT/env.txt"; cat /proc/sys/kernel/sched_rt_runtime_us >> "$OUT/env.txt"; grep -c . /proc/1/schedstat >> "$OUT/env.txt"; "$PY" -V >> "$OUT/env.txt" 2>&1
echo +cpu > /sys/fs/cgroup/cgroup.subtree_control 2>/dev/null
for s in rt nice app_rt overcommit cgroup multicore_rt; do for t in 1 2 3 4 5; do
  "$PY" "$K/collect_ext.py" $s $t "$OUT" || echo FAIL $s $t; sleep 3
done; done
"$PY" "$K/overhead.py" "$OUT"
echo ALLDONE > "$OUT/DONE"
