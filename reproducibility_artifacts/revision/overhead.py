"""Telemetry overhead. Pins everything to CPU0.
Parts: (a) per-PID read cost; (b) collector CPU share vs tracked PIDs/poll rate;
(c) throughput cost to a CPU-bound task; (d) model inference latency."""
import csv,os,random,re,resource,subprocess,sys,time,json
OUT=sys.argv[1];H=os.path.dirname(os.path.abspath(__file__));os.sched_setaffinity(0,{0})
def rd(p):
    with open(p) as f:return f.read()
def sample(pid):
    a=rd(f"/proc/{pid}/schedstat").split();t=rd(f"/proc/{pid}/status");re.search(r"nonvoluntary_ctxt_switches:\s*(\d+)",t);re.search(r"\bvoluntary_ctxt_switches:\s*(\d+)",t);rd(f"/proc/{pid}/stat")
def sleepers(n): return [subprocess.Popen(["sleep","600"]) for _ in range(n)]
def stop(ps):
    for p in ps:p.kill()
    for p in ps:p.wait()
res=[]
# (a)
for n in (5,20,100,500):
    ps=sleepers(n);pids=[p.pid for p in ps];ts=[]
    for _ in range(max(20,4000//n)):
        t=time.perf_counter()
        for x in pids:sample(x)
        ts.append((time.perf_counter()-t)/n*1e6)
    stop(ps);ts.sort();res.append(dict(part="read_us_per_pid",n=n,poll=0,mean=sum(ts)/len(ts),p50=ts[len(ts)//2],p99=ts[int(.99*len(ts))]))
# (b) collector-only process CPU time over 20 s
code='''
import sys,time,re,os
sys.path.insert(0,%r);
def rd(p):
    with open(p) as f:return f.read()
pids=[int(x) for x in sys.argv[2:]];poll=float(sys.argv[1]);end=time.time()+20
while time.time()<end:
    for p in pids:
        rd(f"/proc/{p}/schedstat");t=rd(f"/proc/{p}/status");re.search(r"nonvoluntary_ctxt_switches:\\s*(\\d+)",t)
    time.sleep(poll)
print(os.times()[0]+os.times()[1])
'''%H
for n in (5,20,100,500):
  for poll in (0.25,1.0):
    ps=sleepers(n);rs=[]
    for _ in range(3):
        out=subprocess.check_output([sys.executable,"-c",code,str(poll)]+[str(p.pid) for p in ps]).decode().strip();rs.append(float(out)/20*100)
    stop(ps);res.append(dict(part="collector_cpu_pct",n=n,poll=poll,mean=sum(rs)/3,p50=sorted(rs)[1],p99=max(rs)))
# (c) paired throughput, randomized order: fixed-work loop timed with/without collector
work=[sys.executable,"-c","import time\nt=time.perf_counter();x=0\nfor i in range(60000000):x+=i\nprint(time.perf_counter()-t)"]
bg=sleepers(4);rows=[]
for rep in range(15):
    order=["off","on_5","on_100"];random.Random(rep).shuffle(order)
    for c in order:
        col=None
        if c!="off":
            n=5 if c=="on_5" else 100
            extra=sleepers(0)
            pids=[p.pid for p in bg]*(n//4+1)
            col=subprocess.Popen([sys.executable,"-c",code.replace("end=time.time()+20","end=time.time()+600"),"0.25"]+[str(p) for p in pids[:n]],stdout=subprocess.DEVNULL)
        v=float(subprocess.check_output(work).decode());rows.append(dict(rep=rep,cond=c,sec=v))
        if col: col.kill();col.wait()
stop(bg)
with open(f"{OUT}/overhead_throughput.csv","w",newline="") as f:
    w=csv.DictWriter(f,fieldnames=["rep","cond","sec"]);w.writeheader();w.writerows(rows)
with open(f"{OUT}/overhead.csv","w",newline="") as f:
    w=csv.DictWriter(f,fieldnames=["part","n","poll","mean","p50","p99"]);w.writeheader();w.writerows(res)
print(res)
