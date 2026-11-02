"""Extended collector: same sampling and label as collect_real_data.py
(2 s windows, 0.25 s polling, 46 s trial, disturbance t=12..28), new scenarios.
usage: collect_ext.py <scenario> <trial> <outdir>"""
import csv,os,re,subprocess,sys,time
sys.path.insert(0,os.path.dirname(os.path.abspath(__file__)))
H=os.path.dirname(os.path.abspath(__file__))
SC,TR,OUT=sys.argv[1],sys.argv[2],sys.argv[3]
WINDOW,POLL,TOTAL,T0,DUR=2.0,0.25,46.0,12.0,16.0
PIN=None if SC=="multicore_rt" else "0"
if PIN: os.sched_setaffinity(0,{0})
PY=sys.executable
cpu=lambda d:[PY,H+"/workload_cpu.py",str(d)]; io=lambda d:[PY,H+"/workload_io.py",str(d)]
app=lambda k,d:[PY,H+"/app_workloads.py",k,str(d)]
def rd(p):
    with open(p) as f: return f.read()
def schedstat(pid):
    try: a=rd(f"/proc/{pid}/schedstat").split(); return int(a[0]),int(a[1]),int(a[2])
    except (OSError,ValueError,IndexError): return None
def stat(pid):
    try:
        x=rd(f"/proc/{pid}/stat").rsplit(")",1)[1].split(); return int(x[15]),int(x[16])
    except (OSError,ValueError,IndexError): return None,None
def status(pid):
    try:
        t=rd(f"/proc/{pid}/status")
        g=lambda k:re.search(k,t)
        return (int(g(r"\bvoluntary_ctxt_switches:\s*(\d+)").group(1)),int(g(r"nonvoluntary_ctxt_switches:\s*(\d+)").group(1)),
                int(g(r"VmRSS:\s*(\d+)").group(1)),int(g(r"Threads:\s*(\d+)").group(1)),g(r"State:\s*(\S)").group(1))
    except (OSError,AttributeError,ValueError): return 0,0,0,1,"S"
def qlen():
    try: return int(rd("/proc/loadavg").split()[3].split("/")[0])
    except (OSError,ValueError,IndexError): return 0
MEM=int(re.search(r"MemTotal:\s*(\d+)",rd("/proc/meminfo")).group(1))
procs={};ptype={}
def spawn(name,cmd,t,prefix=()):
    procs[name]=subprocess.Popen(list(prefix)+cmd,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL);ptype[name]=t
if SC in("app_rt",):
    spawn("victim_normal",app("zlib",TOTAL),"cpu_bound")
    spawn("victim_lowprio",app("zlib",TOTAL),"cpu_bound",("nice","-n","19"))
    spawn("io_bound_a",app("sqlite",TOTAL),"io_bound");spawn("io_bound_b",app("http",TOTAL),"io_bound")
else:
    spawn("victim_normal",cpu(TOTAL),"cpu_bound")
    spawn("victim_lowprio",cpu(TOTAL),"cpu_bound",("nice","-n","19"))
    spawn("io_bound_a",io(TOTAL),"io_bound");spawn("io_bound_b",io(TOTAL),"io_bound")
if SC=="multicore_rt":
    # one extra normal competitor so the second CPU is also busy
    spawn("co_runner",cpu(TOTAL),"cpu_bound")
CG="/sys/fs/cgroup/starv"
def disturb():
    if SC in("rt","app_rt"): spawn("hog_rt",["chrt","-f","50"]+cpu(DUR),"cpu_bound")
    elif SC=="nice": spawn("hog_rt",["nice","-n","-15"]+cpu(DUR),"cpu_bound")
    elif SC=="overcommit":
        for i in range(24): spawn(f"hog_{i}",cpu(DUR),"cpu_bound")
    elif SC=="cgroup":
        os.makedirs(CG,exist_ok=True);open(CG+"/cpu.max","w").write("2000 100000")
        open(CG+"/cgroup.procs","w").write(str(procs["victim_normal"].pid))
    elif SC=="multicore_rt":
        # two RT hogs on two vCPUs
        spawn("hog_rt",["chrt","-f","50"]+cpu(DUR),"cpu_bound");spawn("hog_rt2",["chrt","-f","50"]+cpu(DUR),"cpu_bound")
    else: raise SystemExit("bad scenario")
def undisturb():
    if SC=="cgroup":
        open("/sys/fs/cgroup/cgroup.procs","w").write(str(procs["victim_normal"].pid))
phase=lambda t:"baseline" if t<T0 else ("rt_contention" if t<T0+DUR else "recovery")
start=time.time();nxt=WINDOW;last={};w0={};rows=[];done=False;off=False
while True:
    now=time.time()-start
    if now>TOTAL: break
    if not done and now>=T0: disturb();done=True
    if done and not off and now>=T0+DUR: undisturb();off=True
    live=[(n,p.pid) for n,p in procs.items() if p.poll() is None]
    for n,pid in live:
        s=schedstat(pid)
        if s is None: continue
        v,nv,_,_,_=status(pid);snap=(*s,v,nv,now);last[pid]=(n,snap);w0.setdefault(pid,snap)
    if now>=nxt:
        for pid,st in list(w0.items()):
            if pid not in last: continue
            n,e=last[pid];dur=(e[5]-st[5])*1000
            if dur<=0: continue
            b=(e[0]-st[0])/1e6;w=(e[1]-st[1])/1e6;pr,ni=stat(pid);_,_,rss,th,ch=status(pid)
            t=ptype[n];lowc=b<.05*dur;hiw=w>.5*dur
            rows.append(dict(t_window_end=round(now,2),phase=phase(now),pid=pid,process_label=n,process_type=t,priority=pr,nice=ni,window_ms=round(dur,1),burst_time_ms=round(b,2),waiting_time_ms=round(w,2),io_wait_time_ms=round(max(dur-b-w,0),2),context_switches=e[2]-st[2],voluntary_ctxt_switches=e[3]-st[3],nonvoluntary_ctxt_switches=e[4]-st[4],rss_kb=rss,num_threads=th,state_char=ch,queue_length_at_arrival=qlen(),is_starved=int(t=="cpu_bound" and lowc and hiw),is_starved_any=int(lowc and hiw),mem_total_kb=MEM))
        w0={p:last[p][1] for p in last};nxt+=WINDOW
    time.sleep(POLL)
for p in procs.values():
    if p.poll() is None: p.terminate()
for p in procs.values():
    try: p.wait(timeout=5)
    except Exception: p.kill()
if SC=="cgroup":
    try: os.rmdir(CG)
    except OSError: pass
with open(f"{OUT}/ext_{SC}_trial{TR}.csv","w",newline="") as f:
    w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
print(SC,TR,len(rows),sum(r["is_starved"] for r in rows))
