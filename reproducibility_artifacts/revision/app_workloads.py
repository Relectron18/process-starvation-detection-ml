"""Application-style workloads for the extended trials (stdlib only).
usage: app_workloads.py {zlib|sqlite|http} seconds"""
import os,sys,time,zlib,sqlite3,tempfile,threading,socket,http.server
kind,dur=sys.argv[1],float(sys.argv[2]); end=time.time()+dur
if kind=="zlib":
    blob=os.urandom(1<<20)
    while time.time()<end: zlib.compress(blob,9)
elif kind=="sqlite":
    d=tempfile.mkdtemp();c=sqlite3.connect(d+"/t.db");c.execute("create table t(a int,b text)")
    i=0
    while time.time()<end:
        c.execute("insert into t values(?,?)",(i,"x"*200));c.commit();i+=1
        if i%200==0: c.execute("select count(*),sum(a) from t").fetchone()
        time.sleep(0.005)
elif kind=="http":
    class H(http.server.BaseHTTPRequestHandler):
        def do_GET(s):
            b=b"y"*4096;s.send_response(200);s.send_header("Content-Length",str(len(b)));s.end_headers();s.wfile.write(b)
        def log_message(s,*a):pass
    sv=http.server.ThreadingHTTPServer(("127.0.0.1",0),H);p=sv.server_address[1]
    threading.Thread(target=sv.serve_forever,daemon=True).start()
    import urllib.request
    while time.time()<end:
        urllib.request.urlopen(f"http://127.0.0.1:{p}/").read();time.sleep(0.02)
