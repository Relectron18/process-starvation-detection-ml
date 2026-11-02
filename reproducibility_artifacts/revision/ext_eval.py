import sys,glob,json,os,time,warnings
A,RES,OUT=sys.argv[1:4]; sys.path.insert(0,A); os.environ['STARVATION_OUTPUT_DIR']=OUT
import joblib,numpy as np,pandas as pd
from sklearn.metrics import f1_score,precision_score,recall_score,roc_auc_score
from real_data_features import score_raw_csv
warnings.filterwarnings('ignore')
M=[('models','seed42'),('models_seed7_matched','seed7'),('models_seed123_matched','seed123')]
files=[(f,'orig_'+os.path.basename(f).split('_')[2],int(f[-5])) for f in sorted(glob.glob(f'{A}/real_data_*_trial*.csv'))]+[(f,os.path.basename(f).split('_trial')[0][4:],int(f[-5])) for f in sorted(glob.glob(f'{RES}/ext_*_trial*.csv'))]
rows=[];all_=[]
for md,name in M:
    d=f'{A}/{md}';m=joblib.load(f'{d}/best_model.pkl');meta=json.load(open(f'{d}/run_metadata.json'));t=meta['recommended_threshold']
    sc=joblib.load(f'{d}/scaler.pkl');fc=joblib.load(f'{d}/feature_columns.pkl');q=joblib.load(f'{d}/queueing_stats.pkl')
    enc={c:joblib.load(f'{d}/encoder_{c}.pkl') for c in ['process_type','process_state']}
    for f,s,tr in files:
        e=score_raw_csv(f,m,sc,enc,fc,q,t);e['scenario']=s;e['trial']=tr;e['model']=name;e['thr']=t;all_.append(e)
    if name=='seed42':
        X=pd.DataFrame(np.zeros((1,len(fc))),columns=fc).values;ts=[]
        for _ in range(200):
            a=time.perf_counter();m.predict_proba(X);ts.append((time.perf_counter()-a)*1e3)
        B=np.zeros((5,len(fc)));tb=[]
        for _ in range(100):
            a=time.perf_counter();m.predict_proba(B);tb.append((time.perf_counter()-a)*1e3)
        print('inference ms single p50/p99',np.percentile(ts,50).round(2),np.percentile(ts,99).round(2),'batch5 p50',np.percentile(tb,50).round(2))
        m.set_params(n_jobs=1) if hasattr(m,'n_jobs') else 0
        ts=[]
        for _ in range(200):
            a=time.perf_counter();m.predict_proba(X);ts.append((time.perf_counter()-a)*1e3)
        print('n_jobs=1 single p50/p99',np.percentile(ts,50).round(2),np.percentile(ts,99).round(2),'model file MB',round(os.path.getsize(f'{d}/best_model.pkl')/1e6,1))
E_=pd.concat(all_,ignore_index=True);E_.to_csv(f'{OUT}/ext_scored.csv',index=False)
base=E_[E_.model=='seed42'].copy();base['r']=(base.waiting_time_ms+base.burst_time_ms)/(base.burst_time_ms+1e-6);base['ws']=base.waiting_time_ms/base.window_ms
# (1) operational label by scenario
def block(g,score,pred):
    r=[]
    for _,h in g.groupby('trial'):
        y=h.is_starved
        r.append((f1_score(y,h[pred],zero_division=0),precision_score(y,h[pred],zero_division=0),recall_score(y,h[pred],zero_division=0),roc_auc_score(y,h[score]) if 0<y.sum()<len(y) else np.nan,y.sum()))
    return np.nanmean(np.array(r,float),0)
R=[]
base['p_h']=(base.r>=2).astype(int);base['p_w']=(base.ws>.5).astype(int)
for s,g in base.groupby('scenario'):
    a=block(g,'r','p_h');b=block(g,'ws','p_w');R.append(dict(s=s,det='HRRN',f1=a[0],p=a[1],r=a[2],auc=a[3],pos=a[4]));R.append(dict(s=s,det='wait>0.5',f1=b[0],p=b[1],r=b[2],auc=b[3],pos=b[4]))
for mn,gm in E_.groupby('model'):
    for s,g in gm.groupby('scenario'):
        a=block(g,'model_proba','model_pred');R.append(dict(s=s,det=mn,f1=a[0],p=a[1],r=a[2],auc=a[3],pos=a[4]))
R=pd.DataFrame(R);R.to_csv(f'{OUT}/ext_oper.csv',index=False);print(R.pivot(index='s',columns='det',values='f1').round(3).to_string());print(R.pivot(index='s',columns='det',values='p').round(3).to_string());print(R.pivot(index='s',columns='det',values='r').round(3).to_string());print(R.pivot(index='s',columns='det',values='auc').round(3).to_string());print(R[R.det=='seed42'][['s','pos']].round(1).to_string())
# (2) design (episode) label on victim_normal: y=1 window wholly inside disturbance (t_end in [14,28]), y=0 t_end<=12 or >=32
v=E_[E_.process_label=='victim_normal'].copy();v=v[(v.t_window_end<=12.5)|(v.t_window_end.between(14.0,28.2))|(v.t_window_end>=32)]
v['yd']=v.t_window_end.between(14.0,28.2).astype(int);v['ws']=v.waiting_time_ms/v.window_ms;v['r']=(v.waiting_time_ms+v.burst_time_ms)/(v.burst_time_ms+1e-6)
D=[]
for (mn,s),g in v.groupby(['model','scenario']):
    D.append(dict(det=mn,s=s,n1=g.yd.sum(),n0=(1-g.yd).sum(),hit=g[g.yd==1].model_pred.mean(),fa=g[g.yd==0].model_pred.mean(),auc=roc_auc_score(g.yd,g.model_proba),meanp1=g[g.yd==1].model_proba.mean()))
g0=v[v.model=='seed42']
for s,g in g0.groupby('scenario'):
    D.append(dict(det='wait>0.5',s=s,n1=g.yd.sum(),n0=(1-g.yd).sum(),hit=(g[g.yd==1].ws>.5).mean(),fa=(g[g.yd==0].ws>.5).mean(),auc=roc_auc_score(g.yd,g.ws)))
    D.append(dict(det='HRRN',s=s,n1=g.yd.sum(),n0=(1-g.yd).sum(),hit=(g[g.yd==1].r>=2).mean(),fa=(g[g.yd==0].r>=2).mean(),auc=roc_auc_score(g.yd,g.r)))
D=pd.DataFrame(D);D.to_csv(f'{OUT}/ext_design.csv',index=False);print(D.pivot(index='s',columns='det',values='hit').round(3).to_string());print(D.pivot(index='s',columns='det',values='fa').round(3).to_string());print(D.pivot(index='s',columns='det',values='auc').round(3).to_string());print(D[['s','n1','n0']].drop_duplicates().to_string())
print(v[v.model=='seed42'].groupby(['scenario','yd'])[['burst_time_ms','waiting_time_ms']].mean().round(0).to_string())
print(E_[E_.model=='seed42'].groupby(['scenario','process_label']).is_starved.agg(['size','sum']).to_string())
# (3) leave-one-scenario-out threshold, seed42, operational label
ts=np.linspace(.02,.98,97);L=[]
for mn,gm in E_.groupby('model'):
    for s,g in gm.groupby('scenario'):
        o=gm[gm.scenario!=s];th=ts[int(np.argmax([f1_score(o.is_starved,o.model_proba>=x) for x in ts]))]
        p=(g.model_proba>=th).astype(int);L.append(dict(model=mn,s=s,th=th,f1=f1_score(g.is_starved,p),p=precision_score(g.is_starved,p,zero_division=0),r=recall_score(g.is_starved,p)))
L=pd.DataFrame(L);L.to_csv(f'{OUT}/ext_loso.csv',index=False);print(L.pivot(index='s',columns='model',values='f1').round(3).to_string());print(L.pivot(index='s',columns='model',values='th').round(2).to_string())
