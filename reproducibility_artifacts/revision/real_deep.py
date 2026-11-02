import sys,json,glob,os
A=sys.argv[1]; OUT=sys.argv[2]; sys.path.insert(0,A); os.environ['STARVATION_OUTPUT_DIR']=OUT
import joblib,pandas as pd,numpy as np
import sklearn._loss._loss as _L; sys.modules['_loss']=_L
from sklearn.metrics import f1_score,precision_score,recall_score,roc_auc_score
from real_data_features import score_raw_csv
files=sorted(glob.glob(f'{A}/real_data_*_trial*.csv'))
res={}
for md,name in [('models','seed42'),('models_seed7_matched','seed7'),('models_seed123_matched','seed123')]:
    d=f'{A}/{md}';m=joblib.load(f'{d}/best_model.pkl');meta=json.load(open(f'{d}/run_metadata.json'))
    sc=joblib.load(f'{d}/scaler.pkl');fc=joblib.load(f'{d}/feature_columns.pkl');q=joblib.load(f'{d}/queueing_stats.pkl')
    enc={c:joblib.load(f'{d}/encoder_{c}.pkl') for c in ['process_type','process_state']}
    fr=[]
    for f in files:
        b=os.path.basename(f)[:-4].split('_');e=score_raw_csv(f,m,sc,enc,fc,q,meta['recommended_threshold']);e['scenario']=b[2];e['trial']=int(b[3][5:]);fr.append(e)
    res[name]=(pd.concat(fr,ignore_index=True),meta['recommended_threshold'])
for n,(e,t) in res.items():
    vn=e[(e.process_label=='victim_normal')]
    print(n,'thr',t,'victim_normal P(starved): pos mean/min/max',vn[vn.is_starved==1].model_proba.agg(['mean','min','max']).round(3).tolist(),'neg mean/max',vn[vn.is_starved==0].model_proba.agg(['mean','max']).round(3).tolist(),'hits',int(vn[vn.is_starved==1].model_pred.sum()),'of',int(vn.is_starved.sum()))
    # within-process AUC
    for p in ['victim_normal','victim_lowprio']:
        g=e[e.process_label==p]; print('   AUC within',p,round(roc_auc_score(g.is_starved,g.model_proba),4),'n',len(g),'pos',int(g.is_starved.sum()))
    # alt label: drop cpu_bound gate
    y2=((e.burst_time_ms<.05*e.window_ms)&(e.waiting_time_ms>.5*e.window_ms)).astype(int)
    e['y_all']=y2
    for s,g in e.groupby('scenario'):
        r=[(f1_score(h.y_all,h.model_pred),precision_score(h.y_all,h.model_pred),recall_score(h.y_all,h.model_pred),h.y_all.sum()) for _,h in g.groupby('trial')]
        r=np.array(r);print('   alt label',s,'F1 %.3f±%.3f P %.3f±%.3f R %.3f±%.3f pos %.1f'%(r[:,0].mean(),r[:,0].std(ddof=1),r[:,1].mean(),r[:,1].std(ddof=1),r[:,2].mean(),r[:,2].std(ddof=1),r[:,3].mean()))
    # excl. chronic nice19: metrics on all processes but victim_lowprio
    x=e[e.process_label!='victim_lowprio']
    for s,g in x.groupby('scenario'):
        r=np.array([(f1_score(h.is_starved,h.model_pred,zero_division=0),precision_score(h.is_starved,h.model_pred,zero_division=0),recall_score(h.is_starved,h.model_pred)) for _,h in g.groupby('trial')])
        print('   excl lowprio',s,'F1 %.3f P %.3f R %.3f'%tuple(r.mean(0)),'auc',round(roc_auc_score(g.is_starved,g.model_proba),3))
# leave-one-trial-out threshold calibration on real windows, seed 42
e=res['seed42'][0];out=[]
ts=np.linspace(.02,.98,97)
for (s,t),g in e.groupby(['scenario','trial']):
    o=e[~((e.scenario==s)&(e.trial==t))]
    th=ts[int(np.argmax([f1_score(o.is_starved,o.model_proba>=x) for x in ts]))]
    p=(g.model_proba>=th).astype(int);out.append((s,t,th,f1_score(g.is_starved,p),precision_score(g.is_starved,p),recall_score(g.is_starved,p)))
L=pd.DataFrame(out,columns='s t th f1 p r'.split());print(L.round(3).to_string());print(L.groupby('s')[['th','f1','p','r']].agg(['mean','std']).round(3).to_string())
# rep trial for fig 3
g=e[(e.scenario=='rt')&(e.process_label=='victim_normal')&(e.trial==1)][['t_window_end','phase','burst_time_ms','waiting_time_ms','is_starved','model_proba','model_pred']];print(g.round(3).to_string())
