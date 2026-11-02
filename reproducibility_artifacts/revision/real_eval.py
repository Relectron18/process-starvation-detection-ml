"""Real-hardware evidence: repro of stored models, per-process recall, baselines."""
import sys,json,warnings,glob,os
A=sys.argv[1]; OUT=sys.argv[2]; sys.path.insert(0,A); os.environ['STARVATION_OUTPUT_DIR']=OUT
import joblib,pandas as pd,numpy as np,sklearn
import sklearn._loss._loss as _L; sys.modules['_loss']=_L
from sklearn.metrics import f1_score,precision_score,recall_score,roc_auc_score,accuracy_score,average_precision_score
from sklearn.ensemble import IsolationForest
from real_data_features import score_raw_csv
print('sklearn',sklearn.__version__)
files=sorted(glob.glob(f'{A}/real_data_*_trial*.csv'))
def info(f):
    b=os.path.basename(f)[:-4].split('_'); return b[2],int(b[3][5:])
allscored={}
ref=pd.read_csv(f'{A}/multiseed_real_world_results.csv')
for md,name in [('models','seed42'),('models_seed7_matched','seed7'),('models_seed123_matched','seed123')]:
    d=f'{A}/{md}'
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter('always'); m=joblib.load(f'{d}/best_model.pkl')
    meta=json.load(open(f'{d}/run_metadata.json'))
    print(name,type(m).__name__,{k:v for k,v in m.get_params().items() if k in('n_estimators','max_depth','min_samples_split','min_samples_leaf','learning_rate','subsample','colsample_bytree','class_weight')},'thr',meta['recommended_threshold'],meta['best_model'],meta['metrics'],'unpickle warnings',len(w))
    sc=joblib.load(f'{d}/scaler.pkl');fc=joblib.load(f'{d}/feature_columns.pkl');q=joblib.load(f'{d}/queueing_stats.pkl')
    enc={c:joblib.load(f'{d}/encoder_{c}.pkl') for c in ['process_type','process_state']}
    fr=[]
    for f in files:
        s,t=info(f); e=score_raw_csv(f,m,sc,enc,fc,q,meta['recommended_threshold']); e['scenario']=s;e['trial']=t; fr.append(e)
        r=ref[(ref.model==name)&(ref.scenario==s)&(ref.trial==t)].iloc[0]
        f1=f1_score(e.is_starved,e.model_pred)
        if abs(f1-r.f1)>1e-9: print('  MISMATCH',name,s,t,f1,r.f1)
    allscored[name]=(pd.concat(fr,ignore_index=True),m,sc,enc,fc,meta)
    print('  repro check done')
base=allscored['seed42'][0]
print('label positives by process\n',base.groupby(['process_label','phase']).is_starved.agg(['size','sum']).to_string())
print('positives by proc', base.groupby('process_label').is_starved.sum().to_dict(), 'total',base.is_starved.sum())
print('windows with wait>50% by type/label'); print(base.assign(hw=base.waiting_time_ms>0.5*base.window_ms).groupby(['process_type','is_starved']).hw.agg(['size','sum']).to_string())
print('min cpu share of unstarved io windows in rt contention:'); x=base[(base.process_type=='io_bound')&(base.phase=='rt_contention')&(base.scenario=='rt')]; print(((x.burst_time_ms/x.window_ms)<.05).mean(), (x.waiting_time_ms>.5*x.window_ms).mean(), len(x))
# per-process recall for each model
rows=[]
for n,(e,*_) in allscored.items():
    for (s,l),g in e[e.is_starved==1].groupby(['scenario','process_label']):
        rows.append((n,s,l,len(g),g.model_pred.mean()))
    # transient victim_normal only vs chronic
pr=pd.DataFrame(rows,columns='model scenario proc n_pos recall'.split());print(pr.to_string());pr.to_csv(f'{OUT}/recall_by_process.csv',index=False)
# false positives by process
print(base[(base.is_starved==0)&(base.model_pred==1)].groupby(['scenario','process_label','phase']).size())
for n in ['seed7','seed123']:
    e=allscored[n][0]; print(n,'FP:',e[(e.is_starved==0)&(e.model_pred==1)].groupby(['scenario','process_label','phase']).size().to_dict())
# baselines. score + binary rule, evaluated trial-wise
e=base.copy()
r=(e.waiting_time_ms+e.burst_time_ms)/(e.burst_time_ms+1e-6)
e['wshare']=e.waiting_time_ms/e.window_ms; e['cshare']=e.burst_time_ms/e.window_ms
# unsupervised: per-process z-score vs baseline phase of that trial
zs=[]
for (s,t,l),g in e.groupby(['scenario','trial','process_label']):
    b=g[g.phase=='baseline'].wshare
    if len(b)>=3: z=(g.wshare-b.mean())/(b.std()+1e-3)
    else: z=pd.Series(0.0,index=g.index)
    zs.append(z)
e['z']=pd.concat(zs)
# isolation forest on baseline windows pooled per trial (leave-trial-out)
feat=['wshare','cshare','context_switches','queue_length_at_arrival']
e['iso']=0.0;e['iso_pred']=0
for (s,t),g in e.groupby(['scenario','trial']):
    tr=e[(e.phase=='baseline')&~((e.scenario==s)&(e.trial==t))]
    tr=tr[tr.process_label!='victim_lowprio'] if False else tr
    I=IsolationForest(n_estimators=300,contamination='auto',random_state=0).fit(tr[feat])
    e.loc[g.index,'iso']=-I.score_samples(g[feat]); e.loc[g.index,'iso_pred']=(I.predict(g[feat])==-1).astype(int)
cs=e.context_switches
det={
 'RF seed42 (paper)':(e.model_proba,e.model_pred),
 'HRRN R>=2 (paper)':(r,(r>=2).astype(int)),
 'Wait share >0.5':(e.wshare,(e.wshare>.5).astype(int)),
 'CPU share <0.05':(-e.cshare,(e.cshare<.05).astype(int)),
 'No-progress switches<=2/window':(-cs,(cs<=2).astype(int)),
 'Per-process z>3 vs baseline':(e.z,(e.z>3).astype(int)),
 'IsolationForest (LOTO baseline)':(e.iso,e.iso_pred),
}
for n in ['seed7','seed123']:
    x=allscored[n][0]; det['RF/model '+n]=(x.model_proba.values,x.model_pred.values)
rows=[]
for name,(sc_,pred) in det.items():
    sc_=pd.Series(np.asarray(sc_),index=e.index);pred=pd.Series(np.asarray(pred),index=e.index)
    for (s,t),g in e.groupby(['scenario','trial']):
        y=g.is_starved;p=pred[g.index]
        rows.append(dict(det=name,scenario=s,trial=t,acc=accuracy_score(y,p),prec=precision_score(y,p,zero_division=0),rec=recall_score(y,p),f1=f1_score(y,p),auc=roc_auc_score(y,sc_[g.index]),ap=average_precision_score(y,sc_[g.index])))
R=pd.DataFrame(rows);R.to_csv(f'{OUT}/real_baselines_by_trial.csv',index=False)
S=R.groupby(['scenario','det'])[['acc','prec','rec','f1','auc','ap']].agg(['mean','std']).round(3);print(S.to_string())
# real-hardware permutation importance, seed42: shuffle one feature in scaled inputs within trial
e0,m,sc,enc,fc,meta=allscored['seed42']
X=e0.copy()
for col,le in enc.items(): X[col]=X[col].astype(str).map(lambda v:v if v in le.classes_ else le.classes_[0]);X[col]=le.transform(X[col])
X=X.reindex(columns=fc,fill_value=0);Xs=pd.DataFrame(sc.transform(X),columns=fc)
y=e0.is_starved.values;groups=e0.scenario+e0.trial.astype(str);thr=meta['recommended_threshold']
b_f1=f1_score(y,m.predict_proba(Xs.values)[:,1]>=thr);b_auc=roc_auc_score(y,m.predict_proba(Xs.values)[:,1])
rng=np.random.default_rng(0);out=[]
for c in fc:
    d=[];da=[]
    for rep in range(30):
        Z=Xs.copy();Z[c]=Xs[c].groupby(groups.values).transform(lambda s:rng.permutation(s.values))
        p=m.predict_proba(Z.values)[:,1];d.append(b_f1-f1_score(y,p>=thr));da.append(b_auc-roc_auc_score(y,p))
    out.append((c,np.mean(d),np.std(d),np.mean(da)))
P=pd.DataFrame(out,columns='feature df1 sd dauc'.split()).sort_values('df1',ascending=False);print('base f1',b_f1,'auc',b_auc);print(P.round(4).to_string());P.to_csv(f'{OUT}/real_perm_importance.csv',index=False)
