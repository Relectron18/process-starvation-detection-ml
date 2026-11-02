"""Synthetic evidence: 12 seeds; imbalance, feature importance, HRRN fairness."""
import sys,os,warnings,time
A=sys.argv[1]; OUT=sys.argv[2]; sys.path.insert(0,A); os.environ['STARVATION_OUTPUT_DIR']=OUT
import numpy as np,pandas as pd
from sklearn.metrics import *
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.base import clone
from imblearn.over_sampling import SMOTE,RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from pipeline_core import *
SEEDS=[42,7,123,2024,99,555,777,888,314,271,161,202]
def met(y,p,s):
    return dict(accuracy=accuracy_score(y,p),precision=precision_score(y,p),recall=recall_score(y,p),f1=f1_score(y,p),roc_auc=roc_auc_score(y,s),pr_auc=average_precision_score(y,s),mcc=matthews_corrcoef(y,p),bal_acc=balanced_accuracy_score(y,p),prevalence=float(np.mean(y)))
M=[];I=[];FI=[];H=[]
for seed in SEEDS:
    t0=time.time()
    tr,te=split_raw(seed); st=fit_split_stats(tr); tr=apply_split_stats(tr,st); te=apply_split_stats(te,st)
    Xtr,ytr,enc,sc,fc=fit_encoders_scaler(tr); Xte,yte=transform_with(te,enc,sc,fc)
    for n,mdl in get_models(seed).items():
        mdl.fit(Xtr,ytr);M.append(dict(seed=seed,model=n,**met(yte,mdl.predict(Xte),mdl.predict_proba(Xte)[:,1])))
    # imbalance strategies, RF (paper config otherwise)
    rfp=dict(n_estimators=300,max_depth=12,random_state=seed,n_jobs=-1)
    def run(name,model,X,y):
        model.fit(X,y);I.append(dict(seed=seed,strategy=name,**met(yte,model.predict(Xte),model.predict_proba(Xte)[:,1])))
    run('none',RandomForestClassifier(**rfp),Xtr,ytr)
    run('class_weight=balanced (paper)',RandomForestClassifier(class_weight='balanced',**rfp),Xtr,ytr)
    for nm,S in [('random undersampling',RandomUnderSampler(random_state=seed)),('random oversampling',RandomOverSampler(random_state=seed)),('SMOTE',SMOTE(random_state=seed))]:
        Xr,yr=S.fit_resample(Xtr,ytr);run(nm,RandomForestClassifier(**rfp),Xr,yr)
    # threshold moving tuned on inner validation
    Xa,Xv,ya,yv=train_test_split(Xtr,ytr,test_size=.25,random_state=seed,stratify=ytr)
    tm=RandomForestClassifier(class_weight='balanced',**rfp).fit(Xa,ya)
    ts=np.linspace(.05,.95,37);pv=tm.predict_proba(Xv)[:,1];t=ts[int(np.argmax([f1_score(yv,pv>=x) for x in ts]))]
    bal=RandomForestClassifier(class_weight='balanced',**rfp).fit(Xtr,ytr);pt=bal.predict_proba(Xte)[:,1]
    I.append(dict(seed=seed,strategy='balanced + validation threshold',thr=t,**met(yte,(pt>=t).astype(int),pt)))
    # feature importance: permutation on test (F1 drop), plus impurity
    b=f1_score(yte,bal.predict(Xte));rng=np.random.default_rng(seed)
    gi=dict(zip(fc,bal.feature_importances_))
    drops={}
    for c in fc:
        d=[]
        for _ in range(5):
            Z=Xte.copy();Z[c]=rng.permutation(Z[c].values);d.append(b-f1_score(yte,bal.predict(Z)))
        drops[c]=np.mean(d)
    rk=pd.Series(drops).rank(ascending=False)
    for c in fc: FI.append(dict(seed=seed,feature=c,perm_f1_drop=drops[c],gini=gi[c],rank=rk[c]))
    # grouped permutation: wait family
    fam={'wait family':['waiting_time_ms','turnaround_time_ms','wait_to_burst_ratio','pk_wait_anomaly','low_priority_high_wait'],'priority family':['priority','nice_value','cpu_starvation_gap','low_priority_high_wait']}
    for k,cols in fam.items():
        d=[]
        for _ in range(5):
            Z=Xte.copy();ix=rng.permutation(len(Z));Z[cols]=Z[cols].values[ix];d.append(b-f1_score(yte,bal.predict(Z)))
        FI.append(dict(seed=seed,feature='GROUP '+k,perm_f1_drop=np.mean(d),gini=np.nan,rank=np.nan))
    # HRRN fair: fixed vs threshold tuned on train, plus AUC; rule comparators
    rtr=(tr.waiting_time_ms+tr.burst_time_ms)/(tr.burst_time_ms+1e-6);rte=(te.waiting_time_ms+te.burst_time_ms)/(te.burst_time_ms+1e-6)
    qs=np.unique(np.quantile(rtr,np.linspace(.01,.99,99)));tt=qs[int(np.argmax([f1_score(ytr,rtr>=q) for q in qs]))]
    H.append(dict(seed=seed,det='HRRN fixed R>=2',**met(yte,(rte>=2).astype(int),rte)))
    H.append(dict(seed=seed,det='HRRN tuned threshold (train)',thr=tt,**met(yte,(rte>=tt).astype(int),rte)))
    w=tr.waiting_time_ms;qs=np.unique(np.quantile(w,np.linspace(.01,.99,99)));tw=qs[int(np.argmax([f1_score(ytr,w>=q) for q in qs]))]
    H.append(dict(seed=seed,det='Waiting-time threshold (train)',thr=tw,**met(yte,(te.waiting_time_ms>=tw).astype(int),te.waiting_time_ms)))
    print(seed,round(time.time()-t0,1),flush=True)
    for name,L in [('metrics_5models',M),('imbalance',I),('feature_importance',FI),('hrrn_fair',H)]: pd.DataFrame(L).to_csv(f'{OUT}/{name}.csv',index=False)
