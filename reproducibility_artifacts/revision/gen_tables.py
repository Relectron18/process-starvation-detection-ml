import pandas as pd,numpy as np
import os
V='results/';O='tables/';os.makedirs(O,exist_ok=True)
def pm(s,n=3):
    return f"{s.mean():.{n}f} $\\pm$ {s.std(ddof=1):.{n}f}"
m=pd.read_csv(V+'synth/metrics_5models.csv');h=pd.read_csv(V+'synth/hrrn_fair.csv')
L=['\\begin{tabular}{lcccccc}\\toprule\nDetector & Precision & Recall & F1 & ROC AUC & PR AUC & MCC\\\\\\midrule']
for k,lab in [('Logistic Regression','Logistic Regression'),('SVM (RBF)','SVM with RBF kernel'),('Gradient Boosting','Gradient Boosting'),('XGBoost','XGBoost'),('Random Forest','Random Forest')]:
    g=m[m.model==k];L.append(lab+' & '+' & '.join(pm(g[c]) for c in ['precision','recall','f1','roc_auc','pr_auc','mcc'])+'\\\\')
L.append('\\midrule')
for k,lab in [('HRRN fixed R>=2','HRRN, fixed $R\\geq2$'),('HRRN tuned threshold (train)','HRRN, tuned threshold'),('Waiting-time threshold (train)','Waiting time threshold')]:
    g=h[h.det==k];L.append(lab+' & '+' & '.join(pm(g[c]) for c in ['precision','recall','f1','roc_auc','pr_auc','mcc'])+'\\\\')
L.append('\\bottomrule\\end{tabular}');open(O+'t_models.tex','w').write('\n'.join(L))
i=pd.read_csv(V+'synth/imbalance.csv')
L=['\\begin{tabular}{lccccc}\\toprule\nRandom Forest strategy & Precision & Recall & F1 & PR AUC & Balanced acc.\\\\\\midrule']
for k,lab in [('none','No correction'),('class_weight=balanced (paper)','Class weighting, used in this paper'),('balanced + validation threshold','Class weighting + validation threshold'),('random oversampling','Random oversampling'),('random undersampling','Random undersampling'),('SMOTE','SMOTE')]:
    g=i[i.strategy==k];L.append(lab+' & '+' & '.join(pm(g[c]) for c in ['precision','recall','f1','pr_auc','bal_acc'])+'\\\\')
L.append('\\bottomrule\\end{tabular}');open(O+'t_imb.tex','w').write('\n'.join(L))
f=pd.read_csv(V+'synth/feature_importance.csv');rp=pd.read_csv(V+'o2/real_perm_importance.csv').set_index('feature')
g=f[~f.feature.str.startswith('GROUP')].groupby('feature').agg(d=('perm_f1_drop','mean'),s=('perm_f1_drop','std'),gi=('gini','mean'),r=('rank','mean')).sort_values('d',ascending=False)
L=['\\begin{tabular}{lcccc}\\toprule\nFeature & Permutation $\\Delta$F1, synthetic & Mean rank of 19 & Impurity importance & Permutation $\\Delta$F1, real, seed 42\\\\\\midrule']
for k,x in list(g.head(9).iterrows()):
    L.append(k.replace('_','\\_')+f" & {x.d:.4f} $\\pm$ {x.s:.4f} & {x.r:.1f} & {x.gi:.3f} & {str(round(rp.loc[k,"df1"],3)).replace("-","$-$")}\\\\")
L.append('\\bottomrule\\end{tabular}');open(O+'t_fi.tex','w').write('\n'.join(L))
R=pd.read_csv(V+'o2/real_baselines_by_trial.csv')
names=[('RF seed42 (paper)','Random Forest, seed 42'),('RF/model seed7','Random Forest, seed 7'),('RF/model seed123','Gradient Boosting, seed 123'),('HRRN R>=2 (paper)','HRRN, $R\\geq2$'),('Wait share >0.5','Wait share $>0.5$'),('No-progress switches<=2/window','Few switches rule'),('Per-process z>3 vs baseline','Per process z above 3'),('IsolationForest (LOTO baseline)','Isolation Forest')]
L=['\\begin{tabular}{lcccccccc}\\toprule\n & \\multicolumn{4}{c}{RT scenario, 5 trials} & \\multicolumn{4}{c}{Nice scenario, 3 trials}\\\\\\cmidrule(lr){2-5}\\cmidrule(lr){6-9}\nDetector & Prec. & Rec. & F1 & AUC & Prec. & Rec. & F1 & AUC\\\\\\midrule']
for k,lab in names:
    c=[]
    for s in ['rt','nice']:
        g=R[(R.det==k)&(R.scenario==s)];c+=[f"{g[x].mean():.3f}" for x in ['prec','rec','f1','auc']]
    L.append(lab+' & '+' & '.join(c)+'\\\\')
L.append('\\bottomrule\\end{tabular}');open(O+'t_real.tex','w').write('\n'.join(L))
P=pd.read_csv(V+'ext/ext_oper.csv');D=pd.read_csv(V+'ext/ext_design.csv');S=pd.read_csv(V+'ext/ext_scored.csv')
sc=[('orig_rt','RT, original data'),('orig_nice','Nice, original data'),('rt','RT, replication'),('nice','Nice, replication'),('app_rt','Application workloads + RT'),('overcommit','Overcommit, 24 equal hogs'),('cgroup','Cgroup CPU throttling'),('multicore_rt','Two CPUs, two RT hogs')]
L=['\\begin{tabular}{lcccccccccc}\\toprule\n & & \\multicolumn{5}{c}{F1, operational label} & \\multicolumn{4}{c}{Episode hit rate, victim\\_normal}\\\\\\cmidrule(lr){3-7}\\cmidrule(lr){8-11}\nScenario & Windows & S42 & S7 & S123 & HRRN & Wait & S42 & S7 & S123 & HRRN\\\\\\midrule']
for s,lab in sc:
    n=len(S[(S.scenario==s)&(S.model=='seed42')])
    f=lambda d:f"{P[(P.s==s)&(P.det==d)].f1.iloc[0]:.3f}"
    e=lambda d:f"{D[(D.s==s)&(D.det==d)].hit.iloc[0]:.3f}"
    L.append(f"{lab} & {n} & {f('seed42')} & {f('seed7')} & {f('seed123')} & {f('HRRN')} & {f('wait>0.5')} & {e('seed42')} & {e('seed7')} & {e('seed123')} & {e('HRRN')}\\\\")
L.append('\\bottomrule\\end{tabular}');open(O+'t_work.tex','w').write('\n'.join(L))
o=pd.read_csv(V+'res/overhead.csv');a=o[o.part=='collector_cpu_pct']
L=['\\begin{tabular}{lcccc}\\toprule\nTracked PIDs & 5 & 20 & 100 & 500\\\\\\midrule']
r=o[o.part=='read_us_per_pid'].set_index('n')
L.append('Median read cost per PID in microseconds & '+' & '.join(f"{r.loc[n,'p50']:.1f}" for n in [5,20,100,500])+'\\\\')
for p in [0.25,1.0]:
    g=a[a.poll==p].set_index('n');L.append(f'Collector CPU percent at {p:g} s polling & '+' & '.join(f"{g.loc[n,'mean']:.2f}" for n in [5,20,100,500])+'\\\\')
L.append('\\bottomrule\\end{tabular}');open(O+'t_over.tex','w').write('\n'.join(L))
