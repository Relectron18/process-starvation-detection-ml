import os;os.makedirs('figures',exist_ok=True)
import pandas as pd,numpy as np,matplotlib
matplotlib.use('Agg');import matplotlib.pyplot as plt
plt.rcParams.update({'font.family':'serif','font.serif':['Times New Roman','Times','DejaVu Serif'],'font.size':9,'axes.spines.top':False,'axes.spines.right':False,'axes.grid':True,'axes.axisbelow':True,'grid.color':'#e3e3e3','grid.linewidth':.6,'axes.edgecolor':'#777','xtick.color':'#333','ytick.color':'#333'})
A,B='#2563a8','#d9722c';INK='#333333'
d=pd.read_csv('../multiseed_results.csv');assert d.seed.nunique()==12
o=['Logistic Regression','SVM (RBF)','Gradient Boosting','XGBoost','Random Forest'];lab=['Logistic\nRegression','SVM','Gradient\nBoosting','XGBoost','Random\nForest']
fig,ax=plt.subplots(figsize=(4.6,2.6))
data=[d[d.model==m].f1.values for m in o]
ax.boxplot(data,widths=.5,patch_artist=True,showfliers=False,boxprops=dict(facecolor='#c9dcf0',edgecolor=A,linewidth=1),medianprops=dict(color=A,linewidth=2),whiskerprops=dict(color=A),capprops=dict(color=A))
r=np.random.default_rng(0)
for i,v in enumerate(data): ax.scatter(i+1+r.uniform(-.1,.1,len(v)),v,s=11,color=INK,alpha=.75,zorder=3,linewidths=0)
ax.set_xticks(range(1,6),lab);ax.set_ylabel('Test F1');ax.grid(axis='x',visible=False)
fig.tight_layout();fig.savefig('figures/fig_f1.png',dpi=300);plt.close()
R=pd.read_csv('../real_world_multitrial_results.csv')
M=['accuracy','precision','recall','f1'];N=['Accuracy','Precision','Recall','F1']
fig,ax=plt.subplots(figsize=(3.3,2.45));x=np.arange(4);w=.34
for j,(s,l,c,h) in enumerate([('rt','RT real time hog',A,None),('nice','Niceness contention',B,'///')]):
    g=R[R.scenario==s]
    ax.bar(x+(j-.5)*(w+.02),g[M].mean(),w,yerr=g[M].std(ddof=1),color=c,hatch=h,edgecolor='white',linewidth=.5,capsize=2,error_kw=dict(ecolor=INK,linewidth=.8),label=l)
ax.set_xticks(x,N);ax.set_ylim(0,1.05);ax.set_ylabel('Score');ax.grid(axis='x',visible=False)
ax.legend(loc='upper center',bbox_to_anchor=(.5,1.17),ncol=2,frameon=False,fontsize=8,handlelength=1.5,columnspacing=1)
fig.tight_layout();fig.savefig('figures/fig_metrics.png',dpi=300);plt.close()
S=pd.read_csv('results/ext/ext_scored.csv');g=S[(S.scenario=='orig_rt')&(S.trial==1)&(S.model=='seed42')&(S.process_label=='victim_normal')].sort_values('t_window_end')
t=g.t_window_end;fig,(a,b)=plt.subplots(2,1,figsize=(3.3,2.45),sharex=True,gridspec_kw=dict(hspace=.14))
for z in (a,b): z.axvspan(12,28,color='#efefef',zorder=0)
a.plot(t,100*g.burst_time_ms/g.window_ms,color=A,lw=1.6);a.plot(t,100*g.waiting_time_ms/g.window_ms,color=B,lw=1.6,ls='--')
a.set_ylabel('Percent');a.set_ylim(-3,108)
a.text(3,76,'CPU time',color=INK,fontsize=8);a.text(16,44,'Run queue wait',color=INK,fontsize=8)
b.plot(t,g.model_proba,color=A,lw=1.6,label='Predicted probability');b.scatter(t,g.is_starved,s=9,color=INK,zorder=3,label='Label');b.axhline(.53,color='#777',lw=.9,ls=':')
b.text(45.5,.56,'Threshold 0.53',ha='right',fontsize=8,color=INK);b.text(31,.39,'Predicted probability',fontsize=8,color=INK);b.text(1,.73,'Dots show label',fontsize=8,color=INK)
b.set_ylim(-.08,1.08);b.set_ylabel('Probability');b.set_xlabel('Trial time in seconds');b.set_xlim(0,46)
fig.subplots_adjust(left=.16,right=.98,top=.97,bottom=.16);fig.savefig('figures/fig_trial.png',dpi=300);plt.close()
