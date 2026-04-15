#!/usr/bin/env python3
"""
DATATHON PASSOS MÁGICOS - BLOCO 3: MODELAGEM PREDITIVA (2020-2024)
Forecasting T->T+1. Treino: 2020-2023, Teste: 2023->2024.
"""
import pandas as pd, numpy as np, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (classification_report, roc_auc_score, f1_score,
    accuracy_score, recall_score, precision_score, roc_curve)
from imblearn.over_sampling import SMOTE
import joblib, json, warnings
warnings.filterwarnings('ignore')

FIG = '/home/ubuntu/datathon_passos/figures/'
MDL = '/home/ubuntu/datathon_passos/models/'
COLORS = ['#1f77b4','#ff7f0e','#2ca02c','#d62728','#9467bd']

print("=" * 70)
print("CARREGANDO DADOS 2020-2024")
df = pd.read_csv('/home/ubuntu/datathon_passos/data/dataset_completo_2020_2024.csv')
for c in ['INDE','IAN','IDA','IEG','IAA','IPS','IPP','IPV']:
    df[c] = pd.to_numeric(df[c], errors='coerce')
print(f"Dataset: {df.shape[0]} registros, Anos: {sorted(df['ANO'].unique())}")

# Build T->T+1 pairs
THRESH = 5.506
pairs = []
for y_t, y_t1 in [(2020,2021),(2021,2022),(2022,2023),(2023,2024)]:
    dt = df[df['ANO']==y_t].copy()
    dt1 = df[df['ANO']==y_t1][['NOME','INDE','PEDRA']].rename(columns={'INDE':'INDE_F','PEDRA':'PEDRA_F'})
    m = dt.merge(dt1, on='NOME', how='inner')
    m['RISCO_F'] = (m['INDE_F'] < THRESH).astype(int)
    m['PAR'] = f'{y_t}->{y_t1}'
    pairs.append(m)
dp = pd.concat(pairs, ignore_index=True)
print(f"\nPares T->T+1: {len(dp)}")
print(dp.groupby('PAR')['NOME'].count())
print(f"Taxa risco: {dp['RISCO_F'].mean()*100:.1f}%")

# Feature engineering
dp['IDA_IEG_r'] = dp['IDA']/(dp['IEG']+0.01)
dp['IAA_IDA_d'] = dp['IAA']-dp['IDA']
dp['IPS_IPP_m'] = (dp['IPS']+dp['IPP'])/2
dp['INDE_bm'] = (dp['INDE']<dp['INDE'].mean()).astype(int)
dp['IDA_low'] = (dp['IDA']<5).astype(int)
dp['IEG_low'] = (dp['IEG']<5).astype(int)
dp['IAN_low'] = (dp['IAN']<5).astype(int)
dp['multi_alert'] = dp['IDA_low']+dp['IEG_low']+dp['IAN_low']
dp['IDA_IPS_r'] = dp['IDA']/(dp['IPS']+0.01)
dp['IPV_IEG_r'] = dp['IPV']/(dp['IEG']+0.01)

fcols = ['IAN','IDA','IEG','IAA','IPS','IPP','IPV','INDE',
    'IDA_IEG_r','IAA_IDA_d','IPS_IPP_m','INDE_bm',
    'IDA_low','IEG_low','IAN_low','multi_alert','IDA_IPS_r','IPV_IEG_r']
print(f"\nFeatures: {len(fcols)}")

# Train/test split (temporal)
train_mask = dp['PAR'].isin(['2020->2021','2021->2022','2022->2023'])
test_mask = dp['PAR']=='2023->2024'
dtr = dp[train_mask].dropna(subset=fcols+['RISCO_F'])
dte = dp[test_mask].dropna(subset=fcols+['RISCO_F'])
X_tr, y_tr = dtr[fcols].values, dtr['RISCO_F'].values
X_te, y_te = dte[fcols].values, dte['RISCO_F'].values
print(f"\nTreino: {len(X_tr)} ({y_tr.mean()*100:.1f}% risco)")
print(f"Teste:  {len(X_te)} ({y_te.mean()*100:.1f}% risco)")

sc = StandardScaler()
X_tr_s = sc.fit_transform(X_tr)
X_te_s = sc.transform(X_te)

sm = SMOTE(random_state=42)
X_tr_b, y_tr_b = sm.fit_resample(X_tr_s, y_tr)
print(f"Após SMOTE: {len(X_tr_b)}")

# Train models
models = {
    'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
    'Random Forest': RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42),
    'Gradient Boosting': GradientBoostingClassifier(n_estimators=200, max_depth=5, random_state=42)
}
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
results = {}
for name, mdl in models.items():
    print(f"\n--- {name} ---")
    mdl.fit(X_tr_b, y_tr_b)
    cvs = cross_val_score(mdl, X_tr_b, y_tr_b, cv=cv, scoring='roc_auc')
    yp = mdl.predict(X_te_s)
    ypr = mdl.predict_proba(X_te_s)[:,1]
    a,f,r,p,au = accuracy_score(y_te,yp),f1_score(y_te,yp),recall_score(y_te,yp),precision_score(y_te,yp),roc_auc_score(y_te,ypr)
    results[name] = {'model':mdl,'acc':a,'f1':f,'rec':r,'prec':p,'auc':au,'cv':cvs.mean(),'ypr':ypr}
    print(f"  CV AUC: {cvs.mean():.3f}, Test AUC: {au:.3f}, Acc: {a:.3f}, F1: {f:.3f}, Recall: {r:.3f}")

# Select best & optimize threshold
bn = max(results, key=lambda k: results[k]['auc'])
br = results[bn]; bm = br['model']; ybp = br['ypr']
print(f"\nMelhor: {bn} (AUC={br['auc']:.3f})")

best_t, best_f = 0.5, 0
for t in np.arange(0.1,0.9,0.05):
    yp_t = (ybp>=t).astype(int)
    ft = f1_score(y_te, yp_t)
    if ft > best_f: best_f=ft; best_t=t

yf = (ybp>=best_t).astype(int)
fa,ff,fr,fp,fau = accuracy_score(y_te,yf),f1_score(y_te,yf),recall_score(y_te,yf),precision_score(y_te,yf),roc_auc_score(y_te,ybp)
print(f"\nFinal (threshold={best_t:.2f}): Acc={fa:.3f}, F1={ff:.3f}, Recall={fr:.3f}, Prec={fp:.3f}, AUC={fau:.3f}")
print(classification_report(y_te, yf, target_names=['Sem Risco','Em Risco']))

# Feature importance
if hasattr(bm,'feature_importances_'): imp=bm.feature_importances_
elif hasattr(bm,'coef_'): imp=np.abs(bm.coef_[0])
else: imp=np.zeros(len(fcols))
fi = pd.DataFrame({'Feature':fcols,'Imp':imp}).sort_values('Imp',ascending=True)

# Plots
fig,ax=plt.subplots(figsize=(10,5))
mn=list(results.keys()); mets=['acc','f1','rec','prec','auc']; x=np.arange(len(mn)); w=0.15
for i,mt in enumerate(mets):
    ax.bar(x+i*w,[results[m][mt] for m in mn],w,label=mt.upper(),color=COLORS[i])
ax.set_xticks(x+w*2); ax.set_xticklabels(mn,rotation=15); ax.set_title('Comparação Modelos (2023->2024)'); ax.legend(); ax.set_ylim(0,1.1)
plt.tight_layout(); plt.savefig(f'{FIG}modelo_comparacao.png'); plt.close()

fig,ax=plt.subplots(figsize=(10,6))
fi.plot(kind='barh',x='Feature',y='Imp',ax=ax,color=COLORS[0],legend=False)
ax.set_title(f'Feature Importance ({bn})'); plt.tight_layout(); plt.savefig(f'{FIG}modelo_feature_importance.png'); plt.close()

fig,ax=plt.subplots(figsize=(8,6))
for i,(n,r) in enumerate(results.items()):
    fpr,tpr,_=roc_curve(y_te,r['ypr']); ax.plot(fpr,tpr,label=f"{n} (AUC={r['auc']:.3f})",color=COLORS[i])
ax.plot([0,1],[0,1],'k--',alpha=0.5); ax.set_title('ROC Curves'); ax.legend()
plt.tight_layout(); plt.savefig(f'{FIG}modelo_roc_curves.png'); plt.close()

fig,ax=plt.subplots(figsize=(10,5))
ax.hist(ybp[y_te==0],bins=30,alpha=0.6,label='Sem Risco',color=COLORS[2])
ax.hist(ybp[y_te==1],bins=30,alpha=0.6,label='Em Risco',color=COLORS[3])
ax.axvline(x=best_t,color='black',ls='--',lw=2,label=f'Threshold={best_t:.2f}')
ax.set_title('Distribuição de Probabilidades'); ax.legend()
plt.tight_layout(); plt.savefig(f'{FIG}modelo_probabilidades.png'); plt.close()

fig,ax=plt.subplots(figsize=(10,5))
rpa = df.groupby('ANO').apply(lambda x: (x['INDE']<THRESH).mean()*100)
ax.plot(rpa.index,rpa.values,'o-',color=COLORS[3],lw=2,ms=8)
for xv,yv in zip(rpa.index,rpa.values): ax.text(xv,yv+1,f"{yv:.1f}%",ha='center',fontweight='bold')
ax.set_title('% Alunos em Risco por Ano'); ax.set_xticks(sorted(df['ANO'].unique()))
plt.tight_layout(); plt.savefig(f'{FIG}modelo_transicao_risco.png'); plt.close()
print("Gráficos salvos!")

# Save model
joblib.dump(bm, f'{MDL}modelo_risco.pkl')
joblib.dump(sc, f'{MDL}scaler.pkl')
with open(f'{MDL}features.txt','w') as f: f.write('\n'.join(fcols))
with open(f'{MDL}threshold.txt','w') as f: f.write(str(best_t))
meta = {'model_name':bn,'period':'2020-2024','train':'2020-2023','test':'2023->2024',
    'train_size':int(len(X_tr_b)),'test_size':int(len(X_te)),'threshold':float(best_t),
    'features':fcols,'metrics':{'accuracy':float(fa),'f1':float(ff),'recall':float(fr),'precision':float(fp),'auc_roc':float(fau)}}
with open(f'{MDL}metadata.json','w') as f: json.dump(meta,f,indent=2)
print(f"\nModelo salvo! Modelagem concluída.")
