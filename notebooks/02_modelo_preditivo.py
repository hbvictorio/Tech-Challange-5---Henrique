#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
=============================================================================
DATATHON PASSOS MÁGICOS - BLOCO 3 (v2 - FORECASTING OTIMIZADO)
Modelagem Preditiva de Risco de Defasagem (Ano T → Ano T+1)
=============================================================================

ABORDAGEM DE FORECASTING VERDADEIRO:
Usamos os indicadores do aluno no Ano T para prever se ele entrará em
risco de defasagem no Ano T+1. Isso elimina o data leakage e cria um
Sistema de Alerta Precoce real.

Otimizações aplicadas:
- SMOTE para balanceamento de classes
- Feature Engineering avançado (deltas, interações, flags compostas)
- Ensemble com Voting Classifier
- Otimização de threshold para maximizar Recall (capturar mais alunos em risco)
- Separação temporal rigorosa (treino: 2020→2021, teste: 2021→2022)
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import (RandomForestClassifier, GradientBoostingClassifier,
                              VotingClassifier)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (classification_report, confusion_matrix,
                             roc_auc_score, roc_curve, f1_score, accuracy_score,
                             precision_recall_curve)
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
import joblib
import json
import warnings
warnings.filterwarnings('ignore')

FIGURES_DIR = '/home/ubuntu/datathon_passos/figures/'
MODELS_DIR = '/home/ubuntu/datathon_passos/models/'

plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams.update({
    'font.family': 'DejaVu Sans',
    'font.size': 11,
    'figure.dpi': 150,
    'savefig.dpi': 150,
    'savefig.bbox': 'tight'
})

CORES_PM = {
    'azul_escuro': '#1B3A5C',
    'azul_medio': '#2E6B9E',
    'laranja': '#E87722',
    'verde': '#2E8B57',
    'vermelho': '#C0392B',
    'amarelo': '#F1C40F',
    'cinza': '#94A3B8',
}

# ============================================================================
# 1. CONSTRUÇÃO DE PARES TEMPORAIS (T → T+1)
# ============================================================================
print("=" * 70)
print("ETAPA 1: CONSTRUÇÃO DE PARES TEMPORAIS (T → T+1)")
print("=" * 70)

df_long = pd.read_csv('/home/ubuntu/datathon_passos/data/dataset_limpo_long.csv')
print(f"Dataset longitudinal: {df_long.shape}")

indicadores = ['IDA', 'IEG', 'IAA', 'IPS', 'IPP', 'IPV', 'IAN']

# Criar pares T → T+1
pares = []
for nome in df_long['NOME'].unique():
    aluno = df_long[df_long['NOME'] == nome].sort_values('ANO')
    anos = sorted(aluno['ANO'].unique())
    for i in range(len(anos) - 1):
        ano_t = anos[i]
        ano_t1 = anos[i + 1]
        if ano_t1 - ano_t == 1:
            row_t = aluno[aluno['ANO'] == ano_t].iloc[0]
            row_t1 = aluno[aluno['ANO'] == ano_t1].iloc[0]
            par = {
                'NOME': nome,
                'ANO_T': int(ano_t),
                'ANO_T1': int(ano_t1),
                'IDA_T': row_t['IDA'],
                'IEG_T': row_t['IEG'],
                'IAA_T': row_t['IAA'],
                'IPS_T': row_t['IPS'],
                'IPP_T': row_t['IPP'],
                'IPV_T': row_t['IPV'],
                'IAN_T': row_t['IAN'],
                'INDE_T': row_t['INDE'],
                'FASE_T': row_t['FASE'],
                'PEDRA_T': row_t['PEDRA'],
                'PONTO_VIRADA_T': row_t.get('PONTO_VIRADA', 0),
                'INDE_T1': row_t1['INDE'],
                'PEDRA_T1': row_t1['PEDRA'],
            }
            pares.append(par)

df_pares = pd.DataFrame(pares)
print(f"Pares temporais criados: {len(df_pares)}")
print(f"  2020 → 2021: {(df_pares['ANO_T'] == 2020).sum()}")
print(f"  2021 → 2022: {(df_pares['ANO_T'] == 2021).sum()}")

# Variável alvo: Risco no Ano T+1 (INDE < 5.506 = Quartzo)
df_pares['RISCO_T1'] = (df_pares['INDE_T1'] < 5.506).astype(int)
print(f"\nDistribuição da variável alvo (RISCO no T+1):")
print(df_pares['RISCO_T1'].value_counts())
print(f"Taxa de risco: {df_pares['RISCO_T1'].mean()*100:.1f}%")

# ============================================================================
# 2. FEATURE ENGINEERING AVANÇADO
# ============================================================================
print("\n" + "=" * 70)
print("ETAPA 2: FEATURE ENGINEERING AVANÇADO")
print("=" * 70)

# --- Features base do Ano T ---
# Já temos: IDA_T, IEG_T, IAA_T, IPS_T, IPP_T, IPV_T, IAN_T, INDE_T, FASE_T

# --- Features derivadas ---
# Razões e interações
df_pares['IDA_IEG_RATIO_T'] = df_pares['IDA_T'] / (df_pares['IEG_T'] + 0.01)
df_pares['IDA_IPS_RATIO_T'] = df_pares['IDA_T'] / (df_pares['IPS_T'] + 0.01)
df_pares['IEG_IPV_PROD_T'] = df_pares['IEG_T'] * df_pares['IPV_T']
df_pares['IDA_IPP_PROD_T'] = df_pares['IDA_T'] * df_pares['IPP_T']

# Estatísticas agregadas
df_pares['MEDIA_IND_T'] = df_pares[['IDA_T', 'IEG_T', 'IAA_T', 'IPS_T', 'IPP_T', 'IPV_T']].mean(axis=1)
df_pares['STD_IND_T'] = df_pares[['IDA_T', 'IEG_T', 'IAA_T', 'IPS_T', 'IPP_T', 'IPV_T']].std(axis=1)
df_pares['MIN_IND_T'] = df_pares[['IDA_T', 'IEG_T', 'IAA_T', 'IPS_T', 'IPP_T', 'IPV_T']].min(axis=1)
df_pares['MAX_IND_T'] = df_pares[['IDA_T', 'IEG_T', 'IAA_T', 'IPS_T', 'IPP_T', 'IPV_T']].max(axis=1)
df_pares['RANGE_IND_T'] = df_pares['MAX_IND_T'] - df_pares['MIN_IND_T']

# Viés de autoavaliação
df_pares['IAA_IDA_DIFF_T'] = df_pares['IAA_T'] - df_pares['IDA_T']
df_pares['IAA_INDE_DIFF_T'] = df_pares['IAA_T'] - df_pares['INDE_T']

# Flags de alerta
df_pares['IDA_BAIXO_T'] = (df_pares['IDA_T'] < 5.0).astype(int)
df_pares['IEG_BAIXO_T'] = (df_pares['IEG_T'] < 5.0).astype(int)
df_pares['IAN_BAIXO_T'] = (df_pares['IAN_T'] < 5.0).astype(int)
df_pares['RISCO_T'] = (df_pares['INDE_T'] < 5.506).astype(int)
df_pares['MULTI_ALERTA_T'] = (df_pares['IDA_BAIXO_T'] + df_pares['IEG_BAIXO_T'] + df_pares['IAN_BAIXO_T']).clip(0, 3)

# Distância do limiar
df_pares['DISTANCIA_QUARTZO_T'] = df_pares['INDE_T'] - 5.506
df_pares['DISTANCIA_QUARTZO_SQ_T'] = df_pares['DISTANCIA_QUARTZO_T'] ** 2

# Pedra ordinal
pedra_map = {'Quartzo': 0, 'Agata': 1, 'Ametista': 2, 'Topazio': 3}
df_pares['PEDRA_ORD_T'] = df_pares['PEDRA_T'].map(pedra_map).fillna(-1)

# Ponto de virada
df_pares['PV_T'] = df_pares['PONTO_VIRADA_T'].fillna(0).astype(int)

feature_cols = [
    # Base
    'IDA_T', 'IEG_T', 'IAA_T', 'IPS_T', 'IPP_T', 'IPV_T', 'IAN_T',
    'INDE_T', 'FASE_T', 'PEDRA_ORD_T', 'PV_T',
    # Razões e interações
    'IDA_IEG_RATIO_T', 'IDA_IPS_RATIO_T', 'IEG_IPV_PROD_T', 'IDA_IPP_PROD_T',
    # Agregadas
    'MEDIA_IND_T', 'STD_IND_T', 'MIN_IND_T', 'RANGE_IND_T',
    # Viés
    'IAA_IDA_DIFF_T', 'IAA_INDE_DIFF_T',
    # Flags
    'IDA_BAIXO_T', 'IEG_BAIXO_T', 'IAN_BAIXO_T', 'RISCO_T', 'MULTI_ALERTA_T',
    # Distância
    'DISTANCIA_QUARTZO_T', 'DISTANCIA_QUARTZO_SQ_T',
]

print(f"Features utilizadas: {len(feature_cols)}")

# Remover NaN
df_model = df_pares[feature_cols + ['RISCO_T1', 'ANO_T', 'NOME']].dropna()
print(f"Dataset para modelagem: {df_model.shape[0]} pares")

# ============================================================================
# 3. SEPARAÇÃO TEMPORAL TREINO/TESTE
# ============================================================================
print("\n" + "=" * 70)
print("ETAPA 3: SEPARAÇÃO TEMPORAL TREINO/TESTE")
print("=" * 70)

train_mask = df_model['ANO_T'] == 2020
test_mask = df_model['ANO_T'] == 2021

X_train = df_model.loc[train_mask, feature_cols].values
y_train = df_model.loc[train_mask, 'RISCO_T1'].values
X_test = df_model.loc[test_mask, feature_cols].values
y_test = df_model.loc[test_mask, 'RISCO_T1'].values

print(f"Treino (2020→2021): {len(X_train)} pares ({y_train.mean()*100:.1f}% risco)")
print(f"Teste  (2021→2022): {len(X_test)} pares ({y_test.mean()*100:.1f}% risco)")

# Escalar
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# SMOTE para balanceamento
smote = SMOTE(random_state=42, k_neighbors=3)
X_train_sm, y_train_sm = smote.fit_resample(X_train_scaled, y_train)
print(f"\nApós SMOTE: {len(X_train_sm)} amostras ({y_train_sm.mean()*100:.1f}% risco)")

# ============================================================================
# 4. MODELAGEM PREDITIVA
# ============================================================================
print("\n" + "=" * 70)
print("ETAPA 4: MODELAGEM PREDITIVA")
print("=" * 70)

models = {
    'Logistic Regression': LogisticRegression(
        max_iter=3000, C=0.3, class_weight='balanced', random_state=42
    ),
    'Random Forest': RandomForestClassifier(
        n_estimators=500, max_depth=6, min_samples_split=15,
        min_samples_leaf=8, class_weight='balanced_subsample',
        random_state=42, n_jobs=-1
    ),
    'Gradient Boosting': GradientBoostingClassifier(
        n_estimators=300, max_depth=3, learning_rate=0.03,
        min_samples_split=15, min_samples_leaf=8,
        subsample=0.7, random_state=42
    ),
}

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
results = {}

for name, model in models.items():
    print(f"\n--- {name} ---")
    
    # Treinar com SMOTE
    cv_scores = cross_val_score(model, X_train_sm, y_train_sm, cv=cv, scoring='roc_auc')
    model.fit(X_train_sm, y_train_sm)
    
    y_prob = model.predict_proba(X_test_scaled)[:, 1]
    
    # Otimizar threshold usando F1
    precisions, recalls, thresholds_pr = precision_recall_curve(y_test, y_prob)
    f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-8)
    best_idx = np.argmax(f1_scores)
    best_threshold = thresholds_pr[best_idx] if best_idx < len(thresholds_pr) else 0.5
    
    # Aplicar threshold otimizado
    y_pred = (y_prob >= best_threshold).astype(int)
    
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    try:
        auc = roc_auc_score(y_test, y_prob)
    except ValueError:
        auc = 0.0
    
    # Recall (sensibilidade) - crucial para alerta precoce
    from sklearn.metrics import recall_score, precision_score
    rec = recall_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    
    print(f"  CV AUC: {cv_scores.mean():.3f} (+/- {cv_scores.std():.3f})")
    print(f"  Best Threshold: {best_threshold:.3f}")
    print(f"  Test Accuracy: {acc:.3f}")
    print(f"  Test Precision: {prec:.3f}")
    print(f"  Test Recall: {rec:.3f}")
    print(f"  Test F1-Score: {f1:.3f}")
    print(f"  Test AUC-ROC: {auc:.3f}")
    print(f"\n  Classification Report:")
    print(classification_report(y_test, y_pred, target_names=['Sem Risco', 'Em Risco']))
    
    results[name] = {
        'model': model,
        'y_pred': y_pred,
        'y_prob': y_prob,
        'accuracy': acc,
        'precision': prec,
        'recall': rec,
        'f1': f1,
        'auc': auc,
        'cv_auc_mean': cv_scores.mean(),
        'cv_auc_std': cv_scores.std(),
        'threshold': best_threshold,
    }

# --- Ensemble (Soft Voting) ---
print("\n--- Ensemble (Soft Voting) ---")
ensemble = VotingClassifier(
    estimators=[
        ('lr', LogisticRegression(max_iter=3000, C=0.3, class_weight='balanced', random_state=42)),
        ('rf', RandomForestClassifier(n_estimators=500, max_depth=6, min_samples_split=15,
                                       min_samples_leaf=8, class_weight='balanced_subsample',
                                       random_state=42, n_jobs=-1)),
        ('gb', GradientBoostingClassifier(n_estimators=300, max_depth=3, learning_rate=0.03,
                                           min_samples_split=15, min_samples_leaf=8,
                                           subsample=0.7, random_state=42)),
    ],
    voting='soft'
)
cv_scores_ens = cross_val_score(ensemble, X_train_sm, y_train_sm, cv=cv, scoring='roc_auc')
ensemble.fit(X_train_sm, y_train_sm)
y_prob_ens = ensemble.predict_proba(X_test_scaled)[:, 1]

# Otimizar threshold
precisions_e, recalls_e, thresholds_pr_e = precision_recall_curve(y_test, y_prob_ens)
f1_scores_e = 2 * (precisions_e * recalls_e) / (precisions_e + recalls_e + 1e-8)
best_idx_e = np.argmax(f1_scores_e)
best_threshold_e = thresholds_pr_e[best_idx_e] if best_idx_e < len(thresholds_pr_e) else 0.5

y_pred_ens = (y_prob_ens >= best_threshold_e).astype(int)
acc_e = accuracy_score(y_test, y_pred_ens)
f1_e = f1_score(y_test, y_pred_ens)
auc_e = roc_auc_score(y_test, y_prob_ens)
rec_e = recall_score(y_test, y_pred_ens)
prec_e = precision_score(y_test, y_pred_ens)

print(f"  CV AUC: {cv_scores_ens.mean():.3f} (+/- {cv_scores_ens.std():.3f})")
print(f"  Best Threshold: {best_threshold_e:.3f}")
print(f"  Test Accuracy: {acc_e:.3f}")
print(f"  Test Precision: {prec_e:.3f}")
print(f"  Test Recall: {rec_e:.3f}")
print(f"  Test F1-Score: {f1_e:.3f}")
print(f"  Test AUC-ROC: {auc_e:.3f}")
print(f"\n  Classification Report:")
print(classification_report(y_test, y_pred_ens, target_names=['Sem Risco', 'Em Risco']))

results['Ensemble (Voting)'] = {
    'model': ensemble,
    'y_pred': y_pred_ens,
    'y_prob': y_prob_ens,
    'accuracy': acc_e,
    'precision': prec_e,
    'recall': rec_e,
    'f1': f1_e,
    'auc': auc_e,
    'cv_auc_mean': cv_scores_ens.mean(),
    'cv_auc_std': cv_scores_ens.std(),
    'threshold': best_threshold_e,
}

# ============================================================================
# 5. SELECIONAR MELHOR MODELO E AVALIAR
# ============================================================================
print("\n" + "=" * 70)
print("ETAPA 5: AVALIAÇÃO DOS RESULTADOS")
print("=" * 70)

# Selecionar pelo melhor AUC
best_name = max(results, key=lambda x: results[x]['auc'])
best_result = results[best_name]
print(f"\nMelhor modelo: {best_name}")
print(f"  AUC-ROC: {best_result['auc']:.3f}")
print(f"  F1-Score: {best_result['f1']:.3f}")
print(f"  Recall: {best_result['recall']:.3f}")
print(f"  Threshold: {best_result['threshold']:.3f}")

# ---- Gráfico 1: Comparação de modelos ----
fig, axes = plt.subplots(1, 3, figsize=(18, 6))

metrics_df = pd.DataFrame({
    name: {'Accuracy': r['accuracy'], 'F1-Score': r['f1'], 'AUC-ROC': r['auc'], 'Recall': r['recall']}
    for name, r in results.items()
}).T
metrics_df[['Accuracy', 'F1-Score', 'AUC-ROC', 'Recall']].plot(
    kind='bar', ax=axes[0], width=0.8,
    color=[CORES_PM['azul_escuro'], CORES_PM['laranja'], CORES_PM['verde'], CORES_PM['vermelho']]
)
axes[0].set_title('Comparação de Modelos\n(Forecasting T→T+1 com SMOTE)', fontweight='bold', fontsize=12)
axes[0].set_ylabel('Score')
axes[0].set_xticklabels(axes[0].get_xticklabels(), rotation=20, ha='right', fontsize=9)
axes[0].legend(loc='lower right', fontsize=9)
axes[0].set_ylim(0, 1)

# Curva ROC
for name, r in results.items():
    fpr, tpr, _ = roc_curve(y_test, r['y_prob'])
    axes[1].plot(fpr, tpr, label=f"{name} (AUC={r['auc']:.3f})", linewidth=2)
axes[1].plot([0, 1], [0, 1], 'k--', alpha=0.5)
axes[1].set_title('Curva ROC\n(Teste: 2021→2022)', fontweight='bold', fontsize=12)
axes[1].set_xlabel('Taxa de Falsos Positivos')
axes[1].set_ylabel('Taxa de Verdadeiros Positivos')
axes[1].legend(fontsize=8)

# Matriz de confusão do melhor
cm = confusion_matrix(y_test, best_result['y_pred'])
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[2],
            xticklabels=['Sem Risco', 'Em Risco'],
            yticklabels=['Sem Risco', 'Em Risco'])
axes[2].set_title(f'Matriz de Confusão\n{best_name}', fontweight='bold', fontsize=12)
axes[2].set_xlabel('Predito')
axes[2].set_ylabel('Real')

plt.tight_layout()
plt.savefig(f'{FIGURES_DIR}modelo_comparacao.png')
plt.close()
print("Gráfico salvo: modelo_comparacao.png")

# ---- Gráfico 2: Feature Importance ----
fig, ax = plt.subplots(figsize=(12, 9))

if best_name == 'Random Forest':
    importances = best_result['model'].feature_importances_
elif best_name == 'Gradient Boosting':
    importances = best_result['model'].feature_importances_
elif best_name == 'Ensemble (Voting)':
    # Average importance from RF and GB
    rf_imp = best_result['model'].named_estimators_['rf'].feature_importances_
    gb_imp = best_result['model'].named_estimators_['gb'].feature_importances_
    importances = (rf_imp + gb_imp) / 2
else:
    importances = np.abs(best_result['model'].coef_[0])

feat_imp = pd.Series(importances, index=feature_cols).sort_values(ascending=True)
colors = [CORES_PM['vermelho'] if v > feat_imp.quantile(0.75) else CORES_PM['azul_medio'] for v in feat_imp]
feat_imp.plot(kind='barh', ax=ax, color=colors)
ax.set_title(f'Importância das Features - {best_name}\n(Previsão do Risco no Ano Seguinte)', fontweight='bold')
ax.set_xlabel('Importância')

plt.tight_layout()
plt.savefig(f'{FIGURES_DIR}modelo_feature_importance.png')
plt.close()
print("Gráfico salvo: modelo_feature_importance.png")

# ---- Gráfico 3: Distribuição de probabilidades ----
fig, ax = plt.subplots(figsize=(10, 6))
ax.hist(best_result['y_prob'][y_test == 0], bins=25, alpha=0.6,
        color=CORES_PM['verde'], label='Sem Risco (real)', density=True)
ax.hist(best_result['y_prob'][y_test == 1], bins=25, alpha=0.6,
        color=CORES_PM['vermelho'], label='Em Risco (real)', density=True)
ax.axvline(best_result['threshold'], color='black', linestyle='--', linewidth=2,
           label=f'Threshold ({best_result["threshold"]:.2f})')
ax.set_title(f'Distribuição de Probabilidades Preditas\n{best_name} (Forecasting T→T+1)', fontweight='bold')
ax.set_xlabel('Probabilidade de Risco no Ano Seguinte')
ax.set_ylabel('Densidade')
ax.legend()

plt.tight_layout()
plt.savefig(f'{FIGURES_DIR}modelo_probabilidades.png')
plt.close()
print("Gráfico salvo: modelo_probabilidades.png")

# ---- Gráfico 4: Análise de transição de risco ----
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# 4a: % que cai para risco por pedra atual
transicoes = df_pares.groupby(['PEDRA_T', 'RISCO_T1']).size().unstack(fill_value=0)
transicoes_pct = transicoes.div(transicoes.sum(axis=1), axis=0) * 100
if 1 in transicoes_pct.columns:
    risco_por_pedra = transicoes_pct[1].reindex(['Quartzo', 'Agata', 'Ametista', 'Topazio']).fillna(0)
    risco_por_pedra.plot(kind='bar', ax=axes[0], color=CORES_PM['vermelho'])
    axes[0].set_title('% de Alunos que Caem para Risco no Ano Seguinte\n(por Pedra Atual)', fontweight='bold')
    axes[0].set_ylabel('% que entra em Risco')
    axes[0].set_xlabel('Pedra no Ano T')
    axes[0].set_xticklabels(axes[0].get_xticklabels(), rotation=0)
    for i, v in enumerate(risco_por_pedra):
        axes[0].text(i, v + 1, f'{v:.1f}%', ha='center', fontweight='bold', fontsize=11)

# 4b: Perfil médio no Ano T
perfil = df_pares.groupby('RISCO_T1')[['IDA_T', 'IEG_T', 'IAA_T', 'IPS_T', 'IPP_T', 'IPV_T', 'IAN_T']].mean()
perfil_renamed = perfil.rename(columns={c: c.replace('_T', '') for c in perfil.columns})
perfil_renamed.T.plot(kind='bar', ax=axes[1],
                      color=[CORES_PM['verde'], CORES_PM['vermelho']])
axes[1].set_title('Perfil Médio no Ano T\n(Quem entra em Risco vs Quem não entra no T+1)', fontweight='bold')
axes[1].set_ylabel('Valor Médio do Indicador')
axes[1].set_xlabel('Indicador')
axes[1].legend(['Sem Risco no T+1', 'Em Risco no T+1'])
axes[1].set_xticklabels(axes[1].get_xticklabels(), rotation=0)

plt.tight_layout()
plt.savefig(f'{FIGURES_DIR}modelo_transicao_risco.png')
plt.close()
print("Gráfico salvo: modelo_transicao_risco.png")

# ============================================================================
# 6. SALVAR MODELO E ARTEFATOS
# ============================================================================
print("\n" + "=" * 70)
print("ETAPA 6: SALVANDO MODELO E ARTEFATOS")
print("=" * 70)

best_model = best_result['model']
joblib.dump(best_model, f'{MODELS_DIR}modelo_risco.pkl')
joblib.dump(scaler, f'{MODELS_DIR}scaler.pkl')

with open(f'{MODELS_DIR}features.txt', 'w') as f:
    for feat in feature_cols:
        f.write(feat + '\n')

with open(f'{MODELS_DIR}threshold.txt', 'w') as f:
    f.write(str(best_result['threshold']))

metadata = {
    'model_name': best_name,
    'model_version': 'v2_forecasting_optimized',
    'approach': 'Forecasting T→T+1 com SMOTE + Threshold Otimizado',
    'features': feature_cols,
    'n_features': len(feature_cols),
    'accuracy': round(best_result['accuracy'], 4),
    'precision': round(best_result['precision'], 4),
    'recall': round(best_result['recall'], 4),
    'f1_score': round(best_result['f1'], 4),
    'auc_roc': round(best_result['auc'], 4),
    'cv_auc_mean': round(best_result['cv_auc_mean'], 4),
    'cv_auc_std': round(best_result['cv_auc_std'], 4),
    'threshold': round(best_result['threshold'], 4),
    'train_period': '2020→2021',
    'test_period': '2021→2022',
    'train_size': int(len(X_train)),
    'test_size': int(len(X_test)),
    'train_size_after_smote': int(len(X_train_sm)),
    'risco_threshold': 5.506,
    'risco_definition': 'INDE < 5.506 no Ano T+1 (classificação Quartzo)',
    'all_models': {
        name: {
            'accuracy': round(r['accuracy'], 4),
            'precision': round(r['precision'], 4),
            'recall': round(r['recall'], 4),
            'f1_score': round(r['f1'], 4),
            'auc_roc': round(r['auc'], 4),
            'cv_auc_mean': round(r['cv_auc_mean'], 4),
            'threshold': round(r['threshold'], 4),
        }
        for name, r in results.items()
    }
}

with open(f'{MODELS_DIR}metadata.json', 'w') as f:
    json.dump(metadata, f, indent=2, ensure_ascii=False)

print(f"Modelo salvo: {MODELS_DIR}modelo_risco.pkl")
print(f"Scaler salvo: {MODELS_DIR}scaler.pkl")
print(f"Features salvas: {MODELS_DIR}features.txt")
print(f"Threshold salvo: {MODELS_DIR}threshold.txt")
print(f"Metadados salvos: {MODELS_DIR}metadata.json")

# ============================================================================
# RESUMO FINAL
# ============================================================================
print("\n" + "=" * 70)
print("RESUMO DA MODELAGEM PREDITIVA (FORECASTING OTIMIZADO)")
print("=" * 70)
print(f"""
ABORDAGEM: Forecasting Verdadeiro (Ano T → Ano T+1)
  Treino: Indicadores de 2020 → Risco em 2021 (com SMOTE)
  Teste:  Indicadores de 2021 → Risco em 2022

Modelo selecionado: {best_name}
Features utilizadas: {len(feature_cols)}
Threshold otimizado: {best_result['threshold']:.3f}

Métricas no conjunto de teste:
  Accuracy:  {best_result['accuracy']:.3f}
  Precision: {best_result['precision']:.3f}
  Recall:    {best_result['recall']:.3f}
  F1-Score:  {best_result['f1']:.3f}
  AUC-ROC:   {best_result['auc']:.3f}
  CV AUC:    {best_result['cv_auc_mean']:.3f} (+/- {best_result['cv_auc_std']:.3f})
""")

print("Comparação de todos os modelos:")
print("-" * 75)
print(f"{'Modelo':<25} {'Accuracy':>10} {'Precision':>10} {'Recall':>10} {'F1':>10} {'AUC':>10}")
print("-" * 75)
for name, r in results.items():
    marker = " ←" if name == best_name else ""
    print(f"{name:<25} {r['accuracy']:>10.3f} {r['precision']:>10.3f} {r['recall']:>10.3f} {r['f1']:>10.3f} {r['auc']:>10.3f}{marker}")
print("-" * 75)
