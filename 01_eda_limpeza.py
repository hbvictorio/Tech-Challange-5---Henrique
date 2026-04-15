#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
=============================================================================
DATATHON PASSOS MÁGICOS - BLOCO 2
Análise Exploratória de Dados (EDA) e Limpeza
=============================================================================

Este script realiza:
1. Carregamento e limpeza dos dados PEDE (2020, 2021, 2022)
2. Transformação para formato longitudinal (long)
3. Respostas analíticas às 10 perguntas obrigatórias do desafio
4. Geração de gráficos profissionais para cada pergunta

Autor: Equipe Datathon
Data: 2024
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# CONFIGURAÇÃO DE ESTILO
# ============================================================================
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams.update({
    'font.family': 'DejaVu Sans',
    'font.size': 11,
    'axes.titlesize': 14,
    'axes.labelsize': 12,
    'figure.figsize': (12, 6),
    'figure.dpi': 150,
    'savefig.dpi': 150,
    'savefig.bbox': 'tight'
})

# Paleta Passos Mágicos
CORES_PM = {
    'azul_escuro': '#1B3A5C',
    'azul_medio': '#2E6B9E',
    'laranja': '#E87722',
    'cinza': '#6B7B8D',
    'verde': '#2E8B57',
    'vermelho': '#C0392B',
    'amarelo': '#F39C12',
    'roxo': '#8E44AD'
}
CORES_PEDRAS = {
    'Quartzo': '#D4A574',
    'Ágata': '#7B8D9E',
    'Ametista': '#9B59B6',
    'Topázio': '#F1C40F'
}

FIGURES_DIR = '/home/ubuntu/datathon_passos/figures/'

# ============================================================================
# 1. CARREGAMENTO E LIMPEZA DOS DADOS
# ============================================================================
print("=" * 70)
print("ETAPA 1: CARREGAMENTO E LIMPEZA DOS DADOS")
print("=" * 70)

# Carregar dataset original
df_raw = pd.read_csv(
    '/home/ubuntu/upload/PEDE_PASSOS_DATASET_FIAP.xlsx',
    sep=';',
    encoding='utf-8'
)
print(f"\nDataset original: {df_raw.shape[0]} alunos x {df_raw.shape[1]} colunas")

# ---- Limpeza de tipos numéricos ----
# Colunas que devem ser numéricas
indicadores = ['INDE', 'IAN', 'IDA', 'IEG', 'IAA', 'IPS', 'IPP', 'IPV']
for ano in ['2020', '2021', '2022']:
    for ind in indicadores:
        col = f'{ind}_{ano}'
        if col in df_raw.columns:
            df_raw[col] = pd.to_numeric(df_raw[col], errors='coerce')

# Colunas numéricas adicionais
for col in ['IDADE_ALUNO_2020', 'ANOS_PM_2020', 'DEFASAGEM_2021',
            'ANO_INGRESSO_2022', 'CG_2022', 'CF_2022', 'CT_2022',
            'NOTA_PORT_2022', 'NOTA_MAT_2022', 'NOTA_ING_2022',
            'QTD_AVAL_2022']:
    if col in df_raw.columns:
        df_raw[col] = pd.to_numeric(df_raw[col], errors='coerce')

# ---- Limpeza de PONTO_VIRADA ----
for ano in ['2020', '2021', '2022']:
    col = f'PONTO_VIRADA_{ano}'
    if col in df_raw.columns:
        df_raw[col] = df_raw[col].map(
            lambda x: 1 if str(x).strip().lower() in ['sim', '1', 'true', 'yes']
            else (0 if str(x).strip().lower() in ['não', 'nao', '0', 'false', 'no']
                  else np.nan)
        )

# ---- Limpeza de FASE ----
def extract_fase(val):
    """Extrai o número da fase de strings como '2H', '3F', etc."""
    if pd.isna(val):
        return np.nan
    val_str = str(val).strip()
    if val_str == '':
        return np.nan
    # Tenta extrair o primeiro dígito
    for ch in val_str:
        if ch.isdigit():
            return int(ch)
    return np.nan

df_raw['FASE_NUM_2020'] = df_raw['FASE_TURMA_2020'].apply(extract_fase)
if 'FASE_2021' in df_raw.columns:
    df_raw['FASE_NUM_2021'] = pd.to_numeric(df_raw['FASE_2021'], errors='coerce')
if 'FASE_2022' in df_raw.columns:
    df_raw['FASE_NUM_2022'] = pd.to_numeric(df_raw['FASE_2022'], errors='coerce')

# ---- Criar dataset limpo ----
df = df_raw.copy()

# ============================================================================
# 2. TRANSFORMAÇÃO PARA FORMATO LONGITUDINAL
# ============================================================================
print("\n" + "=" * 70)
print("ETAPA 2: TRANSFORMAÇÃO PARA FORMATO LONGITUDINAL")
print("=" * 70)

records = []
for ano in ['2020', '2021', '2022']:
    cols_ano = {
        'NOME': 'NOME',
        f'INDE_{ano}': 'INDE',
        f'IAN_{ano}': 'IAN',
        f'IDA_{ano}': 'IDA',
        f'IEG_{ano}': 'IEG',
        f'IAA_{ano}': 'IAA',
        f'IPS_{ano}': 'IPS',
        f'IPP_{ano}': 'IPP',
        f'IPV_{ano}': 'IPV',
        f'PEDRA_{ano}': 'PEDRA',
        f'PONTO_VIRADA_{ano}': 'PONTO_VIRADA',
        f'FASE_NUM_{ano}': 'FASE',
    }
    # Filtrar colunas existentes
    cols_existentes = {k: v for k, v in cols_ano.items() if k in df.columns}
    df_ano = df[list(cols_existentes.keys())].rename(columns=cols_existentes)
    df_ano['ANO'] = int(ano)

    # Adicionar defasagem se existir
    if f'DEFASAGEM_{ano}' in df.columns:
        df_ano['DEFASAGEM'] = df[f'DEFASAGEM_{ano}']
    if f'NIVEL_IDEAL_{ano}' in df.columns:
        df_ano['NIVEL_IDEAL'] = df[f'NIVEL_IDEAL_{ano}']

    records.append(df_ano)

df_long = pd.concat(records, ignore_index=True)
# Remover linhas onde todos os indicadores são nulos (aluno não participou naquele ano)
ind_cols = ['INDE', 'IAN', 'IDA', 'IEG', 'IAA', 'IPS', 'IPP', 'IPV']
df_long = df_long.dropna(subset=ind_cols, how='all')

print(f"Dataset longitudinal: {df_long.shape[0]} registros (aluno-ano)")
print(f"Alunos únicos: {df_long['NOME'].nunique()}")
print(f"\nDistribuição por ano:")
print(df_long['ANO'].value_counts().sort_index())

# Salvar datasets limpos
df.to_csv('/home/ubuntu/datathon_passos/data/dataset_limpo_wide.csv', index=False)
df_long.to_csv('/home/ubuntu/datathon_passos/data/dataset_limpo_long.csv', index=False)
print("\nDatasets salvos em data/")

# ============================================================================
# PERGUNTA 1: Perfil de defasagem (IAN) e sua evolução
# ============================================================================
print("\n" + "=" * 70)
print("PERGUNTA 1: Perfil de defasagem (IAN) e sua evolução")
print("=" * 70)

# Estatísticas do IAN por ano
ian_stats = df_long.groupby('ANO')['IAN'].describe()
print("\nEstatísticas do IAN por ano:")
print(ian_stats.to_string())

# Classificação de defasagem baseada no IAN
def classificar_defasagem_ian(ian):
    if pd.isna(ian):
        return np.nan
    if ian >= 8:
        return 'Adequado'
    elif ian >= 5:
        return 'Moderada'
    else:
        return 'Severa'

df_long['DEFASAGEM_CLASS'] = df_long['IAN'].apply(classificar_defasagem_ian)

defasagem_dist = df_long.groupby(['ANO', 'DEFASAGEM_CLASS']).size().unstack(fill_value=0)
defasagem_pct = defasagem_dist.div(defasagem_dist.sum(axis=1), axis=0) * 100
print("\nDistribuição de defasagem (%) por ano:")
print(defasagem_pct.round(1).to_string())

# Gráfico 1a: Evolução do IAN médio
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

ian_mean = df_long.groupby('ANO')['IAN'].mean()
axes[0].bar(ian_mean.index.astype(str), ian_mean.values, color=CORES_PM['azul_medio'], width=0.5)
axes[0].set_title('Evolução do IAN Médio por Ano', fontweight='bold')
axes[0].set_xlabel('Ano')
axes[0].set_ylabel('IAN Médio')
for i, (ano, val) in enumerate(ian_mean.items()):
    axes[0].text(i, val + 0.05, f'{val:.2f}', ha='center', fontweight='bold')
axes[0].set_ylim(0, 10)

# Gráfico 1b: Distribuição de defasagem por ano
order = ['Severa', 'Moderada', 'Leve', 'Adequado']
colors = [CORES_PM['vermelho'], CORES_PM['laranja'], CORES_PM['amarelo'], CORES_PM['verde']]
cols_exist = [c for c in order if c in defasagem_pct.columns]
defasagem_pct[cols_exist].plot(kind='bar', stacked=True, ax=axes[1],
                                color=[colors[order.index(c)] for c in cols_exist])
axes[1].set_title('Distribuição de Defasagem por Ano', fontweight='bold')
axes[1].set_xlabel('Ano')
axes[1].set_ylabel('Percentual (%)')
axes[1].legend(title='Classificação', bbox_to_anchor=(1.05, 1))
axes[1].set_xticklabels(axes[1].get_xticklabels(), rotation=0)

plt.tight_layout()
plt.savefig(f'{FIGURES_DIR}p1_ian_defasagem.png')
plt.close()
print("Gráfico salvo: p1_ian_defasagem.png")

# ============================================================================
# PERGUNTA 2: Desempenho acadêmico (IDA) - melhorando, estagnado ou caindo?
# ============================================================================
print("\n" + "=" * 70)
print("PERGUNTA 2: Desempenho acadêmico (IDA)")
print("=" * 70)

ida_stats = df_long.groupby('ANO')['IDA'].describe()
print("\nEstatísticas do IDA por ano:")
print(ida_stats.to_string())

# Análise de alunos que aparecem em múltiplos anos
alunos_multi = df_long.groupby('NOME').filter(lambda x: len(x) >= 2)
pivot_ida = alunos_multi.pivot_table(index='NOME', columns='ANO', values='IDA')

# Calcular tendência individual
def tendencia(row):
    vals = row.dropna().values
    if len(vals) < 2:
        return 'Insuficiente'
    diff = vals[-1] - vals[0]
    if diff > 0.5:
        return 'Melhorando'
    elif diff < -0.5:
        return 'Caindo'
    else:
        return 'Estagnado'

pivot_ida['TENDENCIA'] = pivot_ida.apply(tendencia, axis=1)
tend_dist = pivot_ida['TENDENCIA'].value_counts()
print("\nTendência individual do IDA (alunos com 2+ anos):")
print(tend_dist)

fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Box plot do IDA por ano
df_long.boxplot(column='IDA', by='ANO', ax=axes[0],
                boxprops=dict(color=CORES_PM['azul_escuro']),
                medianprops=dict(color=CORES_PM['laranja'], linewidth=2))
axes[0].set_title('Distribuição do IDA por Ano', fontweight='bold')
fig.suptitle('')
axes[0].set_xlabel('Ano')
axes[0].set_ylabel('IDA')

# Tendência individual
tend_colors = {'Melhorando': CORES_PM['verde'], 'Estagnado': CORES_PM['amarelo'],
               'Caindo': CORES_PM['vermelho'], 'Insuficiente': CORES_PM['cinza']}
tend_filtered = tend_dist.drop('Insuficiente', errors='ignore')
axes[1].bar(tend_filtered.index, tend_filtered.values,
            color=[tend_colors.get(x, CORES_PM['cinza']) for x in tend_filtered.index])
axes[1].set_title('Tendência Individual do IDA', fontweight='bold')
axes[1].set_ylabel('Número de Alunos')
for i, (cat, val) in enumerate(tend_filtered.items()):
    axes[1].text(i, val + 2, str(val), ha='center', fontweight='bold')

plt.tight_layout()
plt.savefig(f'{FIGURES_DIR}p2_ida_desempenho.png')
plt.close()
print("Gráfico salvo: p2_ida_desempenho.png")

# ============================================================================
# PERGUNTA 3: Relação IEG com IDA e IPV
# ============================================================================
print("\n" + "=" * 70)
print("PERGUNTA 3: Relação do engajamento (IEG) com IDA e IPV")
print("=" * 70)

# Correlações
for ano in [2020, 2021, 2022]:
    subset = df_long[df_long['ANO'] == ano].dropna(subset=['IEG', 'IDA', 'IPV'])
    if len(subset) > 10:
        corr_ida = subset['IEG'].corr(subset['IDA'])
        corr_ipv = subset['IEG'].corr(subset['IPV'])
        print(f"\nAno {ano}: Corr(IEG, IDA) = {corr_ida:.3f} | Corr(IEG, IPV) = {corr_ipv:.3f} | n={len(subset)}")

fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Scatter IEG vs IDA
for ano, cor in zip([2020, 2021, 2022], [CORES_PM['azul_escuro'], CORES_PM['azul_medio'], CORES_PM['laranja']]):
    subset = df_long[(df_long['ANO'] == ano)].dropna(subset=['IEG', 'IDA'])
    axes[0].scatter(subset['IEG'], subset['IDA'], alpha=0.3, s=20, color=cor, label=str(ano))
axes[0].set_title('IEG vs IDA (todos os anos)', fontweight='bold')
axes[0].set_xlabel('IEG (Engajamento)')
axes[0].set_ylabel('IDA (Aprendizagem)')
axes[0].legend()

# Scatter IEG vs IPV
for ano, cor in zip([2020, 2021, 2022], [CORES_PM['azul_escuro'], CORES_PM['azul_medio'], CORES_PM['laranja']]):
    subset = df_long[(df_long['ANO'] == ano)].dropna(subset=['IEG', 'IPV'])
    axes[1].scatter(subset['IEG'], subset['IPV'], alpha=0.3, s=20, color=cor, label=str(ano))
axes[1].set_title('IEG vs IPV (todos os anos)', fontweight='bold')
axes[1].set_xlabel('IEG (Engajamento)')
axes[1].set_ylabel('IPV (Ponto de Virada)')
axes[1].legend()

plt.tight_layout()
plt.savefig(f'{FIGURES_DIR}p3_ieg_ida_ipv.png')
plt.close()
print("Gráfico salvo: p3_ieg_ida_ipv.png")

# ============================================================================
# PERGUNTA 4: Coerência IAA com IDA e IEG reais
# ============================================================================
print("\n" + "=" * 70)
print("PERGUNTA 4: Coerência da autoavaliação (IAA) com IDA e IEG")
print("=" * 70)

for ano in [2020, 2021, 2022]:
    subset = df_long[df_long['ANO'] == ano].dropna(subset=['IAA', 'IDA', 'IEG'])
    if len(subset) > 10:
        corr_ida = subset['IAA'].corr(subset['IDA'])
        corr_ieg = subset['IAA'].corr(subset['IEG'])
        # Viés: diferença média IAA - IDA
        vies = (subset['IAA'] - subset['IDA']).mean()
        print(f"Ano {ano}: Corr(IAA,IDA)={corr_ida:.3f} | Corr(IAA,IEG)={corr_ieg:.3f} | Viés(IAA-IDA)={vies:+.2f} | n={len(subset)}")

fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# IAA vs IDA
subset_all = df_long.dropna(subset=['IAA', 'IDA'])
axes[0].scatter(subset_all['IAA'], subset_all['IDA'], alpha=0.2, s=15, color=CORES_PM['azul_medio'])
# Linha de igualdade
lim = [0, 10]
axes[0].plot(lim, lim, 'r--', alpha=0.5, label='Linha de igualdade')
axes[0].set_title('IAA vs IDA (Autoavaliação vs Aprendizagem)', fontweight='bold')
axes[0].set_xlabel('IAA (Autoavaliação)')
axes[0].set_ylabel('IDA (Aprendizagem)')
axes[0].legend()

# Distribuição do viés (IAA - IDA)
vies_all = subset_all['IAA'] - subset_all['IDA']
axes[1].hist(vies_all, bins=30, color=CORES_PM['azul_medio'], edgecolor='white', alpha=0.8)
axes[1].axvline(0, color='red', linestyle='--', linewidth=2, label='Sem viés')
axes[1].axvline(vies_all.mean(), color=CORES_PM['laranja'], linestyle='-', linewidth=2,
                label=f'Média: {vies_all.mean():+.2f}')
axes[1].set_title('Distribuição do Viés (IAA - IDA)', fontweight='bold')
axes[1].set_xlabel('Viés (IAA - IDA)')
axes[1].set_ylabel('Frequência')
axes[1].legend()

plt.tight_layout()
plt.savefig(f'{FIGURES_DIR}p4_iaa_coerencia.png')
plt.close()
print("Gráfico salvo: p4_iaa_coerencia.png")

# ============================================================================
# PERGUNTA 5: Padrões psicossociais (IPS) que antecedem quedas
# ============================================================================
print("\n" + "=" * 70)
print("PERGUNTA 5: Padrões psicossociais (IPS) que antecedem quedas")
print("=" * 70)

# Analisar alunos com dados em anos consecutivos
alunos_2020_2021 = df_long[df_long['ANO'].isin([2020, 2021])].groupby('NOME').filter(lambda x: len(x) == 2)
pivot_ips = alunos_2020_2021.pivot_table(index='NOME', columns='ANO', values=['IPS', 'IDA', 'INDE'])

if not pivot_ips.empty:
    # Calcular variação do IDA e INDE
    pivot_ips[('DELTA_IDA', '')] = pivot_ips[('IDA', 2021)] - pivot_ips[('IDA', 2020)]
    pivot_ips[('DELTA_INDE', '')] = pivot_ips[('INDE', 2021)] - pivot_ips[('INDE', 2020)]
    pivot_ips[('IPS_INICIAL', '')] = pivot_ips[('IPS', 2020)]

    # Classificar: queda = delta_IDA < -1
    pivot_ips[('QUEDA', '')] = pivot_ips[('DELTA_IDA', '')] < -1

    queda_sim = pivot_ips[pivot_ips[('QUEDA', '')] == True][('IPS_INICIAL', '')].describe()
    queda_nao = pivot_ips[pivot_ips[('QUEDA', '')] == False][('IPS_INICIAL', '')].describe()
    print("\nIPS inicial de alunos que TIVERAM queda no IDA (>1 ponto):")
    print(queda_sim)
    print("\nIPS inicial de alunos que NÃO tiveram queda:")
    print(queda_nao)

fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# IPS vs variação do IDA
if not pivot_ips.empty:
    axes[0].scatter(pivot_ips[('IPS_INICIAL', '')], pivot_ips[('DELTA_IDA', '')],
                    alpha=0.3, s=20, color=CORES_PM['azul_medio'])
    axes[0].axhline(0, color='red', linestyle='--', alpha=0.5)
    axes[0].set_title('IPS (2020) vs Variação do IDA (2020→2021)', fontweight='bold')
    axes[0].set_xlabel('IPS em 2020')
    axes[0].set_ylabel('Variação do IDA')

    # Box plot: IPS por grupo de queda
    data_queda = [
        pivot_ips[pivot_ips[('QUEDA', '')] == True][('IPS_INICIAL', '')].dropna(),
        pivot_ips[pivot_ips[('QUEDA', '')] == False][('IPS_INICIAL', '')].dropna()
    ]
    bp = axes[1].boxplot(data_queda, labels=['Com Queda', 'Sem Queda'],
                         patch_artist=True)
    bp['boxes'][0].set_facecolor(CORES_PM['vermelho'])
    bp['boxes'][1].set_facecolor(CORES_PM['verde'])
    axes[1].set_title('IPS Inicial: Alunos com vs sem Queda no IDA', fontweight='bold')
    axes[1].set_ylabel('IPS')

plt.tight_layout()
plt.savefig(f'{FIGURES_DIR}p5_ips_quedas.png')
plt.close()
print("Gráfico salvo: p5_ips_quedas.png")

# ============================================================================
# PERGUNTA 6: Avaliações psicopedagógicas (IPP) vs defasagem (IAN)
# ============================================================================
print("\n" + "=" * 70)
print("PERGUNTA 6: IPP confirma ou contradiz a defasagem do IAN?")
print("=" * 70)

for ano in [2020, 2021, 2022]:
    subset = df_long[df_long['ANO'] == ano].dropna(subset=['IPP', 'IAN'])
    if len(subset) > 10:
        corr = subset['IPP'].corr(subset['IAN'])
        print(f"Ano {ano}: Corr(IPP, IAN) = {corr:.3f} | n={len(subset)}")

fig, axes = plt.subplots(1, 2, figsize=(16, 6))

subset_all = df_long.dropna(subset=['IPP', 'IAN'])
axes[0].scatter(subset_all['IAN'], subset_all['IPP'], alpha=0.2, s=15, color=CORES_PM['azul_medio'])
axes[0].set_title('IAN vs IPP (todos os anos)', fontweight='bold')
axes[0].set_xlabel('IAN (Adequação ao Nível)')
axes[0].set_ylabel('IPP (Psicopedagógico)')

# IPP médio por faixa de defasagem
subset_all['IAN_FAIXA'] = pd.cut(subset_all['IAN'], bins=[0, 4, 6, 8, 10],
                                  labels=['Severa', 'Moderada', 'Leve', 'Adequado'])
ipp_por_faixa = subset_all.groupby('IAN_FAIXA', observed=True)['IPP'].mean()
colors_faixa = [CORES_PM['vermelho'], CORES_PM['laranja'], CORES_PM['amarelo'], CORES_PM['verde']]
axes[1].bar(ipp_por_faixa.index.astype(str), ipp_por_faixa.values,
            color=colors_faixa[:len(ipp_por_faixa)])
axes[1].set_title('IPP Médio por Faixa de Defasagem (IAN)', fontweight='bold')
axes[1].set_xlabel('Faixa de Defasagem')
axes[1].set_ylabel('IPP Médio')
for i, (cat, val) in enumerate(ipp_por_faixa.items()):
    axes[1].text(i, val + 0.05, f'{val:.2f}', ha='center', fontweight='bold')

plt.tight_layout()
plt.savefig(f'{FIGURES_DIR}p6_ipp_ian.png')
plt.close()
print("Gráfico salvo: p6_ipp_ian.png")

# ============================================================================
# PERGUNTA 7: Comportamentos que influenciam o IPV no tempo
# ============================================================================
print("\n" + "=" * 70)
print("PERGUNTA 7: Comportamentos que mais influenciam o IPV")
print("=" * 70)

# Correlação de cada indicador com IPV
features_ipv = ['IDA', 'IEG', 'IAA', 'IPS', 'IPP', 'IAN']
print("\nCorrelação com IPV por ano:")
corr_results = []
for ano in [2020, 2021, 2022]:
    subset = df_long[df_long['ANO'] == ano].dropna(subset=['IPV'] + features_ipv)
    if len(subset) > 10:
        corrs = {f: subset[f].corr(subset['IPV']) for f in features_ipv}
        corrs['ANO'] = ano
        corr_results.append(corrs)
        print(f"  {ano}: " + " | ".join([f"{f}={corrs[f]:.3f}" for f in features_ipv]))

fig, ax = plt.subplots(figsize=(12, 6))
df_corr = pd.DataFrame(corr_results).set_index('ANO')
df_corr.plot(kind='bar', ax=ax, width=0.8,
             color=[CORES_PM['azul_escuro'], CORES_PM['azul_medio'], CORES_PM['laranja'],
                    CORES_PM['verde'], CORES_PM['roxo'], CORES_PM['cinza']])
ax.set_title('Correlação dos Indicadores com IPV por Ano', fontweight='bold')
ax.set_xlabel('Ano')
ax.set_ylabel('Correlação de Pearson')
ax.legend(title='Indicador', bbox_to_anchor=(1.05, 1))
ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
ax.axhline(0, color='black', linewidth=0.5)

plt.tight_layout()
plt.savefig(f'{FIGURES_DIR}p7_ipv_influencias.png')
plt.close()
print("Gráfico salvo: p7_ipv_influencias.png")

# ============================================================================
# PERGUNTA 8: Combinações de indicadores que elevam o INDE
# ============================================================================
print("\n" + "=" * 70)
print("PERGUNTA 8: Combinações de indicadores que elevam o INDE")
print("=" * 70)

# Correlação de cada indicador com INDE
features_inde = ['IDA', 'IEG', 'IAA', 'IPS', 'IPP', 'IPV', 'IAN']
subset_inde = df_long.dropna(subset=['INDE'] + features_inde)
print(f"\nCorrelação com INDE (n={len(subset_inde)}):")
corr_inde = {f: subset_inde[f].corr(subset_inde['INDE']) for f in features_inde}
for f, c in sorted(corr_inde.items(), key=lambda x: -abs(x[1])):
    print(f"  {f}: {c:.3f}")

# Heatmap de correlação
fig, axes = plt.subplots(1, 2, figsize=(16, 7))

corr_matrix = df_long[['INDE'] + features_inde].corr()
mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)
sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='RdYlBu_r',
            center=0, ax=axes[0], mask=mask, square=True,
            linewidths=0.5, cbar_kws={'shrink': 0.8})
axes[0].set_title('Matriz de Correlação dos Indicadores', fontweight='bold')

# Top combinações: scatter com cor por INDE
subset_top = df_long.dropna(subset=['IDA', 'IEG', 'INDE'])
scatter = axes[1].scatter(subset_top['IDA'], subset_top['IEG'],
                          c=subset_top['INDE'], cmap='RdYlGn', alpha=0.4, s=15)
plt.colorbar(scatter, ax=axes[1], label='INDE')
axes[1].set_title('IDA vs IEG (cor = INDE)', fontweight='bold')
axes[1].set_xlabel('IDA (Aprendizagem)')
axes[1].set_ylabel('IEG (Engajamento)')

plt.tight_layout()
plt.savefig(f'{FIGURES_DIR}p8_inde_combinacoes.png')
plt.close()
print("Gráfico salvo: p8_inde_combinacoes.png")

# ============================================================================
# PERGUNTA 9: Padrões para identificar alunos em risco
# ============================================================================
print("\n" + "=" * 70)
print("PERGUNTA 9: Padrões para identificar alunos em risco")
print("=" * 70)

# Definir "risco" como INDE < 5.506 (Quartzo) ou queda significativa
df_long['EM_RISCO'] = df_long['INDE'].apply(lambda x: 1 if pd.notna(x) and x < 5.506 else 0)

risco_dist = df_long.groupby('ANO')['EM_RISCO'].value_counts(normalize=True).unstack() * 100
print("\nPercentual de alunos em risco (Quartzo) por ano:")
print(risco_dist.round(1))

# Perfil médio dos alunos em risco vs não risco
perfil = df_long.groupby('EM_RISCO')[features_inde].mean()
print("\nPerfil médio - Em risco (1) vs Não risco (0):")
print(perfil.round(2).to_string())

fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Radar-like comparison
categorias = features_inde
x = np.arange(len(categorias))
width = 0.35
risco_vals = perfil.loc[1, categorias].values
nao_risco_vals = perfil.loc[0, categorias].values

axes[0].bar(x - width/2, nao_risco_vals, width, label='Sem Risco', color=CORES_PM['verde'], alpha=0.8)
axes[0].bar(x + width/2, risco_vals, width, label='Em Risco', color=CORES_PM['vermelho'], alpha=0.8)
axes[0].set_xticks(x)
axes[0].set_xticklabels(categorias, rotation=45)
axes[0].set_title('Perfil Médio: Em Risco vs Sem Risco', fontweight='bold')
axes[0].set_ylabel('Valor Médio')
axes[0].legend()

# Evolução do % em risco
risco_pct = df_long.groupby('ANO')['EM_RISCO'].mean() * 100
axes[1].plot(risco_pct.index, risco_pct.values, 'o-', color=CORES_PM['vermelho'],
             linewidth=2, markersize=8)
axes[1].set_title('Evolução do Percentual de Alunos em Risco', fontweight='bold')
axes[1].set_xlabel('Ano')
axes[1].set_ylabel('% em Risco (Quartzo)')
for ano, val in risco_pct.items():
    axes[1].text(ano, val + 0.5, f'{val:.1f}%', ha='center', fontweight='bold')

plt.tight_layout()
plt.savefig(f'{FIGURES_DIR}p9_alunos_risco.png')
plt.close()
print("Gráfico salvo: p9_alunos_risco.png")

# ============================================================================
# PERGUNTA 10: Impacto consistente nas fases (Pedras)
# ============================================================================
print("\n" + "=" * 70)
print("PERGUNTA 10: Impacto nas fases Quartzo, Ágata, Ametista e Topázio")
print("=" * 70)

pedra_order = ['Quartzo', 'Ágata', 'Ametista', 'Topázio']
pedra_stats = df_long.groupby(['ANO', 'PEDRA'])[ind_cols].mean()
print("\nINDE médio por Pedra e Ano:")
inde_pedra = df_long.groupby(['ANO', 'PEDRA'])['INDE'].agg(['mean', 'count'])
print(inde_pedra.to_string())

# Distribuição de pedras por ano
pedra_dist = df_long.groupby(['ANO', 'PEDRA']).size().unstack(fill_value=0)
pedra_pct = pedra_dist.div(pedra_dist.sum(axis=1), axis=0) * 100

fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Distribuição de pedras por ano
cols_pedra = [p for p in pedra_order if p in pedra_pct.columns]
pedra_pct[cols_pedra].plot(kind='bar', stacked=True, ax=axes[0],
                            color=[CORES_PEDRAS.get(p, '#999') for p in cols_pedra])
axes[0].set_title('Distribuição de Pedras por Ano', fontweight='bold')
axes[0].set_xlabel('Ano')
axes[0].set_ylabel('Percentual (%)')
axes[0].legend(title='Pedra', bbox_to_anchor=(1.05, 1))
axes[0].set_xticklabels(axes[0].get_xticklabels(), rotation=0)

# INDE médio por pedra ao longo dos anos
for pedra in cols_pedra:
    subset = df_long[df_long['PEDRA'] == pedra].groupby('ANO')['INDE'].mean()
    axes[1].plot(subset.index, subset.values, 'o-', label=pedra,
                 color=CORES_PEDRAS.get(pedra, '#999'), linewidth=2, markersize=8)
axes[1].set_title('INDE Médio por Pedra ao Longo dos Anos', fontweight='bold')
axes[1].set_xlabel('Ano')
axes[1].set_ylabel('INDE Médio')
axes[1].legend(title='Pedra')

plt.tight_layout()
plt.savefig(f'{FIGURES_DIR}p10_impacto_fases.png')
plt.close()
print("Gráfico salvo: p10_impacto_fases.png")

# ============================================================================
# INSIGHT ADICIONAL (Pergunta 11)
# ============================================================================
print("\n" + "=" * 70)
print("INSIGHT ADICIONAL: Análise de Retenção e Progressão")
print("=" * 70)

# Alunos que progrediram de pedra
alunos_2020 = df_long[df_long['ANO'] == 2020][['NOME', 'PEDRA']].rename(columns={'PEDRA': 'PEDRA_2020'})
alunos_2021 = df_long[df_long['ANO'] == 2021][['NOME', 'PEDRA']].rename(columns={'PEDRA': 'PEDRA_2021'})
alunos_2022 = df_long[df_long['ANO'] == 2022][['NOME', 'PEDRA']].rename(columns={'PEDRA': 'PEDRA_2022'})

progressao = alunos_2020.merge(alunos_2021, on='NOME', how='inner')
pedra_num = {'Quartzo': 1, 'Ágata': 2, 'Ametista': 3, 'Topázio': 4}
progressao['NUM_2020'] = progressao['PEDRA_2020'].map(pedra_num)
progressao['NUM_2021'] = progressao['PEDRA_2021'].map(pedra_num)
progressao['DELTA'] = progressao['NUM_2021'] - progressao['NUM_2020']

print(f"\nAlunos com dados 2020→2021: {len(progressao)}")
print("Progressão de pedra:")
print(f"  Subiram: {(progressao['DELTA'] > 0).sum()} ({100*(progressao['DELTA'] > 0).mean():.1f}%)")
print(f"  Mantiveram: {(progressao['DELTA'] == 0).sum()} ({100*(progressao['DELTA'] == 0).mean():.1f}%)")
print(f"  Desceram: {(progressao['DELTA'] < 0).sum()} ({100*(progressao['DELTA'] < 0).mean():.1f}%)")

# Gráfico de progressão
fig, ax = plt.subplots(figsize=(10, 6))
prog_dist = progressao['DELTA'].value_counts().sort_index()
colors_prog = []
for d in prog_dist.index:
    if d > 0:
        colors_prog.append(CORES_PM['verde'])
    elif d == 0:
        colors_prog.append(CORES_PM['amarelo'])
    else:
        colors_prog.append(CORES_PM['vermelho'])

ax.bar(prog_dist.index.astype(str), prog_dist.values, color=colors_prog)
ax.set_title('Progressão de Pedra (2020 → 2021)', fontweight='bold')
ax.set_xlabel('Variação de Pedra (negativo = desceu)')
ax.set_ylabel('Número de Alunos')
for i, (delta, val) in enumerate(prog_dist.items()):
    ax.text(i, val + 1, str(val), ha='center', fontweight='bold')

plt.tight_layout()
plt.savefig(f'{FIGURES_DIR}p11_progressao_pedra.png')
plt.close()
print("Gráfico salvo: p11_progressao_pedra.png")

# ============================================================================
# RESUMO FINAL
# ============================================================================
print("\n" + "=" * 70)
print("RESUMO DA ANÁLISE EXPLORATÓRIA")
print("=" * 70)
print(f"""
Dataset: {df_raw.shape[0]} alunos, {df_raw.shape[1]} colunas
Período: 2020, 2021, 2022
Registros longitudinais: {df_long.shape[0]}

Gráficos gerados:
  - p1_ian_defasagem.png
  - p2_ida_desempenho.png
  - p3_ieg_ida_ipv.png
  - p4_iaa_coerencia.png
  - p5_ips_quedas.png
  - p6_ipp_ian.png
  - p7_ipv_influencias.png
  - p8_inde_combinacoes.png
  - p9_alunos_risco.png
  - p10_impacto_fases.png
  - p11_progressao_pedra.png

Datasets limpos salvos:
  - data/dataset_limpo_wide.csv
  - data/dataset_limpo_long.csv
""")
