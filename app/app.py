#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
=============================================================================
DATATHON PASSOS MÁGICOS - Aplicação Streamlit
Sistema de Alerta Precoce: Previsão de Risco no Ano Seguinte (T → T+1)
=============================================================================
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
import plotly.express as px
import plotly.graph_objects as go
import os

# ============================================================================
# CONFIGURAÇÃO
# ============================================================================
st.set_page_config(
    page_title="Passos Mágicos - Sistema de Alerta Precoce",
    page_icon="✨",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS customizado
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    .main .block-container { padding-top: 2rem; max-width: 1200px; }

    .hero-banner {
        background: linear-gradient(135deg, #1B3A5C 0%, #2E6B9E 100%);
        padding: 2rem 2.5rem;
        border-radius: 12px;
        margin-bottom: 2rem;
        color: white;
    }
    .hero-banner h1 {
        font-family: 'Inter', sans-serif;
        font-size: 2rem;
        font-weight: 700;
        margin-bottom: 0.5rem;
        color: white;
    }
    .hero-banner p {
        font-family: 'Inter', sans-serif;
        font-size: 1.05rem;
        opacity: 0.9;
        margin: 0;
        color: white;
    }

    .metric-card {
        background: white;
        border-radius: 10px;
        padding: 1.5rem;
        box-shadow: 0 2px 8px rgba(0,0,0,0.06);
        border-left: 4px solid #2E6B9E;
        margin-bottom: 1rem;
    }
    .metric-card h3 {
        font-family: 'Inter', sans-serif;
        font-size: 0.85rem;
        font-weight: 600;
        color: #64748B;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        margin-bottom: 0.3rem;
    }
    .metric-card .value {
        font-family: 'Inter', sans-serif;
        font-size: 2rem;
        font-weight: 700;
        color: #1B3A5C;
    }

    .risk-high {
        background: linear-gradient(135deg, #FEE2E2, #FECACA);
        border-radius: 12px;
        padding: 2rem;
        border-left: 5px solid #DC2626;
        margin: 1rem 0;
    }
    .risk-low {
        background: linear-gradient(135deg, #DCFCE7, #BBF7D0);
        border-radius: 12px;
        padding: 2rem;
        border-left: 5px solid #16A34A;
        margin: 1rem 0;
    }

    .info-box {
        background: #F0F9FF;
        border-radius: 10px;
        padding: 1.2rem 1.5rem;
        border-left: 4px solid #2E6B9E;
        margin: 1rem 0;
        font-family: 'Inter', sans-serif;
        font-size: 0.95rem;
        color: #334155;
    }

    .welcome-card {
        background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
        padding: 2rem;
        border-radius: 15px;
        border-left: 5px solid #E87722;
        margin-bottom: 1.5rem;
    }

    .stButton > button {
        background: linear-gradient(135deg, #1B3A5C 0%, #2E6B9E 100%);
        color: white;
        border: none;
        border-radius: 10px;
        padding: 0.75rem 2rem;
        font-size: 1.1rem;
        font-weight: bold;
        width: 100%;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================================
# CARREGAR DADOS E MODELO
# ============================================================================
@st.cache_resource
def load_model():
    base_path = os.path.dirname(os.path.abspath(__file__))
    models_path = os.path.join(base_path, '..', 'models')
    model = joblib.load(os.path.join(models_path, 'modelo_risco.pkl'))
    scaler = joblib.load(os.path.join(models_path, 'scaler.pkl'))
    with open(os.path.join(models_path, 'metadata.json'), 'r') as f:
        metadata = json.load(f)
    with open(os.path.join(models_path, 'threshold.txt'), 'r') as f:
        threshold = float(f.read().strip())
    with open(os.path.join(models_path, 'features.txt'), 'r') as f:
        features = [line.strip() for line in f.readlines() if line.strip()]
    return model, scaler, metadata, threshold, features

@st.cache_data
def load_data():
    base_path = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(base_path, '..', 'data')
    df = pd.read_csv(os.path.join(data_path, 'dataset_limpo_long.csv'))
    return df

try:
    model, scaler, metadata, threshold, feature_names = load_model()
    df = load_data()
    model_loaded = True
except Exception as e:
    model_loaded = False
    st.error(f"Erro ao carregar modelo: {e}")
    df = pd.DataFrame()
    metadata = {}
    threshold = 0.5

# ============================================================================
# SIDEBAR
# ============================================================================
with st.sidebar:
    st.markdown("### ✨ Passos Mágicos")
    st.markdown("**Sistema de Alerta Precoce**")
    st.markdown("---")
    page = st.radio(
        "Navegação",
        ["🏠 Visão Geral", "🔮 Previsão de Risco", "📊 Análise Exploratória", "📖 Sobre o Modelo"],
        label_visibility="collapsed"
    )
    st.markdown("---")
    if model_loaded:
        st.success("Modelo carregado")
        st.caption(f"AUC-ROC: {metadata.get('auc_roc', 0):.3f}")
        st.caption(f"Recall: {metadata.get('recall', 0):.3f}")
    st.markdown(
        '<div class="info-box">'
        '<strong>Forecasting T→T+1</strong><br>'
        'Usa indicadores do ano atual para prever o risco no ano seguinte.'
        '</div>',
        unsafe_allow_html=True
    )

# ============================================================================
# PÁGINA 1: VISÃO GERAL
# ============================================================================
if page == "🏠 Visão Geral":
    st.markdown(
        '<div class="hero-banner">'
        '<h1>Sistema de Alerta Precoce</h1>'
        '<p>Previsão de risco de defasagem educacional para o próximo ano letivo. '
        'Identifique alunos que precisam de atenção antes que a queda aconteça.</p>'
        '</div>',
        unsafe_allow_html=True
    )

    st.markdown("""
    <div class="welcome-card">
        <h3>Olá! Seja muito bem-vindo(a)!</h3>
        <p>Sou a assistente virtual da <strong>Associação Passos Mágicos</strong>.
        Estou aqui para ajudá-lo(a) a identificar alunos que possam precisar de
        atenção especial em sua jornada educacional.</p>
        <p>Com base nos indicadores do PEDE, nosso modelo de inteligência artificial
        analisa o perfil de cada aluno e estima a probabilidade de risco de defasagem
        <strong>no próximo ano letivo</strong>.</p>
        <p><strong>Use o menu lateral para navegar entre as funcionalidades.</strong></p>
    </div>
    """, unsafe_allow_html=True)

    if model_loaded:
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.markdown(
                '<div class="metric-card"><h3>Alunos Analisados</h3>'
                f'<div class="value">{df["NOME"].nunique()}</div></div>',
                unsafe_allow_html=True
            )
        with col2:
            st.markdown(
                '<div class="metric-card"><h3>Período</h3>'
                '<div class="value">2020-2022</div></div>',
                unsafe_allow_html=True
            )
        with col3:
            st.markdown(
                f'<div class="metric-card"><h3>AUC-ROC do Modelo</h3>'
                f'<div class="value">{metadata.get("auc_roc", 0):.1%}</div></div>',
                unsafe_allow_html=True
            )
        with col4:
            st.markdown(
                f'<div class="metric-card"><h3>Recall (Sensibilidade)</h3>'
                f'<div class="value">{metadata.get("recall", 0):.1%}</div></div>',
                unsafe_allow_html=True
            )

        st.markdown("---")
        st.markdown("### Como funciona o Sistema de Alerta Precoce?")

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("#### Abordagem de Forecasting")
            st.markdown("""
            | Aspecto | Detalhe |
            |---------|---------|
            | **Modelo** | Random Forest (500 árvores) |
            | **Features** | 28 variáveis (base + engenharia) |
            | **Treino** | Dados 2020 → Risco em 2021 |
            | **Teste** | Dados 2021 → Risco em 2022 |
            | **Balanceamento** | SMOTE (oversampling) |
            | **Threshold** | Otimizado para F1 |
            """)

        with col2:
            st.markdown("#### Principais Indicadores de Risco")
            st.markdown("""
            Os dados mostram que os alunos que entram em risco no ano seguinte já
            apresentam sinais claros no ano atual:

            - **IDA baixo** (média 3.4 vs 6.7 dos sem risco)
            - **IEG baixo** (média 4.3 vs 7.7 dos sem risco)
            - **INDE próximo do limiar** de 5.506
            - **Múltiplos indicadores abaixo de 5.0**
            """)

        st.markdown("---")
        st.subheader("Distribuição dos Alunos por Classificação (Pedra)")
        pedra_counts = df['PEDRA'].value_counts()
        fig = px.pie(
            values=pedra_counts.values,
            names=pedra_counts.index,
            color=pedra_counts.index,
            color_discrete_map={
                'Quartzo': '#D4A574', 'Agata': '#7B8D9E',
                'Ametista': '#9B59B6', 'Topazio': '#F1C40F'
            },
            hole=0.4
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)

# ============================================================================
# PÁGINA 2: PREVISÃO DE RISCO
# ============================================================================
elif page == "🔮 Previsão de Risco":
    st.markdown(
        '<div class="hero-banner">'
        '<h1>🔮 Previsão de Risco para o Próximo Ano</h1>'
        '<p>Insira os indicadores atuais do aluno para prever o risco de defasagem no ano seguinte.</p>'
        '</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="info-box">'
        '💡 <strong>Dica:</strong> Preencha os 7 indicadores base do aluno (dados do ano atual). '
        'O sistema calculará automaticamente as features derivadas e fará a previsão para o próximo ano.'
        '</div>',
        unsafe_allow_html=True
    )

    with st.form("prediction_form"):
        st.markdown("#### Indicadores do Aluno (Ano Atual)")

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            ida = st.number_input("IDA (Aprendizagem)", 0.0, 10.0, 5.0, 0.1,
                                  help="Indicador de Desempenho Acadêmico")
            ieg = st.number_input("IEG (Engajamento)", 0.0, 10.0, 5.0, 0.1,
                                  help="Indicador de Engajamento")
        with col2:
            iaa = st.number_input("IAA (Autoavaliação)", 0.0, 10.0, 7.0, 0.1,
                                  help="Indicador de Autoavaliação")
            ips = st.number_input("IPS (Psicossocial)", 0.0, 10.0, 6.0, 0.1,
                                  help="Indicador Psicossocial")
        with col3:
            ipp = st.number_input("IPP (Psicopedagógico)", 0.0, 10.0, 5.0, 0.1,
                                  help="Indicador Psicopedagógico")
            ipv = st.number_input("IPV (Ponto de Virada)", 0.0, 10.0, 5.0, 0.1,
                                  help="Indicador de Ponto de Virada")
        with col4:
            ian = st.number_input("IAN (Adequação ao Nível)", 0.0, 10.0, 5.0, 0.1,
                                  help="Indicador de Adequação ao Nível")
            fase = st.selectbox("Fase", [1, 2, 3, 4, 5, 6, 7, 8],
                                help="Fase atual do aluno no programa")

        col5, col6 = st.columns(2)
        with col5:
            pedra = st.selectbox("Pedra Atual", ["Quartzo", "Ágata", "Ametista", "Topázio"],
                                 help="Classificação atual do aluno")
        with col6:
            ponto_virada = st.selectbox("Atingiu Ponto de Virada?", ["Não", "Sim"],
                                        help="Se o aluno atingiu o ponto de virada neste ano")

        submitted = st.form_submit_button("🔮 Prever Risco para o Próximo Ano", use_container_width=True)

    if submitted and model_loaded:
        # Calcular INDE aproximado
        inde = (ida + ieg + iaa + ips + ipp + ipv + ian) / 7

        # Mapear pedra
        pedra_map = {"Quartzo": 0, "Ágata": 1, "Ametista": 2, "Topázio": 3}
        pedra_ord = pedra_map[pedra]
        pv = 1 if ponto_virada == "Sim" else 0

        indicators = [ida, ieg, iaa, ips, ipp, ipv]

        # Construir feature vector (28 features)
        features = np.array([[
            ida,                                    # IDA_T
            ieg,                                    # IEG_T
            iaa,                                    # IAA_T
            ips,                                    # IPS_T
            ipp,                                    # IPP_T
            ipv,                                    # IPV_T
            ian,                                    # IAN_T
            inde,                                   # INDE_T
            fase,                                   # FASE_T
            pedra_ord,                              # PEDRA_ORD_T
            pv,                                     # PV_T
            ida / (ieg + 0.01),                     # IDA_IEG_RATIO_T
            ida / (ips + 0.01),                     # IDA_IPS_RATIO_T
            ieg * ipv,                              # IEG_IPV_PROD_T
            ida * ipp,                              # IDA_IPP_PROD_T
            np.mean(indicators),                    # MEDIA_IND_T
            np.std(indicators),                     # STD_IND_T
            min(indicators),                        # MIN_IND_T
            max(indicators) - min(indicators),      # RANGE_IND_T
            iaa - ida,                              # IAA_IDA_DIFF_T
            iaa - inde,                             # IAA_INDE_DIFF_T
            1 if ida < 5.0 else 0,                  # IDA_BAIXO_T
            1 if ieg < 5.0 else 0,                  # IEG_BAIXO_T
            1 if ian < 5.0 else 0,                  # IAN_BAIXO_T
            1 if inde < 5.506 else 0,               # RISCO_T
            min(3, (1 if ida < 5.0 else 0) + (1 if ieg < 5.0 else 0) + (1 if ian < 5.0 else 0)),  # MULTI_ALERTA_T
            inde - 5.506,                           # DISTANCIA_QUARTZO_T
            (inde - 5.506) ** 2,                    # DISTANCIA_QUARTZO_SQ_T
        ]])

        # Escalar e prever
        features_scaled = scaler.transform(features)
        prob = model.predict_proba(features_scaled)[0][1]
        risco = prob >= threshold

        st.markdown("---")
        st.markdown("### Resultado da Previsão")

        if risco:
            st.markdown(
                f'<div class="risk-high">'
                f'<h2 style="color: #DC2626; margin: 0;">⚠️ ALERTA: Risco Elevado para o Próximo Ano</h2>'
                f'<p style="font-size: 1.1rem; margin-top: 0.5rem; color: #7F1D1D;">'
                f'Probabilidade de entrar em risco: <strong>{prob:.1%}</strong><br>'
                f'Este aluno apresenta indicadores que sugerem risco de queda para a classificação '
                f'Quartzo (INDE &lt; 5.506) no próximo ano letivo.</p>'
                f'</div>',
                unsafe_allow_html=True
            )

            st.markdown("#### 💡 Recomendações de Intervenção")
            recs = []
            if ida < 5.0:
                recs.append("📚 **Reforço Acadêmico Urgente:** IDA abaixo de 5.0 indica dificuldades de aprendizagem significativas. Considere tutoria individualizada.")
            if ieg < 5.0:
                recs.append("🎯 **Programa de Engajamento:** IEG baixo é o principal preditor de risco. Investigue as causas da desmotivação e crie estratégias personalizadas.")
            if ian < 5.0:
                recs.append("📐 **Adequação ao Nível:** IAN baixo indica defasagem série-idade. Avalie a necessidade de nivelamento.")
            if iaa - ida > 2.0:
                recs.append("🪞 **Calibração da Autoavaliação:** O aluno superestima seu desempenho em mais de 2 pontos. Trabalhe a autoconsciência.")
            if not recs:
                recs.append("🔍 **Monitoramento Intensivo:** Embora nenhum indicador isolado esteja crítico, a combinação dos fatores gera risco. Acompanhe de perto.")

            for rec in recs:
                st.markdown(rec)

        else:
            st.markdown(
                f'<div class="risk-low">'
                f'<h2 style="color: #16A34A; margin: 0;">✅ Baixo Risco para o Próximo Ano</h2>'
                f'<p style="font-size: 1.1rem; margin-top: 0.5rem; color: #14532D;">'
                f'Probabilidade de entrar em risco: <strong>{prob:.1%}</strong><br>'
                f'Os indicadores atuais sugerem que o aluno tem boas chances de manter ou melhorar '
                f'sua classificação no próximo ano letivo.</p>'
                f'</div>',
                unsafe_allow_html=True
            )

        # Radar chart dos indicadores
        st.markdown("#### Perfil do Aluno vs Média dos Alunos em Risco")
        categories = ['IDA', 'IEG', 'IAA', 'IPS', 'IPP', 'IPV', 'IAN']
        values_aluno = [ida, ieg, iaa, ips, ipp, ipv, ian]
        values_risco = [3.4, 4.3, 7.2, 6.1, 4.0, 3.8, 3.5]
        values_sem_risco = [6.7, 7.7, 7.8, 7.0, 6.5, 6.8, 7.5]

        fig = go.Figure()
        fig.add_trace(go.Scatterpolar(
            r=values_aluno + [values_aluno[0]],
            theta=categories + [categories[0]],
            fill='toself', name='Aluno Avaliado',
            line_color='#1B3A5C', fillcolor='rgba(27,58,92,0.15)'
        ))
        fig.add_trace(go.Scatterpolar(
            r=values_risco + [values_risco[0]],
            theta=categories + [categories[0]],
            fill='toself', name='Perfil Médio em Risco',
            line_color='#DC2626', fillcolor='rgba(220,38,38,0.08)'
        ))
        fig.add_trace(go.Scatterpolar(
            r=values_sem_risco + [values_sem_risco[0]],
            theta=categories + [categories[0]],
            fill='toself', name='Perfil Médio Sem Risco',
            line_color='#16A34A', fillcolor='rgba(22,163,74,0.08)'
        ))
        fig.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[0, 10])),
            showlegend=True, height=450,
            font=dict(family='Inter', size=12)
        )
        st.plotly_chart(fig, use_container_width=True)

# ============================================================================
# PÁGINA 3: ANÁLISE EXPLORATÓRIA
# ============================================================================
elif page == "📊 Análise Exploratória":
    st.markdown(
        '<div class="hero-banner">'
        '<h1>📊 Análise Exploratória dos Dados</h1>'
        '<p>Explore a evolução dos indicadores dos alunos da Passos Mágicos ao longo dos anos.</p>'
        '</div>',
        unsafe_allow_html=True
    )

    tab1, tab2, tab3 = st.tabs(["📈 Evolução Temporal", "🏷️ Por Classificação", "🔗 Correlações"])

    with tab1:
        indicadores = ['IDA', 'IEG', 'IAA', 'IPS', 'IPP', 'IPV', 'IAN', 'INDE']
        selected = st.multiselect("Selecione indicadores:", indicadores, default=['IDA', 'IEG', 'INDE'])

        if selected:
            medias = df.groupby('ANO')[selected].mean().reset_index()
            medias_melted = medias.melt(id_vars='ANO', var_name='Indicador', value_name='Média')
            fig = px.line(medias_melted, x='ANO', y='Média', color='Indicador',
                          markers=True, title='Evolução dos Indicadores (2020-2022)',
                          color_discrete_sequence=['#1B3A5C', '#2E6B9E', '#E87722', '#2E8B57',
                                                   '#C0392B', '#F1C40F', '#8B5CF6', '#EC4899'])
            fig.update_layout(xaxis=dict(dtick=1), font=dict(family='Inter', size=12), height=450)
            st.plotly_chart(fig, use_container_width=True)

    with tab2:
        ano_sel = st.selectbox("Selecione o ano:", sorted(df['ANO'].unique()), index=2)
        df_ano = df[df['ANO'] == ano_sel]

        if 'PEDRA' in df_ano.columns:
            fig = px.box(df_ano, x='PEDRA', y='INDE', color='PEDRA',
                         title=f'Distribuição do INDE por Classificação ({ano_sel})',
                         category_orders={'PEDRA': ['Quartzo', 'Agata', 'Ametista', 'Topazio']},
                         color_discrete_sequence=['#C0392B', '#E87722', '#8B5CF6', '#2E8B57'])
            fig.update_layout(font=dict(family='Inter', size=12), height=450)
            st.plotly_chart(fig, use_container_width=True)

        col1, col2 = st.columns(2)
        with col1:
            pedra_counts = df_ano['PEDRA'].value_counts()
            fig = px.pie(values=pedra_counts.values, names=pedra_counts.index,
                         title=f'Distribuição de Pedras ({ano_sel})',
                         color_discrete_sequence=['#C0392B', '#E87722', '#8B5CF6', '#2E8B57'])
            fig.update_layout(font=dict(family='Inter', size=12), height=400)
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            medias_pedra = df_ano.groupby('PEDRA')[['IDA', 'IEG', 'IAA']].mean()
            medias_pedra = medias_pedra.reindex(['Quartzo', 'Agata', 'Ametista', 'Topazio'])
            fig = px.bar(medias_pedra.reset_index(), x='PEDRA', y=['IDA', 'IEG', 'IAA'],
                         barmode='group', title=f'Indicadores Médios por Pedra ({ano_sel})',
                         color_discrete_sequence=['#1B3A5C', '#2E6B9E', '#E87722'])
            fig.update_layout(font=dict(family='Inter', size=12), height=400)
            st.plotly_chart(fig, use_container_width=True)

    with tab3:
        ano_corr = st.selectbox("Ano para correlação:", sorted(df['ANO'].unique()), index=2, key='corr_ano')
        df_corr = df[df['ANO'] == ano_corr][['IDA', 'IEG', 'IAA', 'IPS', 'IPP', 'IPV', 'IAN', 'INDE']].dropna()
        corr = df_corr.corr()

        fig = px.imshow(corr, text_auto='.2f', aspect='auto',
                        title=f'Matriz de Correlação ({ano_corr})',
                        color_continuous_scale='RdBu_r', zmin=-1, zmax=1)
        fig.update_layout(font=dict(family='Inter', size=12), height=500)
        st.plotly_chart(fig, use_container_width=True)

# ============================================================================
# PÁGINA 4: SOBRE O MODELO
# ============================================================================
elif page == "📖 Sobre o Modelo":
    st.markdown(
        '<div class="hero-banner">'
        '<h1>📖 Sobre o Modelo Preditivo</h1>'
        '<p>Detalhes técnicos do modelo de forecasting e sua abordagem metodológica.</p>'
        '</div>',
        unsafe_allow_html=True
    )

    st.markdown("### Abordagem: Forecasting Verdadeiro (T → T+1)")
    st.markdown("""
    Diferente de modelos tradicionais que usam dados do mesmo ano para classificar o aluno,
    nosso modelo utiliza os indicadores do **Ano T** (ano atual) para prever se o aluno
    entrará em risco de defasagem no **Ano T+1** (próximo ano).

    Isso é fundamental porque:
    - **Elimina o data leakage:** Não usamos informações do futuro para prever o futuro.
    - **Cria valor real:** A equipe pedagógica pode agir com antecedência.
    - **Métricas honestas:** Os resultados refletem a dificuldade real do problema.
    """)

    st.markdown("### Métricas do Modelo")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("AUC-ROC", f"{metadata.get('auc_roc', 0):.1%}")
    with col2:
        st.metric("Recall", f"{metadata.get('recall', 0):.1%}")
    with col3:
        st.metric("Precision", f"{metadata.get('precision', 0):.1%}")
    with col4:
        st.metric("F1-Score", f"{metadata.get('f1_score', 0):.1%}")

    st.markdown("### Comparação de Modelos Testados")
    models_data = []
    for name, m in metadata.get('all_models', {}).items():
        models_data.append({
            'Modelo': name,
            'Accuracy': f"{m['accuracy']:.1%}",
            'Precision': f"{m['precision']:.1%}",
            'Recall': f"{m['recall']:.1%}",
            'F1-Score': f"{m['f1_score']:.1%}",
            'AUC-ROC': f"{m['auc_roc']:.1%}",
            'CV AUC': f"{m['cv_auc_mean']:.1%}",
        })
    if models_data:
        st.dataframe(pd.DataFrame(models_data), use_container_width=True, hide_index=True)

    st.markdown("### Interpretação das Métricas")
    st.markdown("""
    | Métrica | Valor | Interpretação |
    |---------|-------|---------------|
    | **AUC-ROC** | 76.0% | O modelo tem boa capacidade de discriminar entre alunos em risco e sem risco |
    | **Recall** | 53.5% | Captura mais da metade dos alunos que realmente entrarão em risco |
    | **Precision** | 23.2% | Para cada 4 alertas, ~1 se confirma (trade-off aceitável para alerta precoce) |
    | **CV AUC** | 92.5% | Alta estabilidade em validação cruzada no conjunto de treino |
    """)

    st.markdown(
        '<div class="info-box">'
        '💡 <strong>Por que a Precision é baixa?</strong> Em um sistema de alerta precoce, '
        'é preferível ter mais falsos positivos (alertar alunos que não precisam) do que '
        'falsos negativos (não alertar alunos que realmente precisam). O custo de não intervir '
        'é muito maior do que o custo de intervir desnecessariamente.'
        '</div>',
        unsafe_allow_html=True
    )

    st.markdown("### Features Utilizadas (28)")
    features_desc = {
        'IDA_T': 'Indicador de Desempenho Acadêmico',
        'IEG_T': 'Indicador de Engajamento',
        'IAA_T': 'Indicador de Autoavaliação',
        'IPS_T': 'Indicador Psicossocial',
        'IPP_T': 'Indicador Psicopedagógico',
        'IPV_T': 'Indicador de Ponto de Virada',
        'IAN_T': 'Indicador de Adequação ao Nível',
        'INDE_T': 'Índice de Desenvolvimento Educacional',
        'FASE_T': 'Fase do aluno no programa',
        'PEDRA_ORD_T': 'Classificação ordinal (Quartzo=0 a Topázio=3)',
        'PV_T': 'Se atingiu Ponto de Virada (0/1)',
        'IDA_IEG_RATIO_T': 'Razão Aprendizagem/Engajamento',
        'IDA_IPS_RATIO_T': 'Razão Acadêmico/Psicossocial',
        'IEG_IPV_PROD_T': 'Interação Engajamento x Ponto de Virada',
        'IDA_IPP_PROD_T': 'Interação Aprendizagem x Psicopedagógico',
        'MEDIA_IND_T': 'Média dos 6 indicadores base',
        'STD_IND_T': 'Desvio padrão dos indicadores (heterogeneidade)',
        'MIN_IND_T': 'Menor indicador (ponto fraco)',
        'RANGE_IND_T': 'Amplitude entre maior e menor indicador',
        'IAA_IDA_DIFF_T': 'Viés de autoavaliação (IAA - IDA)',
        'IAA_INDE_DIFF_T': 'Viés de autoavaliação vs INDE',
        'IDA_BAIXO_T': 'Flag: IDA < 5.0',
        'IEG_BAIXO_T': 'Flag: IEG < 5.0',
        'IAN_BAIXO_T': 'Flag: IAN < 5.0',
        'RISCO_T': 'Flag: já está em risco (INDE < 5.506)',
        'MULTI_ALERTA_T': 'Contagem de indicadores baixos (0-3)',
        'DISTANCIA_QUARTZO_T': 'Distância do limiar de risco',
        'DISTANCIA_QUARTZO_SQ_T': 'Distância ao quadrado (não-linearidade)',
    }
    feat_df = pd.DataFrame([
        {'Feature': k, 'Descrição': v} for k, v in features_desc.items()
    ])
    st.dataframe(feat_df, use_container_width=True, hide_index=True)
