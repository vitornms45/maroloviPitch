"""
Dashboard clínico — Backtest de Fila Hospitalar
Execute com: streamlit run dashboard.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

st.set_page_config(
    page_title="Backtest Clínico — Ceratocone",
    page_icon="👁️",
    layout="wide",
)

# ── Carrega ou roda o backtest ─────────────────────────────────────────────────
@st.cache_resource(show_spinner="Carregando modelos...")
def get_models():
    from queue_real import load_models
    return load_models()

@st.cache_data(show_spinner="Executando backtest...")
def get_results():
    import os
    if os.path.exists("backtest_results1.csv"):
        return pd.read_csv("backtest_results1.csv")
    keras_model, yolo_model = get_models()
    from queue_real import run_backtest
    return run_backtest(keras_model, yolo_model)

@st.cache_data
def get_report(_df):
    from clinical_metrics import build_full_report, metrics_to_dataframe
    report = build_full_report(_df)
    metrics_df = metrics_to_dataframe(report)
    return report, metrics_df


# ── Título ─────────────────────────────────────────────────────────────────────
st.title("👁️ Backtest — Detecção de Ceratocone em Fila Hospitalar")
st.caption("Ensemble Keras CNN + YOLOv11 · Priorização: keratoconus detectado com confiança ≥ 85%")

df = get_results()
report, metrics_df = get_report(df)
q = report["queue"]

# ── KPIs principais ────────────────────────────────────────────────────────────
st.subheader("Visão Geral")
col1, col2, col3, col4, col5, col6 = st.columns(6)

ens = report["ensemble"]
col1.metric("Acurácia (Ensemble)",   f"{ens['accuracy']:.1%}")
col2.metric("Sensibilidade",         f"{ens['sensitivity']:.1%}",
            help="Ceratocone detectado / total de ceratocones")
col3.metric("Especificidade",        f"{ens['specificity']:.1%}")
col4.metric("AUC-ROC",               f"{ens['auc_roc']:.3f}")
col5.metric("F1-score",              f"{ens['f1_score']:.3f}")
col6.metric("Taxa Falso Negativo",   f"{ens['fnr']:.1%}",
            delta=f"{-ens['fnr']:.1%}", delta_color="inverse",
            help="Ceratocone NÃO detectado — risco clínico crítico")

st.divider()

# ── Métricas da fila ───────────────────────────────────────────────────────────
st.subheader("Métricas da Fila")
qcol1, qcol2, qcol3, qcol4, qcol5 = st.columns(5)
qcol1.metric("Total de pacientes",   q["total_patients"])
qcol2.metric("Priorizados (≥85%)",   f"{q['prioritized_count']} ({q['pct_prioritized']:.0f}%)")
qcol3.metric("Avanço médio",         f"{q['avg_advance_min']:.1f} min")
qcol4.metric("Concordância modelos", f"{q['model_agreement_rate']:.1%}")
qcol5.metric("Throughput estimado",  f"{q['throughput_per_hour']:.1f} pac/h")

st.divider()

# ── Layout de gráficos ─────────────────────────────────────────────────────────
left, right = st.columns(2)

# ── Comparação de métricas entre modelos ──────────────────────────────────────
with left:
    st.subheader("Comparação entre modelos")
    metric_cols = ["accuracy", "sensitivity", "specificity", "ppv", "npv", "f1_score", "auc_roc"]
    plot_df = metrics_df[metric_cols].reset_index().melt(
        id_vars="model", var_name="Métrica", value_name="Valor"
    )
    fig_bar = px.bar(
        plot_df, x="Métrica", y="Valor", color="model", barmode="group",
        color_discrete_sequence=["#534AB7", "#0F6E56", "#D85A30"],
        labels={"Valor": "", "model": "Modelo"},
        height=380,
    )
    fig_bar.update_layout(
        yaxis_range=[0, 1], legend_title="",
        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=0, r=0, t=10, b=0),
    )
    fig_bar.update_yaxes(tickformat=".0%", gridcolor="rgba(128,128,128,0.15)")
    st.plotly_chart(fig_bar, use_container_width=True)

# ── Matriz de confusão do Ensemble ────────────────────────────────────────────
with right:
    st.subheader("Matriz de confusão — Ensemble")
    cm = ens["confusion_matrix"]
    labels = ["Keratoconus", "Normal"]
    fig_cm = px.imshow(
        cm, text_auto=True, aspect="auto",
        x=labels, y=labels,
        labels=dict(x="Predito", y="Real"),
        color_continuous_scale="Blues",
        height=380,
    )
    fig_cm.update_layout(
        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=0, r=0, t=10, b=0),
        coloraxis_showscale=False,
    )
    fig_cm.update_xaxes(side="bottom")
    st.plotly_chart(fig_cm, use_container_width=True)

st.divider()

# ── Distribuição de confiança ──────────────────────────────────────────────────
left2, right2 = st.columns(2)

with left2:
    st.subheader("Distribuição de confiança do ensemble")
    fig_hist = px.histogram(
        df, x="ensemble_conf", color="true_label_name",
        nbins=30, barmode="overlay", opacity=0.7,
        color_discrete_map={"keratoconus": "#D85A30", "normal": "#0F6E56"},
        labels={"ensemble_conf": "Confiança", "true_label_name": "Classe real"},
        height=340,
    )
    fig_hist.add_vline(x=0.85, line_dash="dash", line_color="gray",
                       annotation_text="limiar 85%", annotation_position="top right")
    fig_hist.update_layout(
        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=0, r=0, t=10, b=0),
    )
    st.plotly_chart(fig_hist, use_container_width=True)

# ── Tempo de espera — fila padrão vs prioritária ───────────────────────────────
with right2:
    st.subheader("Tempo de espera na fila")
    wait_df = pd.DataFrame({
        "Tempo agendado": df["scheduled_time_min"],
        "Tempo de chegada": df["arrival_time_min"],
        "Fila": df["prioritized"].map({True: "Prioritária (KCN ≥85%)", False: "Padrão"}),
        "Classe real": df["true_label_name"],
    })
    fig_scatter = px.scatter(
        wait_df, x="Tempo de chegada", y="Tempo agendado",
        color="Fila",
        color_discrete_map={"Prioritária (KCN ≥85%)": "#D85A30", "Padrão": "#534AB7"},
        symbol="Classe real",
        height=340,
        opacity=0.7,
    )
    fig_scatter.add_shape(type="line",
        x0=0, y0=0, x1=wait_df["Tempo de chegada"].max(),
        y1=wait_df["Tempo de chegada"].max(),
        line=dict(dash="dot", color="gray", width=1))
    fig_scatter.update_layout(
        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=0, r=0, t=10, b=0),
        legend_title="",
    )
    st.plotly_chart(fig_scatter, use_container_width=True)

st.divider()

# ── Timeline da fila ───────────────────────────────────────────────────────────
st.subheader("Timeline da fila — chegada vs atendimento")
timeline_df = df.sort_values("scheduled_time_min").head(60).copy()
timeline_df["Paciente"] = "P" + timeline_df["patient_id"].astype(str).str.zfill(3)
timeline_df["Acerto"] = timeline_df["correct"].map({True: "✓ correto", False: "✗ erro"})

fig_tl = go.Figure()
for _, row in timeline_df.iterrows():
    color = "#D85A30" if row["prioritized"] else "#534AB7"
    fig_tl.add_trace(go.Scatter(
        x=[row["arrival_time_min"], row["scheduled_time_min"]],
        y=[row["Paciente"], row["Paciente"]],
        mode="lines+markers",
        line=dict(color=color, width=2),
        marker=dict(size=6, color=color),
        showlegend=False,
        hovertemplate=(
            f"<b>{row['Paciente']}</b><br>"
            f"Chegada: {row['arrival_time_min']}<br>"
            f"Agendado: {row['scheduled_time_min']}<br>"
            f"Avanço: {row['advance_minutes']}<br>"
            f"Classe: {row['true_label_name']}<br>"
            f"Predito: {row['ensemble_pred_name']}<br>"
            f"Confiança: {row['ensemble_conf']:.1%}<br>"
            f"{row['Acerto']}<extra></extra>"
        )
    ))

fig_tl.update_layout(
    height=max(400, len(timeline_df) * 14),
    xaxis_title="Tempo desde abertura da clínica",
    plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
    margin=dict(l=0, r=0, t=10, b=0),
    showlegend=False,
)
fig_tl.update_yaxes(categoryorder="array",
                    categoryarray=timeline_df["Paciente"].tolist()[::-1])
st.plotly_chart(fig_tl, use_container_width=True)

st.caption("🔴 Ponto final = tempo agendado (após priorização)   ·   Linha longa = paciente avançado na fila")

st.divider()

# ── Tabela detalhada ───────────────────────────────────────────────────────────
with st.expander("📋 Tabela completa de pacientes"):
    display_cols = [
        "patient_id", "true_label_name", "ensemble_pred_name",
        "ensemble_conf", "keras_conf", "yolo_conf",
        "prioritized", "advance_minutes", "arrival_time_min",
        "scheduled_time_min", "correct", "inference_time_ms"
    ]
    rename = {
        "patient_id": "ID", "true_label_name": "Classe Real",
        "ensemble_pred_name": "Predito", "ensemble_conf": "Conf. Ensemble",
        "keras_conf": "Conf. Keras", "yolo_conf": "Conf. YOLO",
        "prioritized": "Prioritário", "advance_minutes": "Avanço",
        "arrival_time_min": "Chegada", "scheduled_time_min": "Agendado",
        "correct": "Acerto", "inference_time_ms": "Inferência (ms)"
    }
    st.dataframe(
        df[display_cols].rename(columns=rename).style.format({
            "Conf. Ensemble": "{:.1%}", "Conf. Keras": "{:.1%}",
            "Conf. YOLO": "{:.1%}", "Inferência (ms)": "{:.1f}",
        }).map(
            lambda v: "background-color: rgba(216,90,48,0.15)" if v is True else "",
            subset=["Prioritário"]
        ),
        use_container_width=True,
        height=400,
    )

# ── Resumo tabular de métricas ─────────────────────────────────────────────────
with st.expander("📊 Tabela de métricas por modelo"):
    st.dataframe(
        metrics_df.style.format("{:.4f}").highlight_max(
            subset=["accuracy", "sensitivity", "specificity", "f1_score", "auc_roc"],
            color="rgba(15,110,86,0.2)"
        ).highlight_min(
            subset=["fnr"],
            color="rgba(15,110,86,0.2)"
        ),
        use_container_width=True,
    )