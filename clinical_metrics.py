"""
Métricas clínicas do backtest — Detecção de Ceratocone
Calcula métricas para Keras, YOLO e Ensemble separadamente.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score,
    precision_score, recall_score, confusion_matrix,
    average_precision_score
)


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray,
                    y_prob: np.ndarray, model_name: str) -> dict:
    """
    Calcula todas as métricas clínicas relevantes para um modelo.

    Parâmetros
    ----------
    y_true     : rótulos reais (0=keratoconus, 1=normal)
    y_pred     : predições binárias
    y_prob     : probabilidade da classe 1 (normal)
    model_name : nome do modelo para identificação no relatório
    """
    cm = confusion_matrix(y_true, y_pred)

    # Extrai TN, FP, FN, TP da matriz de confusão
    # Para classe 0 (keratoconus) como positivo clínico:
    # detectar ceratocone = verdadeiro positivo clínico
    tn, fp, fn, tp = cm.ravel()

    sensitivity   = tp / (tp + fn) if (tp + fn) > 0 else 0.0  # recall da doença
    specificity   = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    ppv           = tp / (tp + fp) if (tp + fp) > 0 else 0.0  # precisão
    npv           = tn / (tn + fn) if (tn + fn) > 0 else 0.0
    accuracy      = accuracy_score(y_true, y_pred)
    f1            = f1_score(y_true, y_pred, zero_division=0)
    auc_roc       = roc_auc_score(y_true, y_prob)
    avg_precision = average_precision_score(y_true, y_prob)

    # Taxa de falsos negativos clínicos (ceratocone não detectado = risco grave)
    fnr = fn / (fn + tp) if (fn + tp) > 0 else 0.0
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0

    return {
        "model":            model_name,
        "accuracy":         accuracy,
        "sensitivity":      sensitivity,   # recall da doença
        "specificity":      specificity,
        "ppv":              ppv,
        "npv":              npv,
        "f1_score":         f1,
        "auc_roc":          auc_roc,
        "avg_precision":    avg_precision,
        "fnr":              fnr,           # taxa de falso negativo — risco clínico
        "fpr":              fpr,
        "tp":               int(tp),
        "tn":               int(tn),
        "fp":               int(fp),
        "fn":               int(fn),
        "confusion_matrix": cm,
    }


def parse_duration(duration_str: str) -> float:
    """Converte string 'Xd Xh Xm' para minutos"""
    if isinstance(duration_str, (int, float)):
        return float(duration_str)
    
    days = hours = mins = 0
    parts = duration_str.split()
    
    for part in parts:
        if part.endswith('d'):
            days = int(part[:-1])
        elif part.endswith('h'):
            hours = int(part[:-1])
        elif part.endswith('m'):
            mins = int(part[:-1])
    
    return days * 24 * 60 + hours * 60 + mins


def compute_queue_metrics(df: pd.DataFrame) -> dict:
    """
    Métricas específicas da fila hospitalar.
    """
    total        = len(df)
    prioritized  = df["prioritized"].sum()
    pct_prior    = prioritized / total * 100

    prior_df     = df[df["prioritized"]]
    standard_df  = df[~df["prioritized"]]

    avg_wait_all      = df["scheduled_time_min"].apply(parse_duration).mean()
    avg_wait_prior    = prior_df["scheduled_time_min"].apply(parse_duration).mean() if len(prior_df) > 0 else 0
    avg_wait_standard = standard_df["scheduled_time_min"].apply(parse_duration).mean() if len(standard_df) > 0 else 0
    avg_advance       = prior_df["advance_minutes"].apply(parse_duration).mean() if len(prior_df) > 0 else 0

    # Acurácia entre priorizados (são os que o modelo tem mais confiança)
    acc_prioritized  = prior_df["correct"].mean() if len(prior_df) > 0 else 0
    acc_standard     = standard_df["correct"].mean() if len(standard_df) > 0 else 0

    # Concordância entre modelos
    agreement_rate   = (df["keras_pred"] == df["yolo_pred"]).mean()

    # Throughput estimado
    arrival_times = df["arrival_time_min"].apply(parse_duration)
    total_time_hours = arrival_times.max() / 60
    throughput_ph    = total / total_time_hours if total_time_hours > 0 else 0

    # Falsos negativos entre priorizados (ceratocone que foi priorizado corretamente)
    kcn_prior = prior_df[prior_df["true_label_name"] == "keratoconus"]
    kcn_detected_in_prior = len(kcn_prior)

    return {
        "total_patients":         total,
        "prioritized_count":      int(prioritized),
        "pct_prioritized":        pct_prior,
        "avg_wait_all_min":       avg_wait_all,
        "avg_wait_prioritized":   avg_wait_prior,
        "avg_wait_standard":      avg_wait_standard,
        "avg_advance_min":        avg_advance,
        "acc_prioritized":        acc_prioritized,
        "acc_standard":           acc_standard,
        "model_agreement_rate":   agreement_rate,
        "throughput_per_hour":    throughput_ph,
        "kcn_in_priority_lane":   kcn_detected_in_prior,
    }


def build_full_report(df: pd.DataFrame) -> dict:
    """
    Gera o relatório completo com métricas dos 3 modelos + fila.
    """
    # y_prob para cada modelo: probabilidade de ser classe 1 (normal)
    # Para roc_auc a curva ROC usa "normal" como positivo,
    # mas clinicamente queremos detectar keratoconus.
    # Invertemos y_true para tornar keratoconus=1 nas métricas clínicas.
    y_true_kcn = 1 - df["true_label"].values  # keratoconus = 1 (positivo clínico)

    metrics_keras = compute_metrics(
        y_true = y_true_kcn,
        y_pred = 1 - df["keras_pred"].values,
        y_prob = 1 - df["keras_conf"].values,   # prob de keratoconus
        model_name = "Keras CNN"
    )
    metrics_yolo = compute_metrics(
        y_true = y_true_kcn,
        y_pred = 1 - df["yolo_pred"].values,
        y_prob = 1 - df["yolo_conf"].values,
        model_name = "YOLOv11"
    )
    metrics_ens = compute_metrics(
        y_true = y_true_kcn,
        y_pred = 1 - df["ensemble_pred"].values,
        y_prob = 1 - df["ensemble_conf"].values,
        model_name = "Ensemble"
    )
    queue = compute_queue_metrics(df)

    return {
        "keras":    metrics_keras,
        "yolo":     metrics_yolo,
        "ensemble": metrics_ens,
        "queue":    queue,
    }


def metrics_to_dataframe(report: dict) -> pd.DataFrame:
    """Converte métricas dos modelos em DataFrame comparativo."""
    cols = ["model", "accuracy", "sensitivity", "specificity",
            "ppv", "npv", "f1_score", "auc_roc", "fnr", "tp", "tn", "fp", "fn"]
    rows = [
        {k: report["keras"][k]    for k in cols},
        {k: report["yolo"][k]     for k in cols},
        {k: report["ensemble"][k] for k in cols},
    ]
    return pd.DataFrame(rows).set_index("model")