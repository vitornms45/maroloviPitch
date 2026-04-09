"""
Simulador de fila hospitalar — Backtest com Keras + YOLOv11
Lógica: pacientes com confiança ≥85% avançam 10–20 min na fila
dependendo do número de vagas prioritárias disponíveis.
"""

import os
import time
import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Optional
from PIL import Image
import torch

# ── Configurações ──────────────────────────────────────────────────────────────
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_SCRIPT_DIR)

TEST_DIR          = os.path.join(_PROJECT_ROOT, "TCC-Modelos", "test")
KERAS_MODEL_PATH  = os.path.join(_PROJECT_ROOT, "TCC-Modelos", "modeloFinalNaoAguentoMaisKeras.h5")
YOLO_MODEL_PATH   = os.path.join(_PROJECT_ROOT, "TCC-Modelos", "Modelo_Yolov11_Improve_Final.pt")
IMG_SIZE          = (224, 224)

CONFIDENCE_THRESHOLD = 0.85   # limite para priorização
PRIORITY_SLOTS       = 5      # vagas simultâneas na fila prioritária
PRIORITY_MIN_MIN     = 10     # mínimo de minutos adiantados
PRIORITY_MAX_MIN     = 20     # máximo de minutos adiantados
ARRIVAL_RATE         = 8.0    # pacientes por hora (lambda Poisson)
KERAS_WEIGHT         = 0.45
YOLO_WEIGHT          = 0.55   # YOLO ligeiramente superior na sua pesquisa


@dataclass
class Patient:
    patient_id:        int
    image_path:        str
    true_label:        int                  # 0=keratoconus, 1=normal
    arrival_time:      float               # minutos desde abertura
    scheduled_time:    float = 0.0         # tempo após priorização
    keras_pred:        int   = -1
    keras_conf:        float = 0.0
    yolo_pred:         int   = -1
    yolo_conf:         float = 0.0
    ensemble_pred:     int   = -1
    ensemble_conf:     float = 0.0
    prioritized:       bool  = False
    advance_minutes:   float = 0.0
    inference_time_ms: float = 0.0
    correct:           bool  = False


def format_duration(minutes: float) -> str:
    """Converte minutos para formato 'Xd Xh Xm'"""
    total_minutes = int(minutes)
    days_part = total_minutes // (24 * 60)
    hours_part = (total_minutes % (24 * 60)) // 60
    mins_part = total_minutes % 60
    return f"{days_part}d {hours_part}h {mins_part}m"


def load_models():
    """Carrega Keras e YOLO uma única vez."""
    from tensorflow.keras.models import load_model
    from ultralytics import YOLO as UltralyticsYOLO

    print("Carregando modelos...")
    keras_model = load_model(KERAS_MODEL_PATH)
    yolo_model  = UltralyticsYOLO(YOLO_MODEL_PATH)
    print("Modelos carregados.")
    return keras_model, yolo_model


def preprocess_keras(path: str) -> np.ndarray:
    """Pré-processa imagem para Keras (array 4D normalizado)."""
    from tensorflow.keras.preprocessing import image as kimage
    img = kimage.load_img(path, target_size=IMG_SIZE)
    arr = kimage.img_to_array(img) / 255.0
    return np.expand_dims(arr, axis=0)


def predict_patient(path: str, keras_model, yolo_model) -> dict:
    """
    Roda os dois modelos e calcula ensemble por média ponderada.
    Retorna dict com predições, confiança e tempo de inferência.
    """
    t0 = time.perf_counter()

    # ── Keras ──────────────────────────────────────────────────────────────────
    keras_input = preprocess_keras(path)
    keras_raw   = float(keras_model.predict(keras_input, verbose=0)[0][0])
    # sigmoid: >0.5 → classe 1 (normal), ≤0.5 → classe 0 (keratoconus)
    keras_pred  = 1 if keras_raw > 0.5 else 0
    keras_conf  = keras_raw if keras_pred == 1 else (1.0 - keras_raw)

    # ── YOLOv11 ────────────────────────────────────────────────────────────────
    img_pil  = Image.open(path).convert("RGB").resize(IMG_SIZE)
    result   = yolo_model(img_pil, imgsz=224, verbose=False)[0]
    probs    = result.probs.data.cpu().numpy()         # [p_keratoconus, p_normal]
    yolo_pred = int(np.argmax(probs))
    yolo_conf = float(probs[yolo_pred])

    # ── Ensemble ───────────────────────────────────────────────────────────────
    # Probabilidade ponderada de ser "normal" (classe 1)
    p_normal_keras = keras_raw
    p_normal_yolo  = float(probs[1])
    p_normal_ens   = KERAS_WEIGHT * p_normal_keras + YOLO_WEIGHT * p_normal_yolo

    ens_pred = 1 if p_normal_ens > 0.5 else 0
    ens_conf = p_normal_ens if ens_pred == 1 else (1.0 - p_normal_ens)

    elapsed_ms = (time.perf_counter() - t0) * 1000

    return {
        "keras_pred":        keras_pred,
        "keras_conf":        keras_conf,
        "yolo_pred":         yolo_pred,
        "yolo_conf":         yolo_conf,
        "ensemble_pred":     ens_pred,
        "ensemble_conf":     ens_conf,
        "inference_time_ms": elapsed_ms,
    }


def simulate_arrivals(n_patients: int, rate: float = ARRIVAL_RATE) -> np.ndarray:
    """
    Gera tempos de chegada usando processo de Poisson.
    rate = pacientes/hora → intervalo médio = 60/rate minutos.
    """
    inter_arrival = np.random.exponential(60.0 / rate, size=n_patients)
    return np.cumsum(inter_arrival)


def prioritize_queue(patients: list[Patient]) -> list[Patient]:
    """
    Pacientes com ensemble_conf ≥ threshold avançam na fila.
    Quanto mais vagas livres no slot prioritário, mais eles avançam.
    """
    priority_queue_usage = 0  # contagem atual de pacientes prioritários no slot

    for p in patients:
        p.scheduled_time = p.arrival_time  # padrão: sem mudança

        if (p.ensemble_conf >= CONFIDENCE_THRESHOLD) and p.ensemble_pred == 0:
            # Vagas livres → avanço máximo; fila cheia → avanço mínimo
            occupancy_ratio  = min(priority_queue_usage / PRIORITY_SLOTS, 1.0)
            advance          = PRIORITY_MAX_MIN - occupancy_ratio * (PRIORITY_MAX_MIN - PRIORITY_MIN_MIN)
            p.advance_minutes = advance
            p.scheduled_time  = max(0.0, p.arrival_time - advance)
            p.prioritized     = True
            priority_queue_usage += 1
        else:
            # Paciente sai do slot prioritário conforme chega na consulta
            priority_queue_usage = max(0, priority_queue_usage - 1)

    return patients


def run_backtest(keras_model=None, yolo_model=None) -> pd.DataFrame:
    """
    Executa o backtest completo:
    1. Coleta imagens do diretório test/
    2. Simula chegadas via Poisson
    3. Roda inferência em cada paciente
    4. Aplica lógica de priorização
    5. Retorna DataFrame com todos os dados
    """
    if keras_model is None or yolo_model is None:
        keras_model, yolo_model = load_models()

    # ── Coleta imagens ─────────────────────────────────────────────────────────
    class_names = sorted(os.listdir(TEST_DIR))
    label_map   = {name: idx for idx, name in enumerate(class_names)}
    print(f"Classes encontradas: {label_map}")

    all_paths, all_labels = [], []
    for cls in class_names:
        cls_dir = os.path.join(TEST_DIR, cls)
        if not os.path.isdir(cls_dir):
            continue
        for fname in os.listdir(cls_dir):
            if fname.lower().endswith((".jpg", ".jpeg", ".png")):
                all_paths.append(os.path.join(cls_dir, fname))
                all_labels.append(label_map[cls])

    n = len(all_paths)
    print(f"Total de imagens: {n}")

    # ── Simula chegadas ────────────────────────────────────────────────────────
    arrival_times = simulate_arrivals(n)

    # Embaralha para simular ordem aleatória de chegada
    idx = np.random.permutation(n)
    all_paths  = [all_paths[i]  for i in idx]
    all_labels = [all_labels[i] for i in idx]

    # ── Inferência paciente a paciente ─────────────────────────────────────────
    patients: list[Patient] = []
    for i, (path, label, arrival) in enumerate(zip(all_paths, all_labels, arrival_times)):
        print(f"  [{i+1:03d}/{n}] {os.path.basename(path)}", end="\r")
        preds = predict_patient(path, keras_model, yolo_model)
        p = Patient(
            patient_id    = i + 1,
            image_path    = path,
            true_label    = label,
            arrival_time  = arrival,
            **preds,
        )
        p.correct = (p.ensemble_pred == p.true_label)
        patients.append(p)

    print(f"\nInferência concluída.")

    # ── Priorização ────────────────────────────────────────────────────────────
    patients.sort(key=lambda p: p.arrival_time)
    patients = prioritize_queue(patients)

    # ── Converte para DataFrame ────────────────────────────────────────────────
    rows = []
    for p in patients:
        rows.append({
            "patient_id":        p.patient_id,
            "image_path":        p.image_path,
            "true_label":        p.true_label,
            "true_label_name":   class_names[p.true_label],
            "arrival_time_min":  format_duration(p.arrival_time),
            "scheduled_time_min": format_duration(p.scheduled_time),
            "advance_minutes":   format_duration(p.advance_minutes),
            "prioritized":       p.prioritized,
            "keras_pred":        p.keras_pred,
            "keras_conf":        p.keras_conf,
            "yolo_pred":         p.yolo_pred,
            "yolo_conf":         p.yolo_conf,
            "ensemble_pred":     p.ensemble_pred,
            "ensemble_conf":     p.ensemble_conf,
            "ensemble_pred_name": class_names[p.ensemble_pred],
            "correct":           p.correct,
            "inference_time_ms": p.inference_time_ms,
        })

    df = pd.DataFrame(rows)
    df.to_csv("backtest_results.csv", index=False)
    print("Resultados salvos em backtest_results.csv")
    return df


if __name__ == "__main__":
    df = run_backtest()
    print(df[["patient_id", "true_label_name", "ensemble_pred_name",
              "ensemble_conf", "prioritized", "advance_minutes"]].head(20))