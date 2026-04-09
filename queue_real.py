"""
Simulador de Retorno Pós-Exame — Fila de Especialista
Fluxo:
  Dia 0  → Consulta com especialista
  Dia 2  → Paciente realiza exame topográfico
  Dia 2  → Exame entra automaticamente no sistema (inferência)
  Dia 16 → Retorno padrão (2 semanas após exame)
  OU
  Dia 2+L → Ligação realizada (L = 0–2 dias após exame entrar no sistema)
  Dia 2+L+R → Retorno agendado (R = 2–4 dias após ligação)
  Condição: ensemble_conf ≥ 85% E classe predita = keratoconus
"""

import os
import time
import numpy as np
import pandas as pd
from dataclasses import dataclass
from PIL import Image
import torch

# ── Configurações ──────────────────────────────────────────────────────────────
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_SCRIPT_DIR)
TEST_DIR          = os.path.join(_PROJECT_ROOT, "TCC-Modelos", "test")
KERAS_MODEL_PATH  = os.path.join(_PROJECT_ROOT, "TCC-Modelos", "modeloFinalNaoAguentoMaisKeras.h5")
YOLO_MODEL_PATH   = os.path.join(_PROJECT_ROOT, "TCC-Modelos", "Modelo_Yolov11_Improve_Final.pt")
IMG_SIZE             = (224, 224)

CONFIDENCE_THRESHOLD  = 0.75
KERATOCONUS_CLASS_IDX = 0        # sorted(['keratoconus', 'normal'])[0]

EXAM_DELAY_DAYS       = 2        # dias entre consulta e exame
STANDARD_RETURN_DAYS  = 16       # dias até retorno padrão (exame + 2 semanas)
CALL_WINDOW_DAYS      = (0, 2)   # dias após entrada do exame no sistema para ligação
RETURN_WINDOW_DAYS    = (2, 4)   # dias após ligação para agendamento do retorno

KERAS_WEIGHT          = 0.45
YOLO_WEIGHT           = 0.55

PATIENTS_PER_DAY      = 12       # pacientes que fazem exame por dia (Poisson)


@dataclass
class ExamPatient:
    patient_id:           int
    image_path:           str
    true_label:           int
    true_label_name:      str

    # ── Datas (em dias desde dia 0 = consulta) ─────────────────────────────────
    consultation_day:     float = 0.0   # dia 0 por definição
    exam_day:             float = 0.0   # consulta + EXAM_DELAY_DAYS + jitter
    system_entry_day:     float = 0.0   # mesmo que exam_day (entrada automática)
    standard_return_day:  float = 0.0   # exam_day + 14 dias
    call_day:             float = 0.0   # só se priorizado
    scheduled_return_day: float = 0.0   # retorno real após priorização ou padrão

    # ── Predições ──────────────────────────────────────────────────────────────
    keras_pred:           int   = -1
    keras_conf:           float = 0.0
    yolo_pred:            int   = -1
    yolo_conf:            float = 0.0
    ensemble_pred:        int   = -1
    ensemble_conf:        float = 0.0
    correct:              bool  = False

    # ── Priorização ────────────────────────────────────────────────────────────
    called:               bool  = False   # recebeu ligação
    days_saved:           float = 0.0     # dias ganhos vs retorno padrão
    inference_time_ms:    float = 0.0


def load_models():
    from tensorflow.keras.models import load_model
    from ultralytics import YOLO as UltralyticsYOLO
    print("Carregando modelos...")
    keras_model = load_model(KERAS_MODEL_PATH)
    yolo_model  = UltralyticsYOLO(YOLO_MODEL_PATH)
    print("Modelos carregados.")
    return keras_model, yolo_model


def preprocess_keras(path: str) -> np.ndarray:
    from tensorflow.keras.preprocessing import image as kimage
    img = kimage.load_img(path, target_size=IMG_SIZE)
    arr = kimage.img_to_array(img) / 255.0
    return np.expand_dims(arr, axis=0)


def predict_patient(path: str, keras_model, yolo_model) -> dict:
    t0 = time.perf_counter()

    keras_input = preprocess_keras(path)
    keras_raw   = float(keras_model.predict(keras_input, verbose=0)[0][0])
    keras_pred  = 1 if keras_raw > 0.5 else 0
    keras_conf  = keras_raw if keras_pred == 1 else (1.0 - keras_raw)

    img_pil   = Image.open(path).convert("RGB").resize(IMG_SIZE)
    result    = yolo_model(img_pil, imgsz=224, verbose=False)[0]
    probs     = result.probs.data.cpu().numpy()
    yolo_pred = int(np.argmax(probs))
    yolo_conf = float(probs[yolo_pred])

    p_normal_ens = KERAS_WEIGHT * keras_raw + YOLO_WEIGHT * float(probs[1])
    ens_pred     = 1 if p_normal_ens > 0.5 else 0
    ens_conf     = p_normal_ens if ens_pred == 1 else (1.0 - p_normal_ens)

    return {
        "keras_pred": keras_pred, "keras_conf": keras_conf,
        "yolo_pred":  yolo_pred,  "yolo_conf":  yolo_conf,
        "ensemble_pred": ens_pred, "ensemble_conf": ens_conf,
        "inference_time_ms": (time.perf_counter() - t0) * 1000,
    }


def simulate_exam_schedule(n_patients: int) -> np.ndarray:
    """
    Gera o dia de consulta de cada paciente.
    Distribuição Poisson: PATIENTS_PER_DAY pacientes/dia em média.
    Retorna array de dias desde abertura do sistema.
    """
    inter = np.random.exponential(1.0 / PATIENTS_PER_DAY, size=n_patients)
    return np.cumsum(inter)


def apply_return_logic(patients: list[ExamPatient]) -> list[ExamPatient]:
    """
    Para cada paciente:
    - Define exam_day = consultation_day + EXAM_DELAY_DAYS + pequeno jitter
    - Define standard_return_day = exam_day + 14
    - Se conf ≥ 85% e pred = keratoconus:
        call_day = system_entry_day + U[0, 2]
        scheduled_return_day = call_day + U[2, 4]
      Senão:
        scheduled_return_day = standard_return_day
    """
    for p in patients:
        jitter         = np.random.uniform(0, 0.5)   # até meio dia de variação
        p.exam_day     = p.consultation_day + EXAM_DELAY_DAYS + jitter
        p.system_entry_day   = p.exam_day
        p.standard_return_day = p.exam_day + 14.0

        if (p.ensemble_conf >= CONFIDENCE_THRESHOLD
                and p.ensemble_pred == KERATOCONUS_CLASS_IDX):
            call_delay           = np.random.uniform(*CALL_WINDOW_DAYS)
            return_delay         = np.random.uniform(*RETURN_WINDOW_DAYS)
            p.call_day           = p.system_entry_day + call_delay
            p.scheduled_return_day = p.call_day + return_delay
            p.called             = True
            p.days_saved         = p.standard_return_day - p.scheduled_return_day
        else:
            p.scheduled_return_day = p.standard_return_day
            p.days_saved           = 0.0

    return patients


def format_duration(days: float) -> str:
    """Converte dias para formato 'Xd Xh Xm'"""
    total_minutes = int(days * 24 * 60)
    days_part = total_minutes // (24 * 60)
    hours_part = (total_minutes % (24 * 60)) // 60
    mins_part = total_minutes % 60
    return f"{days_part}d {hours_part}h {mins_part}m"


def run_backtest(keras_model=None, yolo_model=None) -> pd.DataFrame:
    if keras_model is None or yolo_model is None:
        keras_model, yolo_model = load_models()

    class_names = sorted(os.listdir(TEST_DIR))
    label_map   = {name: idx for idx, name in enumerate(class_names)}
    print(f"Classes: {label_map}")

    all_paths, all_labels = [], []
    for cls in class_names:
        cls_dir = os.path.join(TEST_DIR, cls)
        if not os.path.isdir(cls_dir):
            continue
        for fname in os.listdir(cls_dir):
            if fname.lower().endswith((".jpg", ".jpeg", ".png")):
                all_paths.append(os.path.join(cls_dir, fname))
                all_labels.append(label_map[cls])

    n   = len(all_paths)
    idx = np.random.permutation(n)
    all_paths  = [all_paths[i]  for i in idx]
    all_labels = [all_labels[i] for i in idx]

    consultation_days = simulate_exam_schedule(n)

    patients: list[ExamPatient] = []
    for i, (path, label, c_day) in enumerate(zip(all_paths, all_labels, consultation_days)):
        print(f"  [{i+1:03d}/{n}] {os.path.basename(path)}", end="\r")
        preds = predict_patient(path, keras_model, yolo_model)
        p = ExamPatient(
            patient_id       = i + 1,
            image_path       = path,
            true_label       = label,
            true_label_name  = class_names[label],
            consultation_day = c_day,
            **preds,
        )
        p.correct = (p.ensemble_pred == p.true_label)
        patients.append(p)

    print(f"\nInferência concluída.")
    patients.sort(key=lambda p: p.consultation_day)
    patients = apply_return_logic(patients)

    rows = []
    for p in patients:
        rows.append({
            "patient_id":             p.patient_id,
            "image_path":             p.image_path,
            "true_label":             p.true_label,
            "true_label_name":        p.true_label_name,
            "consultation_day":       round(p.consultation_day, 2),
            "exam_day":               round(p.exam_day, 2),
            "system_entry_day":       round(p.system_entry_day, 2),
            "standard_return_day":    round(p.standard_return_day, 2),
            "call_day":               round(p.call_day, 2) if p.called else None,
            "scheduled_return_day":   round(p.scheduled_return_day, 2),
            "days_saved":             round(p.days_saved, 2),
            "arrival_time_min":       format_duration(p.consultation_day),
            "scheduled_time_min":     format_duration(p.scheduled_return_day),
            "advance_minutes":        format_duration(p.days_saved),
            "prioritized":            p.called,  # called = prioritized
            "keras_pred":             p.keras_pred,
            "keras_conf":             p.keras_conf,
            "yolo_pred":              p.yolo_pred,
            "yolo_conf":              p.yolo_conf,
            "ensemble_pred":          p.ensemble_pred,
            "ensemble_conf":          p.ensemble_conf,
            "ensemble_pred_name":     class_names[p.ensemble_pred],
            "correct":                p.correct,
            "inference_time_ms":      p.inference_time_ms,
        })

    df = pd.DataFrame(rows)
    df.to_csv("backtest_results1.csv", index=False)
    print("Resultados salvos em exam_backtest_results1.csv")
    return df


if __name__ == "__main__":
    df = run_backtest()

    called     = df[df["called"]]
    not_called = df[~df["called"]]

    print(f"\nTotal de pacientes: {len(df)}")
    print(f"Ligações realizadas: {len(called)} ({len(called)/len(df):.1%})")
    print(f"Dias salvos (média): {called['days_saved'].mean():.1f} dias")
    print(f"\nRetorno médio — priorizados:  dia {called['scheduled_return_day'].mean():.1f}")
    print(f"Retorno médio — padrão:       dia {not_called['standard_return_day'].mean():.1f}")