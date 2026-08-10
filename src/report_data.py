"""Consolida results/*.json en las tablas que consumen el notebook y el PDF.

Tener una sola fuente de verdad evita transcribir numeros a mano al reporte.

Laboratorio #2 - CC3092 Deep Learning y Sistemas Inteligentes
Ian Cumes - 23236
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"
ITERATIONS_PATH = RESULTS_DIR / "iterations.json"
FINAL_TEST_PATH = RESULTS_DIR / "final_test.json"


def load_results() -> tuple[list[dict], dict]:
    iterations = json.loads(ITERATIONS_PATH.read_text())
    final_test = json.loads(FINAL_TEST_PATH.read_text()) if FINAL_TEST_PATH.exists() else {}
    return iterations, final_test


def describe_config(cfg: dict) -> str:
    """Resumen legible de una configuracion, para la columna de la tabla."""
    if cfg["arch"] == "MLP":
        parts = ["-".join(str(h) for h in cfg["hidden_sizes"])]
    else:
        parts = ["-".join(str(c) for c in cfg["channels"]), f"pool={cfg['pool']}", f"fc={cfg['fc_hidden']}"]
    parts.append(f"{cfg['optimizer']} lr={cfg['lr']:g}")
    if cfg.get("use_bn"):
        parts.append("BN")
    if cfg.get("dropout", 0) > 0:
        parts.append(f"drop={cfg['dropout']:g}")
    return ", ".join(parts)


def iterations_table(iterations: list[dict]) -> pd.DataFrame:
    """Tabla de las 12 iteraciones con todo lo que exige el enunciado."""
    rows = []
    for r in iterations:
        cfg = r["config"]
        rows.append(
            {
                "ID": r["id"],
                "Arq.": r["arch"],
                "Cambio respecto a la anterior": r["change"],
                "Configuracion": describe_config(cfg),
                "Epochs": cfg["epochs"],
                "Train loss": r["history"]["train_loss"][-1],
                "Val loss": r["history"]["val_loss"][-1],
                "Accuracy": r["val_metrics"]["accuracy"],
                "Precision": r["val_metrics"]["precision_macro"],
                "Recall": r["val_metrics"]["recall_macro"],
                "F1": r["val_metrics"]["f1_macro"],
                "Params": r["n_params"],
                "Tiempo (s)": r["train_time_s"],
            }
        )
    return pd.DataFrame(rows)


def comparison_table(final_test: dict) -> pd.DataFrame:
    """Comparacion directa MLP vs CNN sobre el conjunto de test (seccion 5)."""
    rows = []
    for arch in ("MLP", "CNN"):
        r = final_test[arch]
        m = r["test_metrics"]
        rows.append(
            {
                "Arquitectura": arch,
                "Mejor iteracion": r["best_iteration_id"],
                "Parametros entrenables": r["n_params"],
                "Accuracy (test)": m["accuracy"],
                "Precision (test)": m["precision_macro"],
                "Recall (test)": m["recall_macro"],
                "F1 (test)": m["f1_macro"],
                "Tiempo entren. (s)": r["train_time_s"],
                "Inferencia (ms/img)": r["inference_ms_per_image"],
            }
        )
    return pd.DataFrame(rows)


def top_confusions(cm, k: int = 5) -> list[tuple[int, int, int]]:
    """Los k pares (real, predicho) mas confundidos, ignorando la diagonal."""
    pairs = [
        (i, j, int(cm[i][j]))
        for i in range(len(cm))
        for j in range(len(cm))
        if i != j and cm[i][j] > 0
    ]
    pairs.sort(key=lambda t: t[2], reverse=True)
    return pairs[:k]


def hyperparameter_impact(iterations: list[dict], arch: str) -> pd.DataFrame:
    """Delta de F1-macro de validacion que produjo cada cambio de hiperparametro.

    Responde directamente la primera pregunta de la seccion 6: que cambio tuvo
    el mayor impacto positivo y cual el mayor impacto negativo.
    """
    rows = [r for r in iterations if r["arch"] == arch]
    out = []
    for prev, cur in zip(rows, rows[1:]):
        out.append(
            {
                "Iteracion": cur["id"],
                "Cambio": cur["change"],
                "F1 previo": prev["val_metrics"]["f1_macro"],
                "F1 nuevo": cur["val_metrics"]["f1_macro"],
                "Delta F1": cur["val_metrics"]["f1_macro"] - prev["val_metrics"]["f1_macro"],
            }
        )
    return pd.DataFrame(out).sort_values("Delta F1", ascending=False).reset_index(drop=True)


def overfitting_gap(iterations: list[dict]) -> pd.DataFrame:
    """Brecha val_loss - train_loss al final del entrenamiento.

    Una brecha grande y creciente es la firma cuantitativa del overfitting que
    se observa en las curvas de perdida.
    """
    rows = []
    for r in iterations:
        tr = r["history"]["train_loss"][-1]
        va = r["history"]["val_loss"][-1]
        rows.append(
            {
                "ID": r["id"],
                "Arq.": r["arch"],
                "Train loss": tr,
                "Val loss": va,
                "Brecha (val - train)": va - tr,
            }
        )
    return pd.DataFrame(rows).sort_values("Brecha (val - train)", ascending=False).reset_index(drop=True)
