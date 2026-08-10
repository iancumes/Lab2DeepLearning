"""Bucle de entrenamiento, evaluacion y metricas de clasificacion.

Laboratorio #2 - CC3092 Deep Learning y Sistemas Inteligentes
Ian Cumes - 23236
"""

from __future__ import annotations

import random
import time

import numpy as np
import torch
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
from torch import nn

from .models import build_model, count_parameters

SEED = 23236


def set_seed(seed: int = SEED) -> None:
    """Fija todas las fuentes de aleatoriedad.

    Es indispensable para que las iteraciones sean comparables: si dos configs
    parten de inicializaciones distintas no se puede atribuir la diferencia de
    metricas al hiperparametro que cambio.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def make_optimizer(model: nn.Module, cfg: dict) -> torch.optim.Optimizer:
    name = cfg.get("optimizer", "adam").lower()
    lr = cfg.get("lr", 1e-3)
    wd = cfg.get("weight_decay", 0.0)
    if name == "sgd":
        return torch.optim.SGD(model.parameters(), lr=lr, momentum=cfg.get("momentum", 0.0), weight_decay=wd)
    if name == "adam":
        return torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    raise ValueError(f"Optimizador desconocido: {name}")


@torch.no_grad()
def evaluate(model: nn.Module, loader, criterion, device: torch.device) -> dict:
    """Evalua el modelo y devuelve loss, metricas macro y las predicciones.

    Se usa tanto para validacion (en cada epoch) como para la evaluacion final
    sobre test, de modo que ambas se calculen exactamente igual.
    """
    model.eval()
    total_loss, n = 0.0, 0
    y_true, y_pred = [], []
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        logits = model(xb)
        total_loss += criterion(logits, yb).item() * yb.size(0)
        n += yb.size(0)
        y_true.append(yb.cpu().numpy())
        y_pred.append(logits.argmax(dim=1).cpu().numpy())

    y_true = np.concatenate(y_true)
    y_pred = np.concatenate(y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )
    return {
        "loss": total_loss / n,
        "accuracy": float((y_true == y_pred).mean()),
        "precision_macro": float(precision),
        "recall_macro": float(recall),
        "f1_macro": float(f1),
        "y_true": y_true,
        "y_pred": y_pred,
    }


def train_one(cfg: dict, loaders: dict, device: torch.device | None = None, verbose: bool = True) -> dict:
    """Entrena una configuracion y registra todo lo que exige el enunciado.

    Devuelve la configuracion usada, el historial de perdida por epoch
    (entrenamiento y validacion), las metricas finales de validacion, el
    numero de parametros entrenables y el tiempo de entrenamiento.
    """
    device = device or torch.device("cpu")
    set_seed(cfg.get("seed", SEED))

    model = build_model(cfg).to(device)
    criterion = nn.CrossEntropyLoss()  # espera logits crudos: aplica log_softmax internamente
    optimizer = make_optimizer(model, cfg)
    epochs = cfg.get("epochs", 10)

    history = {"train_loss": [], "val_loss": [], "val_accuracy": []}
    start = time.perf_counter()

    for epoch in range(1, epochs + 1):
        model.train()
        running, seen = 0.0, 0
        for xb, yb in loaders["train"]:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()
            running += loss.item() * yb.size(0)
            seen += yb.size(0)

        train_loss = running / seen
        val = evaluate(model, loaders["val"], criterion, device)
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val["loss"])
        history["val_accuracy"].append(val["accuracy"])
        if verbose:
            print(
                f"  [{cfg['id']}] epoch {epoch:2d}/{epochs} "
                f"train_loss={train_loss:.4f} val_loss={val['loss']:.4f} val_acc={val['accuracy']:.4f}",
                flush=True,
            )

    elapsed = time.perf_counter() - start
    final_val = evaluate(model, loaders["val"], criterion, device)
    return {
        "id": cfg["id"],
        "arch": cfg["arch"],
        "change": cfg.get("change", ""),
        "config": {k: v for k, v in cfg.items() if k not in {"id", "change"}},
        "history": history,
        "val_metrics": {k: v for k, v in final_val.items() if k not in {"y_true", "y_pred"}},
        "n_params": count_parameters(model),
        "train_time_s": elapsed,
        "model": model,
    }


@torch.no_grad()
def measure_inference_time(model: nn.Module, loader, device: torch.device, n_batches: int = 10) -> float:
    """Tiempo medio de inferencia por imagen, en milisegundos.

    Necesario para la pregunta de despliegue en produccion de la seccion 6.
    """
    model.eval()
    total_time, total_images = 0.0, 0
    for i, (xb, _) in enumerate(loader):
        if i >= n_batches:
            break
        xb = xb.to(device)
        t0 = time.perf_counter()
        model(xb)
        total_time += time.perf_counter() - t0
        total_images += xb.size(0)
    return (total_time / total_images) * 1000.0


def final_test_evaluation(cfg: dict, loaders: dict, device: torch.device | None = None) -> dict:
    """Re-entrena la mejor configuracion y la evalua UNA SOLA VEZ sobre test."""
    device = device or torch.device("cpu")
    result = train_one(cfg, loaders, device=device, verbose=True)
    model = result.pop("model")
    criterion = nn.CrossEntropyLoss()
    test = evaluate(model, loaders["test"], criterion, device)
    cm = confusion_matrix(test["y_true"], test["y_pred"], labels=list(range(10)))
    result["test_metrics"] = {k: v for k, v in test.items() if k not in {"y_true", "y_pred"}}
    result["confusion_matrix"] = cm.tolist()
    result["inference_ms_per_image"] = measure_inference_time(model, loaders["test"], device)
    return result
