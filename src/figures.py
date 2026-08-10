"""Generacion de todas las figuras del laboratorio.

Cada funcion guarda un PNG en results/figures/ y devuelve la ruta, de modo que
el notebook y el generador del PDF consuman exactamente las mismas imagenes.

Laboratorio #2 - CC3092 Deep Learning y Sistemas Inteligentes
Ian Cumes - 23236
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

FIG_DIR = Path(__file__).resolve().parent.parent / "results" / "figures"
DPI = 150

MLP_COLOR = "#1f77b4"
CNN_COLOR = "#d62728"


def _save(fig, name: str) -> Path:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    path = FIG_DIR / name
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_samples(dataset, n: int = 12, name: str = "samples.png") -> Path:
    """Visualiza n ejemplos del dataset con su etiqueta (requisito de la seccion 2)."""
    cols = 6
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 1.3, rows * 1.5))
    for i, ax in enumerate(np.array(axes).ravel()):
        if i < n:
            img, label = dataset[i]
            ax.imshow(np.asarray(img).squeeze(), cmap="gray")
            ax.set_title(f"y = {label}", fontsize=9)
        ax.axis("off")
    fig.suptitle("Ejemplos de MNIST con su etiqueta", fontsize=11)
    return _save(fig, name)


def plot_class_distribution(counts: dict[int, int], title: str, name: str = "class_distribution.png") -> Path:
    """Distribucion de observaciones por clase: sirve para juzgar el balance."""
    keys = sorted(counts)
    values = [counts[k] for k in keys]
    mean = np.mean(values)
    fig, ax = plt.subplots(figsize=(6, 2.8))
    ax.bar([str(k) for k in keys], values, color=MLP_COLOR)
    ax.axhline(mean, color="gray", linestyle="--", linewidth=1, label=f"media = {mean:.0f}")
    ax.set_xlabel("Digito")
    ax.set_ylabel("Observaciones")
    ax.set_title(title, fontsize=11)
    ax.legend(fontsize=8)
    return _save(fig, name)


def plot_loss_curves(iterations: list[dict], arch: str, name: str | None = None) -> Path:
    """Curvas de perdida de train y validacion de todas las iteraciones de una arquitectura.

    Linea continua = entrenamiento, punteada = validacion. La brecha entre
    ambas es la senal visual de overfitting.
    """
    rows = [r for r in iterations if r["arch"] == arch]
    name = name or f"loss_curves_{arch.lower()}.png"
    cmap = plt.get_cmap("tab10")

    fig, ax = plt.subplots(figsize=(6.2, 3.4))
    for i, r in enumerate(rows):
        epochs = range(1, len(r["history"]["train_loss"]) + 1)
        color = cmap(i % 10)
        ax.plot(epochs, r["history"]["train_loss"], color=color, linewidth=1.4, label=f"{r['id']} train")
        ax.plot(epochs, r["history"]["val_loss"], color=color, linewidth=1.4, linestyle="--", label=f"{r['id']} val")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Cross-entropy loss")
    ax.set_title(f"Curvas de perdida - {arch} ({len(rows)} iteraciones)", fontsize=11)
    ax.set_yscale("log")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=6, ncol=2, loc="upper right")
    return _save(fig, name)


def plot_confusion_matrix(cm, title: str, name: str) -> Path:
    """Matriz de confusion 10x10 con los conteos anotados."""
    cm = np.asarray(cm)
    fig, ax = plt.subplots(figsize=(4.4, 4.0))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(10))
    ax.set_yticks(range(10))
    ax.set_xlabel("Prediccion")
    ax.set_ylabel("Etiqueta real")
    ax.set_title(title, fontsize=11)
    threshold = cm.max() / 2
    for i in range(10):
        for j in range(10):
            if cm[i, j] == 0:
                continue
            ax.text(
                j, i, str(cm[i, j]), ha="center", va="center", fontsize=6,
                color="white" if cm[i, j] > threshold else "black",
            )
    fig.colorbar(im, ax=ax, fraction=0.046)
    return _save(fig, name)


def plot_params_vs_accuracy(final_test: dict, name: str = "params_vs_accuracy.png") -> Path:
    """Relaciona el numero de parametros con el accuracy en test (seccion 5)."""
    fig, ax = plt.subplots(figsize=(5.2, 3.2))
    for arch, color in (("MLP", MLP_COLOR), ("CNN", CNN_COLOR)):
        r = final_test[arch]
        x, y = r["n_params"], r["test_metrics"]["accuracy"] * 100
        ax.scatter(x, y, s=140, color=color, zorder=3, label=f"{arch} ({r['best_iteration_id']})")
        ax.annotate(
            f"{x:,} params\n{y:.2f}%", (x, y), textcoords="offset points",
            xytext=(0, -32), ha="center", fontsize=8,
        )
    ax.set_xscale("log")
    ax.set_xlabel("Parametros entrenables (escala log)")
    ax.set_ylabel("Accuracy en test (%)")
    ax.set_title("Parametros vs. accuracy en test", fontsize=11)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, loc="lower left")
    return _save(fig, name)


def plot_val_metric_by_iteration(iterations: list[dict], name: str = "val_metrics_by_iteration.png") -> Path:
    """Accuracy de validacion por iteracion: hace visible el efecto de cada cambio."""
    fig, axes = plt.subplots(1, 2, figsize=(7.6, 2.9), sharey=False)
    for ax, (arch, color) in zip(axes, (("MLP", MLP_COLOR), ("CNN", CNN_COLOR))):
        rows = [r for r in iterations if r["arch"] == arch]
        ids = [r["id"] for r in rows]
        accs = [r["val_metrics"]["accuracy"] * 100 for r in rows]
        bars = ax.bar(ids, accs, color=color, alpha=0.85)
        best = int(np.argmax(accs))
        bars[best].set_edgecolor("black")
        bars[best].set_linewidth(1.8)
        ax.set_ylim(min(accs) - 0.6, max(accs) + 0.35)
        ax.set_title(f"{arch}: accuracy de validacion", fontsize=10)
        ax.set_ylabel("Accuracy (%)")
        ax.grid(axis="y", alpha=0.3)
        for b, a in zip(bars, accs):
            ax.text(b.get_x() + b.get_width() / 2, a, f"{a:.2f}", ha="center", va="bottom", fontsize=7)
    fig.tight_layout()
    return _save(fig, name)
