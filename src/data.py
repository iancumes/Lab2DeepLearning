"""Carga, exploracion y preparacion del dataset MNIST.

Laboratorio #2 - CC3092 Deep Learning y Sistemas Inteligentes
Ian Cumes - 23236
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Subset
from torchvision import transforms
from torchvision.datasets import MNIST

SEED = 23236
DATA_ROOT = Path(__file__).resolve().parent.parent / "data"

# El mirror por defecto de torchvision (yann.lecun.com) devuelve 403 desde
# muchos entornos; el mirror de OSSCI es el que se usa en produccion.
MNIST.mirrors = ["https://ossci-datasets.s3.amazonaws.com/mnist/"]

# Estadisticas clasicas de MNIST. En `compute_train_stats` se verifica que
# coinciden con las calculadas sobre nuestro split de entrenamiento.
MNIST_MEAN = 0.1307
MNIST_STD = 0.3081


def raw_datasets(root: Path | str = DATA_ROOT):
    """Devuelve MNIST train/test *sin normalizar* (solo ToTensor -> [0, 1]).

    Se usa para la exploracion de datos: permite inspeccionar el rango real
    de los pixeles antes de decidir la normalizacion.
    """
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    to_tensor = transforms.ToTensor()
    train = MNIST(root=str(root), train=True, download=True, transform=to_tensor)
    test = MNIST(root=str(root), train=False, download=True, transform=to_tensor)
    return train, test


def stratified_split(targets, val_size: int = 6000, seed: int = SEED):
    """Divide los indices de entrenamiento en train/val de forma estratificada.

    Estratificar mantiene la misma proporcion de cada digito en ambos
    subconjuntos, de modo que la metrica de validacion sea comparable entre
    iteraciones y no dependa de un sorteo afortunado.
    """
    targets = np.asarray(targets)
    idx = np.arange(len(targets))
    train_idx, val_idx = train_test_split(
        idx, test_size=val_size, random_state=seed, stratify=targets
    )
    return np.sort(train_idx), np.sort(val_idx)


def compute_train_stats(dataset, indices) -> tuple[float, float]:
    """Media y desviacion estandar de los pixeles calculadas SOLO sobre train.

    Calcularlas sobre el conjunto completo filtraria informacion de validacion
    y test hacia el preprocesamiento (data leakage).
    """
    data = dataset.data[indices].float().div_(255.0)
    return float(data.mean()), float(data.std())


def class_distribution(targets) -> dict[int, int]:
    """Conteo de observaciones por clase (para responder si esta balanceado)."""
    targets = np.asarray(targets)
    values, counts = np.unique(targets, return_counts=True)
    return {int(v): int(c) for v, c in zip(values, counts)}


def build_dataloaders(
    batch_size: int = 128,
    val_size: int = 6000,
    seed: int = SEED,
    root: Path | str = DATA_ROOT,
    num_workers: int = 2,
):
    """Construye los DataLoaders de train / validacion / test ya normalizados.

    Devuelve tambien el diccionario de metadatos usado en la seccion de
    exploracion del notebook.
    """
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)

    plain_train = MNIST(root=str(root), train=True, download=True)
    train_idx, val_idx = stratified_split(plain_train.targets.numpy(), val_size, seed)
    mean, std = compute_train_stats(plain_train, train_idx)

    tf = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((mean,), (std,))]
    )
    full_train = MNIST(root=str(root), train=True, download=True, transform=tf)
    test_set = MNIST(root=str(root), train=False, download=True, transform=tf)

    train_set = Subset(full_train, train_idx.tolist())
    val_set = Subset(full_train, val_idx.tolist())

    generator = torch.Generator().manual_seed(seed)
    common = dict(num_workers=num_workers, pin_memory=False, persistent_workers=num_workers > 0)
    loaders = {
        "train": DataLoader(
            train_set, batch_size=batch_size, shuffle=True, generator=generator, **common
        ),
        "val": DataLoader(val_set, batch_size=512, shuffle=False, **common),
        "test": DataLoader(test_set, batch_size=512, shuffle=False, **common),
    }
    meta = {
        "n_train": len(train_set),
        "n_val": len(val_set),
        "n_test": len(test_set),
        "mean": mean,
        "std": std,
        "image_shape": tuple(full_train.data.shape[1:]),
        "n_classes": int(len(full_train.classes)),
        "train_class_counts": class_distribution(plain_train.targets.numpy()[train_idx]),
        "val_class_counts": class_distribution(plain_train.targets.numpy()[val_idx]),
        "test_class_counts": class_distribution(test_set.targets.numpy()),
    }
    return loaders, meta
