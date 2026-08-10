"""Arquitecturas MLP y CNN para la clasificacion de digitos de MNIST.

Ambas clases son parametrizables por un diccionario de configuracion para
poder recorrer el espacio de hiperparametros cambiando una variable a la vez.

Laboratorio #2 - CC3092 Deep Learning y Sistemas Inteligentes
Ian Cumes - 23236
"""

from __future__ import annotations

import torch
from torch import nn


def count_parameters(model: nn.Module) -> int:
    """Numero total de parametros entrenables del modelo."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


class MLP(nn.Module):
    """Perceptron multicapa: recibe la imagen APLANADA como vector de 784.

    Al aplanar, la estructura espacial 28x28 se pierde: cada pixel pasa a ser
    una feature independiente y cada neurona oculta necesita un peso por pixel.
    """

    def __init__(
        self,
        hidden_sizes: tuple[int, ...] = (128,),
        dropout: float = 0.0,
        use_bn: bool = False,
        in_features: int = 28 * 28,
        n_classes: int = 10,
    ):
        super().__init__()
        layers: list[nn.Module] = [nn.Flatten()]
        prev = in_features
        for h in hidden_sizes:
            layers.append(nn.Linear(prev, h))
            if use_bn:
                layers.append(nn.BatchNorm1d(h))
            layers.append(nn.ReLU(inplace=True))
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            prev = h
        layers.append(nn.Linear(prev, n_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class CNN(nn.Module):
    """Red convolucional: recibe la imagen como tensor 2D (N, 1, 28, 28).

    Cada bloque es Conv2d -> [BatchNorm2d] -> ReLU -> Pool(2). Con `channels`
    de largo >= 2 se cumple el requisito de al menos dos capas convolucionales.
    El pooling es intercambiable (`max` / `avg`) para poder comparar
    empiricamente nn.MaxPool2d contra nn.AvgPool2d.
    """

    def __init__(
        self,
        channels: tuple[int, ...] = (16, 32),
        kernel_size: int = 3,
        pool: str = "max",
        use_bn: bool = False,
        dropout: float = 0.0,
        fc_hidden: int = 128,
        in_channels: int = 1,
        image_size: int = 28,
        n_classes: int = 10,
    ):
        super().__init__()
        if len(channels) < 2:
            raise ValueError("La CNN debe tener al menos dos capas convolucionales")
        if pool not in {"max", "avg"}:
            raise ValueError("pool debe ser 'max' o 'avg'")

        pool_cls = nn.MaxPool2d if pool == "max" else nn.AvgPool2d
        padding = kernel_size // 2  # 'same': conserva el tamano espacial

        blocks: list[nn.Module] = []
        prev = in_channels
        size = image_size
        for c in channels:
            blocks.append(nn.Conv2d(prev, c, kernel_size=kernel_size, padding=padding))
            if use_bn:
                blocks.append(nn.BatchNorm2d(c))
            blocks.append(nn.ReLU(inplace=True))
            blocks.append(pool_cls(kernel_size=2))
            prev = c
            size //= 2  # cada pooling 2x2 reduce a la mitad alto y ancho
        self.features = nn.Sequential(*blocks)

        head: list[nn.Module] = [nn.Flatten(), nn.Linear(prev * size * size, fc_hidden), nn.ReLU(inplace=True)]
        if dropout > 0:
            head.append(nn.Dropout(dropout))
        head.append(nn.Linear(fc_hidden, n_classes))
        self.classifier = nn.Sequential(*head)

        self.channels = channels
        self.kernel_size = kernel_size
        self.final_spatial = size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(x))

    def receptive_field(self) -> int:
        """Campo receptivo (en pixeles de la entrada) de la ultima capa conv.

        Formula incremental: RF_out = RF_in + (k - 1) * jump, donde `jump` es
        el producto de los strides acumulados. Cada Conv2d aporta (k-1)*jump y
        cada Pool2d(2) aporta (2-1)*jump y luego duplica el jump.
        """
        rf, jump = 1, 1
        for _ in self.channels:
            rf += (self.kernel_size - 1) * jump  # conv stride 1
            rf += (2 - 1) * jump                 # pool 2x2
            jump *= 2                            # pool stride 2
        return rf


def build_model(cfg: dict) -> nn.Module:
    """Instancia el modelo descrito por una configuracion de iteracion."""
    arch = cfg["arch"]
    if arch == "MLP":
        return MLP(
            hidden_sizes=tuple(cfg["hidden_sizes"]),
            dropout=cfg.get("dropout", 0.0),
            use_bn=cfg.get("use_bn", False),
        )
    if arch == "CNN":
        return CNN(
            channels=tuple(cfg["channels"]),
            kernel_size=cfg.get("kernel_size", 3),
            pool=cfg.get("pool", "max"),
            use_bn=cfg.get("use_bn", False),
            dropout=cfg.get("dropout", 0.0),
            fc_hidden=cfg.get("fc_hidden", 128),
        )
    raise ValueError(f"Arquitectura desconocida: {arch}")
