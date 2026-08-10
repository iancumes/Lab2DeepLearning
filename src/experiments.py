"""Busqueda sistematica de hiperparametros: 12 iteraciones (6 MLP + 6 CNN).

Estrategia: se parte de una configuracion baseline por arquitectura y se cambia
UNA sola variable a la vez, de modo que cualquier diferencia en las metricas de
validacion sea atribuible a ese cambio concreto.

Los resultados se escriben a results/iterations.json apenas termina cada
iteracion, asi la busqueda es reanudable si se interrumpe.

Laboratorio #2 - CC3092 Deep Learning y Sistemas Inteligentes
Ian Cumes - 23236
"""

from __future__ import annotations

import json
from pathlib import Path

import torch

from .data import build_dataloaders
from .train import SEED, final_test_evaluation

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"
ITERATIONS_PATH = RESULTS_DIR / "iterations.json"
FINAL_TEST_PATH = RESULTS_DIR / "final_test.json"

MLP_EPOCHS = 10
CNN_EPOCHS = 8


def mlp_configs() -> list[dict]:
    """6 iteraciones del MLP; cada una cambia una variable respecto de la previa."""
    base = dict(
        arch="MLP",
        hidden_sizes=(128,),
        dropout=0.0,
        use_bn=False,
        optimizer="sgd",
        lr=0.01,
        batch_size=128,
        epochs=MLP_EPOCHS,
        seed=SEED,
    )
    cfgs = []

    cfgs.append({**base, "id": "M1", "change": "Baseline: 1 capa oculta de 128, SGD lr=0.01"})
    cfgs.append({**base, "id": "M2", "change": "Optimizador SGD -> Adam (lr=1e-3)", "optimizer": "adam", "lr": 1e-3})

    m3 = {**cfgs[-1], "id": "M3", "change": "Ancho de la capa oculta 128 -> 256", "hidden_sizes": (256,)}
    cfgs.append(m3)

    m4 = {**m3, "id": "M4", "change": "Profundidad: 1 -> 2 capas ocultas (256, 128)", "hidden_sizes": (256, 128)}
    cfgs.append(m4)

    m5 = {**m4, "id": "M5", "change": "Regularizacion: + Dropout 0.3", "dropout": 0.3}
    cfgs.append(m5)

    m6 = {**m5, "id": "M6", "change": "Normalizacion: + BatchNorm1d", "use_bn": True}
    cfgs.append(m6)

    return cfgs


def cnn_configs() -> list[dict]:
    """6 iteraciones de la CNN; misma logica de una variable a la vez."""
    base = dict(
        arch="CNN",
        channels=(16, 32),
        kernel_size=3,
        pool="max",
        use_bn=False,
        dropout=0.0,
        fc_hidden=128,
        optimizer="adam",
        lr=1e-3,
        batch_size=128,
        epochs=CNN_EPOCHS,
        seed=SEED,
    )
    cfgs = []

    cfgs.append({**base, "id": "C1", "change": "Baseline: 2 bloques conv (16, 32), MaxPool, Adam lr=1e-3"})

    c2 = {**base, "id": "C2", "change": "Capacidad: canales (16,32) -> (32,64)", "channels": (32, 64)}
    cfgs.append(c2)

    c3 = {**c2, "id": "C3", "change": "Pooling: MaxPool2d -> AvgPool2d", "pool": "avg"}
    cfgs.append(c3)

    # C4 se construye sobre el mejor pooling encontrado; se resuelve en runtime.
    c4 = {**c2, "id": "C4", "change": "Normalizacion: + BatchNorm2d", "use_bn": True}
    cfgs.append(c4)

    c5 = {**c4, "id": "C5", "change": "Regularizacion: + Dropout 0.25 antes de la FC final", "dropout": 0.25}
    cfgs.append(c5)

    c6 = {**c5, "id": "C6", "change": "Profundidad: 2 -> 3 bloques conv (32, 64, 128)", "channels": (32, 64, 128)}
    cfgs.append(c6)

    return cfgs


def all_configs() -> list[dict]:
    return mlp_configs() + cnn_configs()


def _serializable(cfg: dict) -> dict:
    return {k: (list(v) if isinstance(v, tuple) else v) for k, v in cfg.items()}


def load_iterations() -> list[dict]:
    if ITERATIONS_PATH.exists():
        return json.loads(ITERATIONS_PATH.read_text())
    return []


def run_search(device: torch.device | None = None, force: bool = False) -> list[dict]:
    """Ejecuta las 12 iteraciones, saltando las que ya esten en el JSON."""
    from .train import train_one  # import diferido: torch tarda en cargar

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

    done = [] if force else load_iterations()
    done_ids = {r["id"] for r in done}

    configs = all_configs()
    # Los DataLoaders solo dependen del batch_size, que es constante en esta
    # busqueda; se construyen una sola vez para no releer MNIST 12 veces.
    loaders, meta = build_dataloaders(batch_size=configs[0]["batch_size"])
    print(f"Datos: train={meta['n_train']} val={meta['n_val']} test={meta['n_test']}", flush=True)

    for cfg in configs:
        if cfg["id"] in done_ids:
            print(f"[{cfg['id']}] ya calculada, se omite", flush=True)
            continue
        print(f"\n=== {cfg['id']} ({cfg['arch']}): {cfg['change']} ===", flush=True)
        res = train_one(_serializable(cfg), loaders, device=device)
        res.pop("model", None)
        done.append(res)
        done.sort(key=lambda r: (r["arch"] != "MLP", r["id"]))
        ITERATIONS_PATH.write_text(json.dumps(done, indent=2))
        print(
            f"[{cfg['id']}] val_acc={res['val_metrics']['accuracy']:.4f} "
            f"f1={res['val_metrics']['f1_macro']:.4f} params={res['n_params']:,} "
            f"t={res['train_time_s']:.1f}s",
            flush=True,
        )
    return done


def best_config_per_arch(iterations: list[dict]) -> dict[str, dict]:
    """Mejor iteracion por arquitectura segun F1-macro de validacion.

    Desempate por accuracy de validacion. La seleccion se hace exclusivamente
    con validacion; test se toca una sola vez, despues.
    """
    best = {}
    for arch in ("MLP", "CNN"):
        candidates = [r for r in iterations if r["arch"] == arch]
        best[arch] = max(
            candidates,
            key=lambda r: (r["val_metrics"]["f1_macro"], r["val_metrics"]["accuracy"]),
        )
    return best


def run_final_test(device: torch.device | None = None, force: bool = False) -> dict:
    """Re-entrena la mejor config de cada arquitectura y evalua UNA VEZ en test."""
    if FINAL_TEST_PATH.exists() and not force:
        return json.loads(FINAL_TEST_PATH.read_text())

    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    iterations = load_iterations()
    if len(iterations) < 12:
        raise RuntimeError("Faltan iteraciones; ejecute run_search() primero")

    best = best_config_per_arch(iterations)
    loaders, _ = build_dataloaders(batch_size=128)

    out = {}
    for arch, row in best.items():
        cfg = dict(row["config"])
        cfg["id"] = row["id"]
        cfg["change"] = row["change"]
        print(f"\n=== Evaluacion final en test: {arch} (mejor = {row['id']}) ===", flush=True)
        res = final_test_evaluation(cfg, loaders, device=device)
        res["best_iteration_id"] = row["id"]
        out[arch] = res
        print(
            f"[{arch}] test_acc={res['test_metrics']['accuracy']:.4f} "
            f"f1={res['test_metrics']['f1_macro']:.4f} params={res['n_params']:,}",
            flush=True,
        )

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    FINAL_TEST_PATH.write_text(json.dumps(out, indent=2))
    return out


if __name__ == "__main__":
    torch.set_num_threads(4)
    run_search()
    run_final_test()
