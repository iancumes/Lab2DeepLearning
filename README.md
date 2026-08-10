# Laboratorio #2 — Redes Neuronales Convolucionales (MNIST)

**Curso:** CC3092 — Deep Learning y Sistemas Inteligentes
**Estudiante:** Ian Cumes · **Carné:** 23236

Comparación entre un **MLP** y una **CNN** para la clasificación de los dígitos de MNIST, con una
búsqueda sistemática de hiperparámetros de **12 iteraciones** (6 por arquitectura) y evaluación final
sobre el conjunto de test.

## Entregables

| Entregable | Ruta |
|---|---|
| Reporte PDF (máx. 3 páginas) | [`report/Laboratorio2_CNN_IanCumes_23236.pdf`](report/Laboratorio2_CNN_IanCumes_23236.pdf) |
| Notebook completo y comentado | [`notebooks/Lab2_CNN_MNIST_IanCumes_23236.ipynb`](notebooks/Lab2_CNN_MNIST_IanCumes_23236.ipynb) |

## Estructura

```
src/data.py          Carga de MNIST, exploración, split estratificado y normalización
src/models.py        Arquitecturas MLP y CNN parametrizables + conteo de parámetros
src/train.py         Bucle de entrenamiento, métricas macro y evaluación
src/experiments.py   Las 12 configuraciones y la búsqueda (reanudable)
src/figures.py       Todas las figuras del laboratorio
src/report_data.py   Consolidación de resultados en tablas
report/build_report.py  Genera el PDF a partir de los resultados reales
results/             Resultados de la ejecución (JSON + figuras)
```

## Reproducir

```bash
bash run_all.sh
```

El script instala dependencias, corre las 12 iteraciones, evalúa una única vez sobre test, ejecuta el
notebook con sus outputs y genera el PDF. La búsqueda es **reanudable**: cada iteración se escribe a
`results/iterations.json` apenas termina, así que volver a lanzarla omite lo ya calculado.

Para correr solo una parte:

```bash
python3 -m src.experiments      # entrenamiento y evaluación
python3 report/build_report.py  # regenera el PDF desde results/
```

## Metodología

- **Split:** 54 000 entrenamiento / 6 000 validación (estratificado, `random_state=23236`) + 10 000 test.
  El conjunto de test se toca **una sola vez**, al final; toda la selección de hiperparámetros se hace
  sobre validación.
- **Normalización:** media y desviación calculadas **solo sobre el split de entrenamiento** para evitar
  *data leakage*.
- **Búsqueda:** secuencial, cambiando **una variable a la vez** respecto de la iteración anterior, de
  modo que cada delta de métrica sea atribuible a un único cambio.
- **Métricas:** accuracy, precision, recall y F1 **macro** (las clases de MNIST tienen un desbalance
  leve, ~9 %–11 %).
- **Reproducibilidad:** semilla 23236 fijada en `random`, `numpy` y `torch` al inicio de cada iteración.

Los números del PDF se generan programáticamente desde `results/*.json`; no se transcriben a mano.

## Requisitos

Python 3.11+ y las dependencias de [`requirements.txt`](requirements.txt). PyTorch CPU es suficiente:
la ejecución completa (12 iteraciones + reentrenamiento de los dos modelos finales) toma
**alrededor de 2 horas en 4 núcleos**, sin GPU. Como la búsqueda es reanudable, se puede cortar y
retomar sin perder trabajo.
