#!/usr/bin/env bash
# Reproduce el laboratorio completo de punta a punta:
# dependencias -> 12 iteraciones -> evaluacion en test -> notebook ejecutado -> PDF.
#
# Laboratorio #2 - CC3092 Deep Learning y Sistemas Inteligentes
# Ian Cumes - 23236
set -euo pipefail

cd "$(dirname "$0")"

echo "==> 1/4 Instalando dependencias"
pip install -q -r requirements.txt

echo "==> 2/4 Busqueda de hiperparametros (12 iteraciones) y evaluacion final en test"
# Reanudable: las iteraciones ya presentes en results/iterations.json se omiten.
python3 -u -m src.experiments

echo "==> 3/4 Ejecutando el notebook con sus outputs"
jupyter nbconvert --to notebook --execute --inplace \
  --ExecutePreprocessor.timeout=7200 \
  notebooks/Lab2_CNN_MNIST_IanCumes_23236.ipynb

echo "==> 4/4 Generando el reporte PDF (maximo 3 paginas)"
python3 report/build_report.py

echo "==> Listo."
echo "    PDF      : report/Laboratorio2_CNN_IanCumes_23236.pdf"
echo "    Notebook : notebooks/Lab2_CNN_MNIST_IanCumes_23236.ipynb"
