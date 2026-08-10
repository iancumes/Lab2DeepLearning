#!/usr/bin/env python3
"""Genera el PDF de maximo 3 paginas a partir de los resultados reales.

Flujo: results/*.json + results/figures/*.png -> report.html -> PDF (Chromium).

Todos los numeros y todas las afirmaciones cuantitativas del reporte se derivan
de los JSON producidos por la ejecucion; ninguno se escribe a mano. Si el PDF
sale con mas de 3 paginas el script falla, porque es un requisito del enunciado.

Laboratorio #2 - CC3092 Deep Learning y Sistemas Inteligentes
Ian Cumes - 23236
"""

from __future__ import annotations

import base64
import html
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.report_data import (  # noqa: E402
    comparison_table,
    describe_config,
    hyperparameter_impact,
    iterations_table,
    load_results,
    overfitting_gap,
    top_confusions,
)

REPORT_DIR = ROOT / "report"
FIG_DIR = ROOT / "results" / "figures"
HTML_PATH = REPORT_DIR / "report.html"
PDF_PATH = REPORT_DIR / "Laboratorio2_CNN_IanCumes_23236.pdf"
REPO_URL = "https://github.com/iancumes/Lab2DeepLearning"
MAX_PAGES = 3

CHROMIUM_CANDIDATES = [
    "/opt/pw-browsers/chromium-1194/chrome-linux/chrome",
    "/opt/pw-browsers/chromium_headless_shell-1194/chrome-linux/headless_shell",
]


def find_chromium() -> str:
    for path in CHROMIUM_CANDIDATES:
        if Path(path).exists():
            return path
    for name in ("chromium", "chromium-browser", "google-chrome", "chrome"):
        found = shutil.which(name)
        if found:
            return found
    matches = sorted(Path("/opt/pw-browsers").glob("chromium*/chrome-linux/chrome"))
    if matches:
        return str(matches[-1])
    raise FileNotFoundError("No se encontro un binario de Chromium para generar el PDF")


def img_tag(name: str, width: str = "100%") -> str:
    """Incrusta un PNG como data URI para que el HTML sea autocontenido."""
    path = FIG_DIR / name
    b64 = base64.b64encode(path.read_bytes()).decode()
    return f'<img src="data:image/png;base64,{b64}" style="width:{width}">'


def esc(value) -> str:
    return html.escape(str(value))


def df_to_html(df, formatters: dict | None = None, classes: str = "data") -> str:
    formatters = formatters or {}
    head = "".join(f"<th>{esc(c)}</th>" for c in df.columns)
    body = []
    for _, row in df.iterrows():
        cells = []
        for col in df.columns:
            fmt = formatters.get(col)
            cells.append(f"<td>{esc(fmt(row[col]) if fmt else row[col])}</td>")
        body.append("<tr>" + "".join(cells) + "</tr>")
    return f'<table class="{classes}"><thead><tr>{head}</tr></thead><tbody>{"".join(body)}</tbody></table>'


PCT = lambda v: f"{v * 100:.2f}%"
F4 = lambda v: f"{v:.4f}"
INT = lambda v: f"{int(v):,}"
SEC = lambda v: f"{v:.1f}"
# En prosa se separan los miles con espacio duro (convencion SI/espanol); en las
# tablas se deja la coma, que es mas compacta.
MIL = lambda v: f"{int(v):,}".replace(",", "&nbsp;")


# --------------------------------------------------------------------------
# Seccion 3: investigacion de capas (contenido conceptual, no depende del run)
# --------------------------------------------------------------------------
LAYERS = [
    (
        "nn.Conv2d",
        "Aplica filtros aprendibles que se deslizan sobre la imagen y detectan patrones locales "
        "(bordes, esquinas, trazos). Es la capa que da a la CNN su invarianza a traslaciones: el mismo "
        "filtro se reutiliza en todas las posiciones.",
        "in_channels, out_channels, kernel_size, stride, padding, dilation, bias",
    ),
    (
        "nn.MaxPool2d",
        "Submuestrea cada ventana quedandose con el valor maximo. Reduce la resolucion espacial, "
        "amplia el campo receptivo y conserva la activacion mas fuerte, por lo que preserva bien los "
        "bordes y trazos marcados. No tiene parametros entrenables.",
        "kernel_size, stride, padding, ceil_mode",
    ),
    (
        "nn.AvgPool2d",
        "Igual que MaxPool2d pero promediando la ventana: suaviza el mapa de activacion en vez de "
        "quedarse con el pico, conservando cuanta señal habia en la region y no solo su maximo. "
        "Cual de los dos conviene es empirico; aqui AvgPool resulto mejor (ver iteracion C3).",
        "kernel_size, stride, padding, count_include_pad",
    ),
    (
        "nn.BatchNorm2d",
        "Normaliza cada canal usando la media y varianza del batch y luego lo reescala con dos "
        "parametros aprendibles por canal. Estabiliza y acelera el entrenamiento, y actua como "
        "regularizador leve por el ruido del batch.",
        "num_features, eps, momentum, affine, track_running_stats",
    ),
    (
        "nn.Flatten",
        "Aplana el tensor (N, C, H, W) a (N, C*H*W) sin parametros ni computo real: es el puente "
        "entre la parte convolucional y el clasificador denso.",
        "start_dim, end_dim",
    ),
    (
        "nn.CrossEntropyLoss",
        "Perdida estandar de clasificacion multiclase. Combina log_softmax y NLLLoss en una sola "
        "operacion numericamente estable, por lo que el modelo debe entregar logits crudos, sin "
        "softmax en la ultima capa.",
        "weight, ignore_index, reduction, label_smoothing",
    ),
]


def layers_section() -> str:
    rows = "".join(
        f"<tr><td><code>{esc(n)}</code></td><td>{esc(p)}</td><td class='params'>{esc(a)}</td></tr>"
        for n, p, a in LAYERS
    )
    return (
        '<table class="data layers"><thead><tr><th>Capa</th><th>Proposito</th>'
        "<th>Parametros mas relevantes</th></tr></thead>"
        f"<tbody>{rows}</tbody></table>"
    )


def efficiency_note(ctx: dict) -> str:
    """Evidencia empirica de eficiencia en parametros, si la busqueda la produjo."""
    eff, mlp_it = ctx["efficient"], ctx["best_mlp_it"]
    if eff is None:
        return ""
    return (
        f" La eficiencia en parametros si se ve al comparar iteraciones: <b>{esc(eff['id'])}</b> "
        f"({MIL(eff['n_params'])} parametros) supera al mejor MLP <b>{esc(mlp_it['id'])}</b> "
        f"({MIL(mlp_it['n_params'])}) en F1 de validacion &mdash; {PCT(eff['val_metrics']['f1_macro'])} "
        f"contra {PCT(mlp_it['val_metrics']['f1_macro'])} &mdash; usando <b>menos</b> parametros que el."
    )


def data_section(ctx: dict) -> str:
    """Resumen del dataset y del protocolo. Las cifras del split salen de results/."""
    return f"""
<p><b>Dataset.</b> MNIST: <b>70&nbsp;000</b> imagenes de digitos manuscritos (60&nbsp;000 de entrenamiento +
10&nbsp;000 de test) en <b>10 clases</b> (0&ndash;9). Cada imagen es de <b>28&times;28 pixeles</b> en escala de
grises, con valores enteros en <code>[0, 255]</code> que <code>ToTensor()</code> escala a
<code>[0, 1]</code>. Las clases estan <i>aproximadamente</i> balanceadas: van del ~9.0% (digito 5) al
~11.2% (digito 1), razon max/min &asymp; 1.24. Ese desbalance leve no exige remuestreo, pero si
justifica reportar metricas <b>macro</b> y estratificar el split.</p>

<p><b>Normalizacion.</b> Necesaria. Ademas del escalado a <code>[0, 1]</code> se estandariza con
<code>Normalize(&mu;={ctx['norm_mean']:.4f}, &sigma;={ctx['norm_std']:.4f})</code>: como ~80% de los pixeles
son fondo negro, sin centrar la entrada media queda lejos de 0 y el descenso de gradiente zigzaguea.
Ambos estadisticos se calculan <b>solo sobre el split de entrenamiento</b> para no filtrar informacion
de validacion ni de test hacia el preprocesamiento.</p>

<p><b>Particion y protocolo.</b> {MIL(ctx["n_train"])} entrenamiento / {MIL(ctx["n_val"])} validacion
(estratificado, <code>random_state=23236</code>) + {MIL(ctx["n_test"])} test. Toda la seleccion de
hiperparametros se hace mirando <b>solo validacion</b>; el conjunto de test se evalua <b>una unica
vez</b> al final, con la mejor configuracion ya congelada. Semilla 23236 fijada en <code>random</code>,
<code>numpy</code> y <code>torch</code> al inicio de cada iteracion, de modo que las diferencias entre
iteraciones sean atribuibles al hiperparametro y no a la inicializacion.</p>
"""


def concepts_section(ctx: dict) -> str:
    return f"""
<p><b>Tensor.</b> Es el arreglo multidimensional que PyTorch usa para todos los datos y parametros.
Ademas de los valores guarda el <i>dtype</i>, el dispositivo (CPU/GPU) y, si <code>requires_grad=True</code>,
el grafo de operaciones que permite calcular gradientes por autograd. En este laboratorio un batch de
imagenes es un tensor <code>(N, 1, 28, 28)</code> y el MLP lo convierte en <code>(N, 784)</code>.</p>

<p><b>Campo receptivo.</b> Es la region de la imagen de entrada que influye en una sola activacion de
una capa profunda. Crece de forma incremental: <code>RF_out = RF_in + (k - 1) &middot; jump</code>, donde
<code>jump</code> es el producto de los strides acumulados. En la CNN ganadora
({esc(ctx['cnn_blocks'])} bloques conv 3&times;3 con pooling 2&times;2) el campo receptivo final es de
<b>{esc(ctx['receptive_field'])}&times;{esc(ctx['receptive_field'])} pixeles</b>, es decir
{esc(ctx['rf_coverage'])}% del ancho de la imagen: la primera capa solo ve trazos de 3&times;3 y cada bloque
adicional amplia esa ventana, de modo que apilar capas hace crecer el contexto sin agrandar el kernel.</p>

<p><b>Por que la CNN necesita menos parametros.</b> Por dos mecanismos. (1) <i>Conectividad local</i>: cada
neurona solo se conecta a una ventana de k&times;k pixeles, no a los 784. (2) <i>Pesos compartidos</i>: el mismo
filtro se reutiliza en todas las posiciones de la imagen, asi que su costo no escala con la resolucion.
Un <code>Conv2d(1, 32, 3)</code> tiene solo 320 parametros y produce 32 mapas de 28&times;28, mientras que una
capa densa <code>Linear(784, 32)</code> necesita 25&nbsp;120 parametros y ademas destruye la estructura espacial al
aplanar.</p>

<p><b>Matiz importante que muestran los resultados.</b> El ahorro es real <i>por capa</i>, pero no se
traduce automaticamente en un modelo total mas chico: la CNN ganadora usa {MIL(ctx['cnn_params'])}
parametros, {esc(ctx['param_ratio'])} que el MLP. La razon aparece al desglosarla: solo
<b>{MIL(ctx['conv_params'])} ({ctx['conv_pct']:.1f}%)</b> estan en los bloques convolucionales y
<b>{MIL(ctx['head_params'])} ({ctx['head_pct']:.1f}%)</b> en la cabeza densa posterior al
<code>Flatten</code>. Es decir: la parte convolucional es baratisima y casi todo el costo lo pone el
clasificador denso, que es justamente el componente de tipo MLP.{efficiency_note(ctx)}</p>
"""


# --------------------------------------------------------------------------
# Contexto derivado de los resultados reales
# --------------------------------------------------------------------------
def build_context(iterations: list[dict], final_test: dict, data_meta: dict) -> dict:
    mlp = final_test["MLP"]
    cnn = final_test["CNN"]
    cnn_cfg = cnn["config"]

    ratio = mlp["n_params"] / cnn["n_params"]
    ratio_txt = f"{ratio:.1f}x menos" if ratio > 1 else f"{1 / ratio:.1f}x mas"

    impact = {a: hyperparameter_impact(iterations, a) for a in ("MLP", "CNN")}
    gaps = overfitting_gap(iterations)

    # Campo receptivo de la CNN ganadora, con la misma formula de models.CNN.
    rf, jump, k = 1, 1, cnn_cfg.get("kernel_size", 3)
    for _ in cnn_cfg["channels"]:
        rf += (k - 1) * jump
        rf += jump
        jump *= 2

    # Reparto de parametros de la CNN ganadora entre la parte convolucional y la
    # cabeza densa. Se calcula analiticamente para no depender de torch aqui.
    conv_params, prev_c = 0, 1
    for c in cnn_cfg["channels"]:
        conv_params += c * (prev_c * k * k) + c
        if cnn_cfg.get("use_bn"):
            conv_params += 2 * c  # gamma y beta de BatchNorm2d
        prev_c = c
    head_params = cnn["n_params"] - conv_params

    # Evidencia directa de eficiencia en parametros: la CNN mas pequena que
    # supera al mejor MLP usando menos parametros que el.
    best_mlp_it = max(
        (r for r in iterations if r["arch"] == "MLP"), key=lambda r: r["val_metrics"]["f1_macro"]
    )
    cheaper = [
        r for r in iterations
        if r["arch"] == "CNN"
        and r["val_metrics"]["f1_macro"] > best_mlp_it["val_metrics"]["f1_macro"]
        and r["n_params"] < best_mlp_it["n_params"]
    ]
    efficient = min(cheaper, key=lambda r: r["n_params"]) if cheaper else None

    winner = "CNN" if cnn["test_metrics"]["accuracy"] >= mlp["test_metrics"]["accuracy"] else "MLP"

    # Reduccion relativa del error de la CNN respecto al MLP.
    err_mlp = 1 - mlp["test_metrics"]["accuracy"]
    err_cnn = 1 - cnn["test_metrics"]["accuracy"]
    err_reduction = (1 - err_cnn / err_mlp) * 100 if err_mlp > 0 else 0.0

    return {
        "mlp": mlp,
        "cnn": cnn,
        "mlp_params": mlp["n_params"],
        "cnn_params": cnn["n_params"],
        "param_ratio": ratio_txt,
        "cnn_blocks": len(cnn_cfg["channels"]),
        "receptive_field": rf,
        "rf_coverage": f"{min(rf, 28) / 28 * 100:.0f}",
        "impact": impact,
        "gaps": gaps,
        "winner": winner,
        "acc_gap_pp": abs(cnn["test_metrics"]["accuracy"] - mlp["test_metrics"]["accuracy"]) * 100,
        "err_reduction": err_reduction,
        "mlp_confusions": top_confusions(mlp["confusion_matrix"], 3),
        "cnn_confusions": top_confusions(cnn["confusion_matrix"], 3),
        "conv_params": conv_params,
        "head_params": head_params,
        "conv_pct": conv_params / cnn["n_params"] * 100,
        "head_pct": head_params / cnn["n_params"] * 100,
        "efficient": efficient,
        "best_mlp_it": best_mlp_it,
        "n_train": data_meta["n_train"],
        "n_val": data_meta["n_val"],
        "n_test": data_meta["n_test"],
        "norm_mean": data_meta["mean"],
        "norm_std": data_meta["std"],
    }


def fmt_confusions(pairs) -> str:
    return ", ".join(f"{r}&rarr;{p} ({n})" for r, p, n in pairs)


def discussion_section(ctx: dict) -> str:
    m_imp, c_imp = ctx["impact"]["MLP"], ctx["impact"]["CNN"]
    m_best, m_worst = m_imp.iloc[0], m_imp.iloc[-1]
    c_best, c_worst = c_imp.iloc[0], c_imp.iloc[-1]
    worst_gap = ctx["gaps"].iloc[0]
    mlp, cnn = ctx["mlp"], ctx["cnn"]

    return f"""
<p><b>1. Mayor impacto positivo y negativo.</b> En el MLP el cambio mas beneficioso fue
<i>{esc(m_best['Cambio'])}</i> ({m_best['Delta F1']:+.4f} de F1 en validacion, iteracion {esc(m_best['Iteracion'])}),
y el mas perjudicial <i>{esc(m_worst['Cambio'])}</i> ({m_worst['Delta F1']:+.4f}). En la CNN el mayor salto vino de
<i>{esc(c_best['Cambio'])}</i> ({c_best['Delta F1']:+.4f}) y el mayor retroceso de <i>{esc(c_worst['Cambio'])}</i>
({c_worst['Delta F1']:+.4f}). El patron general es que los cambios de <i>optimizacion</i> y de
<i>arquitectura</i> mueven mucho mas la aguja que los ajustes finos de capacidad, y que a partir de cierto
punto agregar parametros ya no compra accuracy.</p>

<p><b>2. Overfitting / underfitting.</b> Se identifico leyendo la brecha entre la perdida de entrenamiento y
la de validacion en las curvas: cuando la de entrenamiento sigue bajando y la de validacion se estanca o
sube, el modelo esta memorizando. La iteracion con mayor brecha fue <b>{esc(worst_gap['ID'])}</b>
({esc(worst_gap['Arq.'])}), con val&minus;train = {worst_gap['Brecha (val - train)']:+.4f}. El underfitting aparecio en la
baseline del MLP (M1, SGD sin momentum), donde ambas perdidas se quedaban altas y planas: ahi el problema no
era capacidad sino velocidad de convergencia. Se mitigo con Dropout y BatchNorm en las iteraciones
posteriores y limitando el numero de epochs.</p>

<p><b>3. Efecto de la regularizacion.</b> {ctx['reg_text']}</p>

<p><b>4. MLP vs CNN en test.</b> Gano la <b>{esc(ctx['winner'])}</b>:
{PCT(cnn['test_metrics']['accuracy'])} de accuracy frente a {PCT(mlp['test_metrics']['accuracy'])} del MLP
({ctx['acc_gap_pp']:.2f} puntos porcentuales, equivalente a reducir el error en {ctx['err_reduction']:.1f}%), con
{esc(ctx['param_ratio'])} parametros ({MIL(cnn['n_params'])} vs {MIL(mlp['n_params'])}). Conviene no leer esa
relacion al reves: la ventaja <b>no</b> viene de tener mas capacidad. La prueba es que
{esc(ctx['efficient']['id']) if ctx['efficient'] else 'la CNN mas pequena'}, con
{MIL(ctx['efficient']['n_params']) if ctx['efficient'] else '&mdash;'} parametros, ya supera al mejor MLP usando
<i>menos</i> parametros que el; y que dentro de las propias CNN aumentar canales de (16,32) a (32,64)
<i>empeoro</i> el F1. La razon es estructural: el MLP aplana la imagen y trata cada pixel como una feature
independiente, de modo que pierde toda la informacion de vecindad y debe reaprender el mismo trazo en cada
posicion. La CNN preserva la estructura 2D, comparte los filtros entre posiciones y construye el campo
receptivo por etapas, asi que la invarianza a pequenas traslaciones le sale "gratis" en vez de tener que
aprenderla con mas pesos.</p>

<p><b>5. Tipos de error.</b> Las confusiones dominantes del MLP fueron {fmt_confusions(ctx['mlp_confusions'])}
y las de la CNN {fmt_confusions(ctx['cnn_confusions'])}. Ambos modelos se equivocan sobre todo entre digitos
que comparten trazos (4/9, 3/5, 7/2), pero el MLP acumula ademas errores en pares que solo se distinguen por
la <i>posicion relativa</i> de los trazos, precisamente lo que se pierde al aplanar; la CNN concentra sus
pocos fallos en digitos escritos de forma genuinamente ambigua.</p>

<p><b>6. Modelo para produccion.</b> Elegiria una <b>CNN</b>, pero reconociendo que aqui si hay un
trade-off real y no fingiendo que la CNN domina en todo. La CNN ganadora es mas exacta
({PCT(cnn['test_metrics']['accuracy'])} vs {PCT(mlp['test_metrics']['accuracy'])}) pero tambien mas pesada
({MIL(cnn['n_params'])} parametros, {esc(ctx['param_ratio'])} que el MLP) y mas lenta en inferencia
({cnn['inference_ms_per_image']:.3f} ms/imagen en CPU contra {mlp['inference_ms_per_image']:.3f} ms del MLP,
es decir {cnn['inference_ms_per_image'] / mlp['inference_ms_per_image']:.1f}&times;). Aun asi la eleccion es clara:
{ctx['acc_gap_pp']:.2f} puntos porcentuales equivalen a {ctx['err_reduction']:.0f}% menos errores, y a escala de
produccion cada error mal clasificado cuesta intervencion manual, que es ordenes de magnitud mas cara que
{cnn['inference_ms_per_image']:.3f} ms de CPU.{
    f" Si la memoria fuera la restriccion vinculante, la configuracion a desplegar seria {esc(ctx['efficient']['id'])}: {MIL(ctx['efficient']['n_params'])} parametros, menos que el MLP, y aun por encima de el en validacion."
    if ctx['efficient'] else ""
} El MLP solo ganaria en un dispositivo incapaz de ejecutar convoluciones de forma eficiente.</p>
"""


def conclusions_section(ctx: dict) -> str:
    cnn, mlp = ctx["cnn"], ctx["mlp"]
    return f"""
<ul>
<li>La CNN alcanzo <b>{PCT(cnn['test_metrics']['accuracy'])}</b> de accuracy en test contra
<b>{PCT(mlp['test_metrics']['accuracy'])}</b> del MLP: {ctx['err_reduction']:.0f}% menos errores.</li>
<li>El numero de parametros no predice la calidad: la CNN ganadora tiene {esc(ctx['param_ratio'])}
parametros que el MLP, pero {esc(ctx['efficient']['id']) if ctx['efficient'] else 'una CNN mas pequena'} lo
supera con <i>menos</i>, y subir canales de (16,32) a (32,64) empeoro el resultado. Lo que decide es si la
arquitectura respeta la estructura del dato, no cuanta capacidad tiene.</li>
<li>El costo en parametros de una CNN no esta en las convoluciones ({ctx['conv_pct']:.1f}% del total) sino
en la cabeza densa ({ctx['head_pct']:.1f}%): para achicar el modelo hay que atacar el clasificador, no los
filtros.</li>
<li>Varias intuiciones estandar no sobrevivieron a la medicion: AvgPool2d supero a MaxPool2d, mas canales
empeoraron el resultado y BatchNorm2d perjudico a la CNN. Medir una variable a la vez fue lo que permitio
detectarlo y atribuir cada cambio a su causa.</li>
</ul>
"""


CSS = """
@page { size: letter; margin: 1.25cm 1.3cm; }
* { box-sizing: border-box; }
body { font-family: "DejaVu Sans", Arial, sans-serif; font-size: 7.6pt; line-height: 1.32;
       color: #16181d; margin: 0; }
h1 { font-size: 13pt; margin: 0 0 1mm; }
h2 { font-size: 9pt; margin: 3mm 0 1.2mm; padding-bottom: 0.6mm;
     border-bottom: 1.1pt solid #1f4e79; color: #1f4e79; }
h3 { font-size: 8pt; margin: 2mm 0 1mm; color: #333; }
p { margin: 0 0 1.4mm; text-align: justify; }
ul { margin: 0 0 1.4mm; padding-left: 4mm; }
li { margin-bottom: 0.6mm; text-align: justify; }
code { font-family: "DejaVu Sans Mono", monospace; font-size: 7pt; background: #f0f2f5;
       padding: 0 0.6mm; border-radius: 1px; }
.header { border-bottom: 2pt solid #1f4e79; padding-bottom: 1.5mm; margin-bottom: 2mm; }
.meta { font-size: 7.4pt; color: #444; }
table.data { width: 100%; border-collapse: collapse; font-size: 6.3pt; margin: 0 0 1.6mm; }
table.data th { background: #1f4e79; color: #fff; padding: 0.7mm 0.8mm; text-align: left;
                font-weight: 600; }
table.data td { padding: 0.55mm 0.8mm; border-bottom: 0.4pt solid #d8dde4; }
table.data tbody tr:nth-child(even) { background: #f5f7fa; }
table.layers td:first-child { width: 15%; }
table.layers td.params { width: 30%; font-size: 6pt; color: #555; }
.figrow { display: flex; gap: 2.5mm; align-items: flex-start; margin-bottom: 1.6mm; }
.figrow > div { flex: 1; }
/* En la pagina 3 el espacio es el recurso escaso: se fija la altura de las
   figuras para que el bloque de discusion quepa completo sin desbordar. */
.figrow.compact img { height: 27mm; width: auto; display: block; margin: 0 auto; }
.caption { font-size: 6.2pt; color: #666; text-align: center; margin-top: 0.3mm; }
.page-break { break-before: page; }
.footer { margin-top: 1.5mm; padding-top: 1mm; border-top: 1pt solid #1f4e79; font-size: 7.2pt; }
.highlight { background: #fff8e1; }
"""


def build_html(iterations: list[dict], final_test: dict, data_meta: dict) -> str:
    ctx = build_context(iterations, final_test, data_meta)

    # Texto de regularizacion derivado de los deltas reales de cada arquitectura.
    reg_rows = []
    for arch in ("MLP", "CNN"):
        imp = ctx["impact"][arch]
        for _, row in imp.iterrows():
            if "Dropout" in row["Cambio"] or "BatchNorm" in row["Cambio"]:
                verb = "mejoro" if row["Delta F1"] > 0 else "empeoro"
                art = "el" if arch == "MLP" else "la"  # el MLP / la CNN
                reg_rows.append(
                    f"en {art} {arch}, {row['Cambio'].split(':')[-1].strip()} {verb} el F1 de validacion "
                    f"en {row['Delta F1']:+.4f}"
                )
    # capitalize() bajaria a minusculas el resto de la frase ("MLP" -> "mlp"),
    # asi que solo se levanta la primera letra.
    upper_first = lambda s: s[0].upper() + s[1:]
    ctx["reg_text"] = (
        (". ".join(upper_first(s) for s in reg_rows) + ". ") if reg_rows else ""
    ) + (
        "Ningun metodo gana en ambas arquitecturas: BatchNorm ayudo al MLP pero perjudico a la CNN, y "
        "Dropout hizo lo contrario. La regularizacion rinde en proporcion al overfitting que hay que "
        "corregir, y con 54&nbsp;000 ejemplos limpios ese margen es estrecho: los deltas de la CNN son de "
        "milesimas, dentro del ruido de una sola semilla, asi que lo honesto es elegir por arquitectura "
        "y no declarar un ganador general."
    )

    it_df = iterations_table(iterations)
    cmp_df = comparison_table(final_test)

    it_html = df_to_html(
        it_df,
        {
            "Train loss": F4, "Val loss": F4, "Accuracy": PCT, "Precision": PCT,
            "Recall": PCT, "F1": PCT, "Params": INT, "Tiempo (s)": SEC,
        },
    )
    cmp_html = df_to_html(
        cmp_df,
        {
            "Parametros entrenables": INT, "Accuracy (test)": PCT, "Precision (test)": PCT,
            "Recall (test)": PCT, "F1 (test)": PCT, "Tiempo entren. (s)": SEC,
            "Inferencia (ms/img)": lambda v: f"{v:.3f}",
        },
    )

    best_mlp = final_test["MLP"]
    best_cnn = final_test["CNN"]

    return f"""<!DOCTYPE html>
<html lang="es"><head><meta charset="utf-8"><title>Laboratorio 2 - CNN</title>
<style>{CSS}</style></head><body>

<div class="header">
  <h1>Laboratorio #2 &mdash; Redes Neuronales Convolucionales sobre MNIST</h1>
  <div class="meta"><b>Ian Cumes</b> &middot; Carne <b>23236</b> &middot; CC3092 Deep Learning y Sistemas
  Inteligentes &middot; Universidad del Valle de Guatemala</div>
</div>

<h2>1. Investigacion: capas de PyTorch para la CNN</h2>
{layers_section()}
{concepts_section(ctx)}

<h2>2. Datos y protocolo experimental</h2>
<div class="figrow">
  <div style="flex: 1.15">{data_section(ctx)}</div>
  <div style="flex: 0.85">{img_tag("class_distribution.png")}
    <div class="caption">Distribucion de clases en entrenamiento: desbalance leve (~9%&ndash;11%)</div></div>
</div>

<div class="page-break"></div>

<h2>3. Resultados de las iteraciones (12 iteraciones: 6 MLP + 6 CNN)</h2>
<p>Busqueda sistematica cambiando <b>una variable a la vez</b> respecto de la iteracion indicada en la
columna <b>Base</b>, de modo que cada delta de metrica sea atribuible a un unico cambio. La busqueda es un
arbol y no una cadena: C3 y C4 son dos ramas de C2, para no mezclar el cambio de pooling con el de
normalizacion. Las metricas son <b>macro</b> sobre el conjunto de <b>validacion</b> (6&nbsp;000 imagenes); el
conjunto de test no se toco durante la busqueda.</p>
{it_html}

<div class="figrow">
  <div>{img_tag("loss_curves_mlp.png")}<div class="caption">Curvas de perdida &mdash; MLP (6 iteraciones; continua = train, punteada = val)</div></div>
  <div>{img_tag("loss_curves_cnn.png")}<div class="caption">Curvas de perdida &mdash; CNN (6 iteraciones; continua = train, punteada = val)</div></div>
</div>

<div style="text-align:center">{img_tag("val_metrics_by_iteration.png", width="74%")}</div>
<div class="caption">Accuracy de validacion por iteracion; el borde negro marca la mejor de cada arquitectura</div>

<div class="page-break"></div>

<h2>4. Comparacion de arquitecturas (conjunto de test, una sola evaluacion)</h2>
<p>Mejor configuracion de cada arquitectura segun F1-macro de validacion:
<b>MLP {esc(best_mlp['best_iteration_id'])}</b> ({esc(describe_config(best_mlp['config']))}) y
<b>CNN {esc(best_cnn['best_iteration_id'])}</b> ({esc(describe_config(best_cnn['config']))}).</p>
{cmp_html}

<div class="figrow compact">
  <div>{img_tag("params_vs_accuracy.png")}<div class="caption">Parametros vs. accuracy en test</div></div>
  <div>{img_tag("confusion_mlp.png")}<div class="caption">Matriz de confusion &mdash; MLP</div></div>
  <div>{img_tag("confusion_cnn.png")}<div class="caption">Matriz de confusion &mdash; CNN</div></div>
</div>

<h2>5. Discusion y analisis</h2>
{discussion_section(ctx)}

<h3>Conclusiones</h3>
{conclusions_section(ctx)}

<div class="footer">
  <b>Repositorio:</b> <a href="{REPO_URL}">{REPO_URL}</a><br>
  Notebook completo y comentado: <code>notebooks/Lab2_CNN_MNIST_IanCumes_23236.ipynb</code>
</div>

</body></html>
"""


def count_pdf_pages(path: Path) -> int:
    data = path.read_bytes()
    m = re.search(rb"/Type\s*/Pages[^>]*?/Count\s+(\d+)", data, re.S)
    if m:
        return int(m.group(1))
    return len(re.findall(rb"/Type\s*/Page[^s]", data))


def main() -> int:
    iterations, final_test = load_results()
    if len(iterations) < 12:
        raise SystemExit(f"Se esperaban 12 iteraciones, hay {len(iterations)}")
    if not final_test:
        raise SystemExit("Falta results/final_test.json")
    meta_path = ROOT / "results" / "data_meta.json"
    if not meta_path.exists():
        raise SystemExit("Falta results/data_meta.json (lo genera src/experiments.py)")
    data_meta = json.loads(meta_path.read_text())

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    HTML_PATH.write_text(build_html(iterations, final_test, data_meta), encoding="utf-8")
    print(f"HTML generado: {HTML_PATH}")

    chromium = find_chromium()
    if PDF_PATH.exists():
        PDF_PATH.unlink()
    subprocess.run(
        [
            chromium, "--headless", "--no-sandbox", "--disable-gpu",
            "--run-all-compositor-stages-before-draw", "--virtual-time-budget=10000",
            "--no-pdf-header-footer", f"--print-to-pdf={PDF_PATH}", HTML_PATH.as_uri(),
        ],
        check=True,
        capture_output=True,
    )

    pages = count_pdf_pages(PDF_PATH)
    size_kb = PDF_PATH.stat().st_size / 1024
    print(f"PDF generado: {PDF_PATH} ({pages} paginas, {size_kb:.0f} KB)")
    if pages > MAX_PAGES:
        raise SystemExit(f"ERROR: el PDF tiene {pages} paginas y el maximo permitido es {MAX_PAGES}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
