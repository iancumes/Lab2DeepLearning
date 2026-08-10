# Explicación completa del Laboratorio #2 — Redes Neuronales Convolucionales

**Curso:** CC3092 — Deep Learning y Sistemas Inteligentes
**Estudiante:** Ian Cumes · **Carné:** 23236
**Repositorio:** https://github.com/iancumes/Lab2DeepLearning

Este documento explica **todo lo que se trabajó** en el laboratorio, con el mismo nivel de
profundidad que exige el enunciado: qué es cada concepto, qué significa cada parámetro de cada capa,
por qué se tomó cada decisión, y qué dicen realmente los resultados obtenidos. Está pensado para
leerse de principio a fin sin tener el PDF de requerimientos al lado — si algo del enunciado se
menciona, aquí se explica desde cero.

Todas las cifras citadas salen de `results/iterations.json` y `results/final_test.json` — ninguna
está inventada ni redondeada de forma optimista. Puedes verificarlas abriendo esos archivos.

---

## Índice

1. [Objetivo del laboratorio](#1-objetivo-del-laboratorio)
2. [El dataset: MNIST](#2-el-dataset-mnist)
3. [Tensores y PyTorch](#3-tensores-y-pytorch)
4. [Las capas de `torch.nn`, explicadas una por una](#4-las-capas-de-torchnn-explicadas-una-por-una)
5. [Campo receptivo](#5-campo-receptivo)
6. [Por qué una CNN necesita menos parámetros que un MLP](#6-por-qué-una-cnn-necesita-menos-parámetros-que-un-mlp)
7. [Las dos arquitecturas implementadas](#7-las-dos-arquitecturas-implementadas)
8. [Metodología de entrenamiento](#8-metodología-de-entrenamiento)
9. [La búsqueda de hiperparámetros: las 12 iteraciones](#9-la-búsqueda-de-hiperparámetros-las-12-iteraciones)
10. [Resultados finales y comparación](#10-resultados-finales-y-comparación)
11. [Discusión: lo que confirmó la intuición y lo que la contradijo](#11-discusión-lo-que-confirmó-la-intuición-y-lo-que-la-contradijo)
12. [Respuestas a las preguntas de discusión del enunciado](#12-respuestas-a-las-preguntas-de-discusión-del-enunciado)
13. [Glosario](#13-glosario)
14. [Mapa del repositorio](#14-mapa-del-repositorio)

---

## 1. Objetivo del laboratorio

El enunciado (CC3092, Laboratorio #2) pide construir, entrenar y comparar dos tipos de red neuronal
sobre el mismo problema de clasificación:

- Un **MLP** (*Multi-Layer Perceptron*, perceptrón multicapa): una red de capas totalmente
  conectadas (`Linear`), que recibe la imagen **aplanada** como un vector.
- Una **CNN** (*Convolutional Neural Network*, red neuronal convolucional): una red que recibe la
  imagen como **tensor 2D** y usa al menos dos capas convolucionales.

El objetivo no es solo "que funcionen", sino **entender por qué una funciona mejor que la otra** para
datos con estructura espacial como las imágenes, y demostrarlo empíricamente con una búsqueda
sistemática de hiperparámetros (mínimo 10 iteraciones, aquí se hicieron 12), midiendo en cada paso
qué efecto tuvo cada cambio.

El dataset elegido es **MNIST**: dígitos manuscritos del 0 al 9. Es el "hola mundo" de la visión por
computadora porque es pequeño, limpio, bien balanceado y suficientemente difícil como para que la
diferencia entre arquitecturas sea visible pero no abrumadora.

---

## 2. El dataset: MNIST

### 2.1 Qué es y cuánto pesa

MNIST tiene **70 000 imágenes** en total: 60 000 de entrenamiento y 10 000 de test, repartidas en
**10 clases** (los dígitos 0–9). Cada imagen es de **28×28 píxeles**, en **escala de grises** (un
solo canal de color), con valores de intensidad originalmente enteros en el rango `[0, 255]`.

En este laboratorio, ese conjunto de 60 000 se dividió además en:

| Conjunto | Tamaño | Uso |
|---|---|---|
| Entrenamiento | 54 000 | Ajustar los pesos del modelo (backpropagation) |
| Validación | 6 000 | Elegir la mejor configuración de hiperparámetros |
| Test | 10 000 | Evaluar el modelo final **una sola vez** |

La separación entre validación y test es uno de los puntos metodológicos más importantes del
laboratorio: **la validación se usa para decidir**, el **test se usa para medir**. Si se usara el
mismo conjunto para ambas cosas, la métrica final estaría inflada — habríamos elegido la
configuración que mejor le queda a ese conjunto particular, no la que generaliza mejor. Por eso el
protocolo del laboratorio evalúa 12 configuraciones distintas sobre validación, elige la mejor de
cada arquitectura, y **solo entonces** las toca contra test, una vez cada una.

### 2.2 ¿Están balanceadas las clases?

No perfectamente, pero casi. En el split de entrenamiento real usado (54 000 imágenes):

| Dígito | Observaciones (train) | % |
|---|---|---|
| 0 | 5 331 | 9.87 % |
| 1 | 6 068 | 11.24 % |
| 2 | 5 362 | 9.93 % |
| 3 | 5 518 | 10.22 % |
| 4 | 5 258 | 9.74 % |
| 5 | 4 879 | 9.04 % |
| 6 | 5 326 | 9.86 % |
| 7 | 5 638 | 10.44 % |
| 8 | 5 266 | 9.75 % |
| 9 | 5 354 | 9.92 % |

El dígito **5** es el menos frecuente (9.04 %) y el **1** el más frecuente (11.24 %), con una razón
máximo/mínimo de **≈1.24**. Es un desbalance leve — nada que requiera sobremuestreo, submuestreo ni
pesos por clase — pero sí es suficiente para justificar dos decisiones tomadas en el laboratorio:

1. Usar métricas **macro** (se explican en la sección 8) en vez de simplemente "accuracy global",
   para que las clases pequeñas no queden invisibles detrás de las grandes.
2. Hacer el split train/validación **estratificado**: se preserva la misma proporción de cada dígito
   en ambos subconjuntos, para que la comparación entre iteraciones no dependa de qué dígitos
   cayeron por azar en cada lado.

### 2.3 Dimensión de la imagen y rango de los píxeles

Cada imagen es un arreglo de **28 × 28 = 784 píxeles**, en un solo canal (escala de grises, no
RGB). El archivo original de MNIST guarda esos valores como **enteros sin signo de 8 bits**
(`uint8`), es decir, en el rango `[0, 255]`, donde 0 es negro puro y 255 es blanco puro.

Cuando `torchvision.transforms.ToTensor()` carga la imagen, la convierte automáticamente a
**float32** y la reescala a `[0, 1]` (divide entre 255). Ese primer escalado ya es indispensable:
alimentar una red con valores en el orden de las centenas produce activaciones y gradientes
desproporcionadamente grandes, lo que satura las no-linealidades y desestabiliza el entrenamiento.

### 2.4 ¿Es necesario normalizar?

Sí, y hay dos escalados distintos en juego que conviene no confundir:

1. **Escalado a `[0, 1]`** (`ToTensor()`): obligatorio, ya mencionado arriba.
2. **Estandarización** (`Normalize(mean, std)`): centra los datos para que tengan media 0 y
   desviación estándar 1, aplicando `x_normalizado = (x - mean) / std`.

La estandarización importa especialmente en MNIST por un motivo concreto: **el 80.9 % de los
píxeles son exactamente 0** (fondo negro). Sin centrar, la entrada promedio de la red no está cerca
de 0 sino de ~0.13, y ese sesgo sistemático hace que el descenso de gradiente **zigzaguee** en vez de
avanzar en línea recta hacia el mínimo de la función de pérdida — la superficie de optimización queda
mal condicionada.

El detalle metodológico que hay que cuidar aquí es que la media y la desviación **se calculan
únicamente sobre el conjunto de entrenamiento** (`src/data.py::compute_train_stats`), nunca sobre
todo el dataset. Si se calcularan sobre el total, información del conjunto de validación y de test
se estaría filtrando indirectamente hacia el preprocesamiento — un tipo de fuga de datos (*data
leakage*) que infla artificialmente el desempeño reportado.

En este laboratorio, calculadas sobre el split real de 54 000 imágenes de entrenamiento, esas
constantes dieron:

```
media (μ) = 0.130620
desviación estándar (σ) = 0.308055
```

que coinciden con los valores "canónicos" de MNIST que suelen citarse en la literatura (0.1307 /
0.3081) — una buena señal de que el pipeline de datos está bien construido.

### 2.5 Visualización

El notebook incluye una celda que muestra 12 ejemplos reales del dataset con su etiqueta
correspondiente (`src/figures.py::plot_samples`, guardada en
`results/figures/samples.png`), cumpliendo el requisito de "visualizar al menos 10 ejemplos con su
etiqueta".

---

## 3. Tensores y PyTorch

Un **tensor** es la estructura de datos central de PyTorch: un arreglo multidimensional, similar a
un `ndarray` de NumPy, pero con dos capacidades adicionales que lo hacen apto para deep learning:

- Puede vivir en **CPU o GPU** indistintamente (el mismo código funciona en ambos, solo cambia
  `.to(device)`). En este laboratorio se corrió íntegramente en CPU, con 4 núcleos.
- Si se marca con `requires_grad=True`, PyTorch construye automáticamente un **grafo de
  operaciones** a medida que el tensor participa en cálculos. Ese grafo es lo que permite calcular
  gradientes por **autograd**: al llamar `.backward()`, PyTorch recorre el grafo hacia atrás y
  calcula la derivada de la pérdida respecto a cada parámetro entrenable, sin que el programador
  tenga que derivar nada a mano.

En este laboratorio los tensores relevantes son:

- Un batch de imágenes: `(N, 1, 28, 28)` — N imágenes, 1 canal, 28 filas, 28 columnas.
- El mismo batch aplanado para el MLP: `(N, 784)`.
- La salida de cualquiera de los dos modelos: `(N, 10)` — un vector de **logits** (puntuaciones
  crudas, sin normalizar) por clase, para cada imagen del batch.

---

## 4. Las capas de `torch.nn`, explicadas una por una

Esta sección cubre exactamente las capas que el enunciado pide investigar, con cada parámetro
explicado individualmente, más las capas adicionales que se usaron en la implementación real
(`nn.Linear`, `nn.ReLU`, `nn.Dropout`, `nn.BatchNorm1d`).

### 4.1 `nn.Conv2d` — la capa convolucional

**Qué hace.** Desliza un conjunto de filtros (*kernels*) aprendibles sobre la imagen. Cada filtro es
una pequeña matriz de pesos (por ejemplo 3×3) que se multiplica elemento a elemento con cada ventana
de la imagen y se suma, produciendo un único número por posición. El resultado, repetido en toda la
imagen, es un **mapa de activación**: una nueva "imagen" que responde fuerte donde el filtro
encontró el patrón que aprendió a detectar (un borde, una esquina, una curva).

**Por qué importa.** Es la capa que le da a la CNN su propiedad más importante: **invarianza a
traslaciones**. El mismo filtro se aplica en todas las posiciones de la imagen, así que un patrón
detectado en la esquina superior izquierda se reconoce igual de bien en el centro, sin tener que
aprenderlo dos veces.

**Parámetros del constructor:**

| Parámetro | Qué controla |
|---|---|
| `in_channels` | Número de canales de entrada. En MNIST es 1 (escala de grises); en RGB sería 3. |
| `out_channels` | Número de filtros, es decir, cuántos mapas de activación produce la capa. |
| `kernel_size` | Tamaño de la ventana del filtro (en este laboratorio, siempre 3×3). Kernels chicos apilados suelen funcionar mejor que uno grande, porque agregan no-linealidad entre medio. |
| `stride` | Cuántos píxeles se desplaza el filtro en cada paso. Con `stride=1` (el usado aquí) no se salta ningún píxel; con `stride>1` se submuestrea al mismo tiempo que se convoluciona. |
| `padding` | Cuántas filas/columnas de ceros se agregan alrededor de la imagen antes de convolucionar. Con `padding = kernel_size // 2` (lo que hace `src/models.py`) el tamaño espacial de salida es igual al de entrada — sin padding, cada convolución "come" un borde y la imagen se encoge. |
| `dilation` | Separa los elementos del kernel entre sí, ampliando el campo receptivo sin agregar parámetros. No se usó en este laboratorio (queda en su valor por defecto). |
| `bias` | Si la capa suma un sesgo aprendible además del producto punto. Se suele desactivar cuando la sigue un `BatchNorm2d` (porque el término de sesgo del BatchNorm ya cumple ese rol), aunque en esta implementación se dejó activo por simplicidad. |

**Fórmula del tamaño de salida** (para una dimensión, se aplica igual a alto y ancho):

```
H_out = (H_in + 2·padding − dilation·(kernel_size − 1) − 1) / stride + 1
```

**Conteo de parámetros** de una capa `Conv2d`:

```
parámetros = out_channels × (in_channels × kernel_size × kernel_size) + out_channels
```

Ese último `+ out_channels` es el término de sesgo (un bias por filtro). Importante: esta cantidad
**no depende del tamaño de la imagen** — una `Conv2d(1, 32, 3)` tiene 320 parámetros ya sea que la
imagen de entrada sea de 28×28 o de 2800×2800. Esa independencia del tamaño de entrada es la raíz de
por qué las CNN escalan mejor que las capas densas (ver sección 6).

### 4.2 `nn.MaxPool2d` — submuestreo por máximo

**Qué hace.** Divide el mapa de activación en ventanas (aquí, de 2×2) y se queda con el **valor
máximo** de cada una, reduciendo la resolución espacial a la mitad en cada dimensión.

**Por qué importa.** Tres razones: (1) reduce el costo computacional de las capas siguientes; (2)
**amplía el campo receptivo** de las capas posteriores sin agregar parámetros (ver sección 5); (3) da
cierta invarianza a pequeñas traslaciones — si el trazo se corre un píxel dentro de la ventana, el
máximo probablemente no cambia.

**Parámetros del constructor:** `kernel_size` (tamaño de la ventana, 2×2 en este laboratorio),
`stride` (por defecto igual a `kernel_size`, es decir, ventanas sin solaparse), `padding`,
`ceil_mode` (si redondea hacia arriba el tamaño de salida cuando no divide exacto).

**Detalle clave:** `MaxPool2d` **no tiene parámetros entrenables**. No aprende nada; es una operación
fija.

### 4.3 `nn.AvgPool2d` — submuestreo por promedio

**Qué hace.** Igual que `MaxPool2d`, pero en vez de quedarse con el máximo de la ventana, calcula el
**promedio**. Tampoco tiene parámetros entrenables.

**Parámetros del constructor:** los mismos que `MaxPool2d`, más `count_include_pad` (si los ceros
del padding cuentan en el promedio o se excluyen).

**La intuición habitual vs. lo que se midió.** La intuición típica dice que en imágenes con trazos
claros sobre fondo negro —como MNIST— `MaxPool2d` debería ganar, porque preserva la intensidad del
trazo mientras que `AvgPool2d` la diluye al promediarla con los píxeles de fondo alrededor. **Esa
intuición no se sostuvo en este laboratorio**: en la iteración C3, cambiar de MaxPool a AvgPool
mejoró el F1-macro de validación de 0.9873 a 0.9892. Una lectura plausible es que el promedio
conserva cuánta señal había en toda la región y no solo su pico, lo que da una estimación menos
frágil ante un solo píxel brillante aislado por ruido. Es el ejemplo más claro del laboratorio de por
qué hay que **medir en vez de razonar** cuando se elige un hiperparámetro.

### 4.4 `nn.BatchNorm2d` (y `nn.BatchNorm1d`) — normalización por lotes

**Qué hace.** Para cada canal (en `BatchNorm2d`) o cada feature (en `BatchNorm1d`), normaliza sus
activaciones usando la media y la varianza calculadas **sobre el batch actual**, y luego aplica una
transformación afín aprendible: `y = γ · x_normalizado + β`, donde `γ` (escala) y `β` (desplazamiento)
son dos parámetros por canal que la red aprende durante el entrenamiento.

**Por qué importa.** Mantiene las activaciones en un rango estable capa a capa, lo que **condiciona
mejor** la superficie de pérdida: permite usar tasas de aprendizaje más altas y acelera la
convergencia. También actúa como un regularizador leve, porque las estadísticas de cada batch varían
ligeramente de una pasada a otra, introduciendo ruido.

**Parámetros del constructor:** `num_features` (número de canales o de neuronas de la capa anterior,
debe coincidir exactamente), `eps` (constante pequeña sumada a la varianza para evitar dividir entre
cero), `momentum` (con qué velocidad se actualizan las medias/varianzas móviles usadas en
inferencia), `affine` (si aprende `γ` y `β` o los deja fijos en 1 y 0), `track_running_stats` (si
mantiene un promedio móvil de media/varianza para usarlo en modo evaluación).

**Detalle de comportamiento importante.** BatchNorm se comporta distinto en modo entrenamiento
(`model.train()`, usa las estadísticas del batch actual) que en modo evaluación (`model.eval()`, usa
las estadísticas móviles acumuladas durante todo el entrenamiento). Por eso el bucle de
entrenamiento en `src/train.py` alterna explícitamente entre ambos modos.

**Lo que se midió, sin idealizar.** En este laboratorio, BatchNorm **ayudó al MLP** (+0.0046 de F1 al
agregar `BatchNorm1d` en M6) pero **perjudicó a la CNN** (−0.0011 al agregar `BatchNorm2d` en C4).
No hay un ganador universal — se explica con más detalle en la sección 11.

### 4.5 `nn.Flatten` — el puente entre convolución y clasificador denso

**Qué hace.** Convierte un tensor `(N, C, H, W)` en `(N, C·H·W)`, es decir, aplana todas las
dimensiones excepto la de batch. No tiene parámetros ni realiza ningún cómputo real — solo
reinterpreta cómo están dispuestos los mismos números en memoria.

**Parámetros del constructor:** `start_dim` (por defecto 1, para no aplanar también la dimensión de
batch — aplanarla mezclaría imágenes distintas entre sí, un error grave), `end_dim`.

**Doble papel en este laboratorio.** `Flatten` es la **primera capa del MLP** (ahí es exactamente
donde se pierde la estructura espacial de la imagen — cada píxel pasa a ser una feature suelta sin
relación declarada con sus vecinos), y también es el puente que usa la CNN entre su parte
convolucional y su clasificador denso final.

### 4.6 `nn.CrossEntropyLoss` — la función de pérdida

**Qué hace.** Es la pérdida estándar para clasificación multiclase. Internamente combina dos pasos en
una sola operación numéricamente estable:

1. `log_softmax`: convierte los logits crudos en log-probabilidades.
2. `NLLLoss` (*Negative Log-Likelihood*): toma el negativo del log de la probabilidad asignada a la
   clase correcta.

**Consecuencia práctica que hay que respetar.** El modelo debe entregar **logits crudos**, sin
`Softmax` en la última capa. Aplicar `Softmax` manualmente y luego `CrossEntropyLoss` sería un error
común: la operación se aplicaría dos veces y los gradientes saldrían aplanados (más pequeños de lo
que deberían), enlenteciendo el aprendizaje. Por eso tanto el MLP como la CNN de este laboratorio
terminan en un `nn.Linear(..., 10)` sin ninguna activación después.

**Parámetros del constructor:** `weight` (permite ponderar clases distinto — útil ante desbalances
fuertes; aquí no hizo falta dado el desbalance leve del dataset), `ignore_index` (excluye una
etiqueta del cálculo), `reduction` (`'mean'` por defecto: promedia la pérdida del batch), `label_smoothing`
(suaviza las etiquetas duras 0/1 para reducir el exceso de confianza del modelo).

### 4.7 Capas adicionales usadas en la implementación

Aunque no las pide explícitamente el enunciado, estas capas son parte central de ambos modelos:

- **`nn.Linear(in_features, out_features)`** — la capa "densa" o "totalmente conectada": cada neurona
  de salida se conecta a **todas** las entradas, con un peso independiente por conexión más un sesgo.
  Tiene `in_features × out_features + out_features` parámetros. Es el bloque constructor del MLP y
  también el clasificador final de la CNN.
- **`nn.ReLU(inplace=True)`** — la no-linealidad *Rectified Linear Unit*: `f(x) = max(0, x)`. Sin
  funciones de activación no-lineales entre las capas, apilar `Linear`s sería matemáticamente
  equivalente a una sola capa lineal, sin importar cuántas se apilen. `inplace=True` es una
  optimización de memoria (modifica el tensor en el lugar en vez de crear uno nuevo), sin efecto en
  el resultado.
- **`nn.Dropout(p)`** — durante el entrenamiento, apaga aleatoriamente una fracción `p` de las
  neuronas de esa capa en cada paso (forzando a la red a no depender de ninguna neurona en
  particular), y reescala las restantes para compensar. En evaluación no apaga nada. Es un
  regularizador puro: no tiene efecto si no hay overfitting que corregir.

---

## 5. Campo receptivo

**Definición.** El **campo receptivo** (*receptive field*) de una neurona en una capa profunda es la
región de la **imagen de entrada original** que puede influir en su valor. En una CNN crece de forma
incremental, capa por capa, según esta fórmula:

```
RF_out = RF_in + (k − 1) · jump
jump_out = jump_in · stride
```

donde `k` es el tamaño del kernel de la capa actual y `jump` es el producto de los strides
acumulados hasta ese punto de la red (el "salto" entre posiciones consecutivas del mapa de
activación, medido en píxeles de la imagen original).

**Por qué importa.** Es lo que explica por qué **apilar capas** funciona: cada capa adicional
amplía el contexto que "ve" la red, sin necesidad de agrandar el kernel (que sería mucho más caro en
parámetros). Las primeras capas de una CNN detectan trazos muy locales (unos pocos píxeles); las
capas más profundas, gracias al campo receptivo acumulado, ya "ven" buena parte del dígito completo.

**Cálculo real para la CNN ganadora de este laboratorio (C3: 2 bloques Conv3×3 + MaxPool2×2 cada
uno)**, usando la fórmula implementada en `src/models.py::CNN.receptive_field`:

| Capa | Campo receptivo tras esta capa | `jump` acumulado |
|---|---|---|
| Conv2d bloque 1 (k=3) | 3×3 | 1 |
| Pool2d bloque 1 (2×2) | 4×4 | 2 |
| Conv2d bloque 2 (k=3) | 8×8 | 2 |
| Pool2d bloque 2 (2×2) | 10×10 | 4 |

El campo receptivo final es de **10×10 píxeles**, es decir, aproximadamente el **35.7 % del ancho
total de la imagen** (28×28). Cada bloque convolucional adicional (como en C6, que usa 3 bloques)
ampliaría todavía más ese contexto.

---

## 6. Por qué una CNN necesita menos parámetros que un MLP

Este es uno de los puntos de investigación explícitos del enunciado, y conviene explicarlo con
cuidado porque los resultados reales de este laboratorio matizan la respuesta ingenua.

### 6.1 El argumento estructural

Hay dos mecanismos, y ambos actúan a la vez:

1. **Conectividad local.** Una capa `Linear` conecta *todas* las entradas con *todas* las salidas.
   Una capa `Conv2d` conecta cada neurona de salida solo con una ventana pequeña (k×k) de la entrada,
   no con la imagen completa. En MNIST, eso significa conectarse a 9 píxeles (kernel 3×3) en vez de
   a los 784 completos.
2. **Pesos compartidos.** El mismo filtro se aplica en *todas* las posiciones de la imagen. Detectar
   un borde vertical en la esquina superior izquierda usa exactamente los mismos 9 pesos que
   detectarlo en el centro de la imagen. Una capa `Linear`, en cambio, tiene que aprender el patrón
   por separado para cada posición posible, porque cada conexión tiene su propio peso independiente.

**Ejemplo numérico concreto** (tomado del notebook): una `Conv2d(1, 32, kernel_size=3)` tiene
**320 parámetros** (32 filtros × 9 pesos + 32 sesgos) y produce 32 mapas de activación completos de
28×28. Una capa densa equivalente en número de salidas, `Linear(784, 32)`, necesita **25 120
parámetros** — 78 veces más — y además destruye por completo la relación de vecindad entre píxeles al
tratar cada uno como una feature independiente.

### 6.2 El matiz que muestran los resultados reales — no simplificar de más

El ahorro anterior es real **por capa**, pero **no se traduce automáticamente en que el modelo
completo sea más pequeño**. La evidencia de este mismo laboratorio:

- La **CNN ganadora (C3)** tiene **421 642 parámetros**.
- El **MLP ganador (M6)** tiene **235 914 parámetros**.
- Es decir, **la CNN ganadora usa 1.8× más parámetros que el MLP**, no menos.

¿Por qué? Al desglosar dónde están esos 421 642 parámetros de la CNN:

| Componente | Parámetros | % del total |
|---|---|---|
| Bloques convolucionales | 18 816 | 4.5 % |
| Clasificador denso final | 402 826 | 95.5 % |

Casi todo el costo en parámetros de la CNN **no** está en las convoluciones — está en la capa densa
final (`nn.Linear`) que viene después del `Flatten`, exactamente el mismo tipo de capa que domina al
MLP. Las convoluciones son, en efecto, baratísimas; el clasificador que las sigue no lo es.

La demostración correcta de que la eficiencia en parámetros es real está en otro lado: la iteración
**C1** (la CNN más simple probada, con solo 2 bloques conv de 16/32 canales) alcanza **206 922
parámetros** y **supera al mejor MLP en F1 de validación** (0.9878 contra 0.9824 de M6) usando
**menos** parámetros que él. Ese es el contraejemplo correcto: la arquitectura convolucional *puede*
lograr mejor desempeño con menos parámetros, pero eso depende de cómo se diseñe el clasificador
final, no es una propiedad automática de "usar convoluciones".

### 6.3 La razón de fondo no es solo el conteo de parámetros

Más importante que el ahorro numérico es el **sesgo inductivo**: la CNN incorpora de fábrica la
suposición correcta sobre cómo está organizada una imagen (los píxeles vecinos están relacionados, y
un patrón significa lo mismo esté donde esté). Esa suposición no hay que aprenderla a partir de los
datos — viene incluida en la arquitectura. El MLP, al no tener esa suposición, tiene que *intentar
inferirla* de los ejemplos de entrenamiento, lo cual es más difícil y menos confiable, sin importar
cuántos parámetros tenga disponibles.

---

## 7. Las dos arquitecturas implementadas

Ambas están definidas en `src/models.py` como clases parametrizables por un diccionario de
configuración, lo que permitió recorrer el espacio de búsqueda sin duplicar código.

### 7.1 `MLP`

```python
MLP(hidden_sizes=(128,), dropout=0.0, use_bn=False, in_features=784, n_classes=10)
```

Estructura: `Flatten → [Linear → (BatchNorm1d) → ReLU → (Dropout)] × n → Linear(n_classes)`

| Argumento | Significado |
|---|---|
| `hidden_sizes` | Tupla con el ancho de cada capa oculta. `(128,)` es una sola capa de 128 neuronas; `(256, 128)` son dos capas, 256 y luego 128. |
| `dropout` | Probabilidad de apagar cada neurona oculta durante el entrenamiento. `0.0` lo desactiva. |
| `use_bn` | Si intercala una capa `BatchNorm1d` después de cada `Linear` oculta (antes del `ReLU`). |
| `in_features` | Tamaño del vector de entrada (784 = 28×28, fijo para MNIST). |
| `n_classes` | Número de neuronas de salida (10 dígitos). |

### 7.2 `CNN`

```python
CNN(channels=(16, 32), kernel_size=3, pool="max", use_bn=False,
    dropout=0.0, fc_hidden=128, in_channels=1, image_size=28, n_classes=10)
```

Estructura: `[Conv2d → (BatchNorm2d) → ReLU → Pool(2)] × n → Flatten → Linear(fc_hidden) → ReLU → (Dropout) → Linear(n_classes)`

| Argumento | Significado |
|---|---|
| `channels` | Tupla con el número de filtros de cada bloque convolucional. `(16, 32)` son dos bloques: el primero produce 16 mapas, el segundo 32. Debe tener al menos 2 elementos (requisito del enunciado: mínimo 2 capas convolucionales). |
| `kernel_size` | Tamaño del kernel de todas las convoluciones (3×3 en todo el laboratorio). |
| `pool` | `"max"` o `"avg"` — qué tipo de pooling usar después de cada bloque convolucional. Es el hiperparámetro que se puso a prueba directamente en la iteración C3. |
| `use_bn` | Si intercala `BatchNorm2d` después de cada `Conv2d` (antes del `ReLU`). |
| `dropout` | Probabilidad de Dropout aplicada justo antes de la última capa `Linear` del clasificador. |
| `fc_hidden` | Ancho de la capa densa oculta del clasificador, entre el `Flatten` y la salida final. |
| `in_channels` | Canales de la imagen de entrada (1, escala de grises). |
| `image_size` | Tamaño de la imagen de entrada en píxeles (28). Se usa para calcular el tamaño espacial final tras los poolings, y así el tamaño exacto de la primera capa `Linear` del clasificador. |
| `n_classes` | Número de clases de salida (10). |

Cada bloque usa `padding = kernel_size // 2` (padding "same"), de modo que la convolución en sí
misma no reduce el tamaño espacial — solo lo reduce el `Pool2d(2)` que sigue, a la mitad por bloque.
Por eso, con imágenes de 28×28 y 2 bloques, el tamaño espacial final es 28 → 14 → 7.

---

## 8. Metodología de entrenamiento

Implementada en `src/train.py`.

### 8.1 Función de pérdida y optimizadores

- **Pérdida:** `nn.CrossEntropyLoss()` en ambos modelos (sección 4.6).
- **Optimizadores probados:**
  - **SGD** (*Stochastic Gradient Descent*): actualiza los pesos restando el gradiente escalado por
    la tasa de aprendizaje (`lr`), opcionalmente con `momentum` (que acumula una fracción del
    gradiente anterior para suavizar la trayectoria y acelerar la convergencia en direcciones
    consistentes). Usado como baseline en M1 con `lr=0.01` y sin momentum.
  - **Adam**: mantiene, para cada parámetro, un promedio móvil del gradiente (momento de primer
    orden) y de su magnitud al cuadrado (momento de segundo orden), y usa ambos para adaptar la tasa
    de aprendizaje efectiva de cada parámetro individualmente. En la práctica converge más rápido y
    con menos ajuste manual que SGD puro. Se cambió a Adam desde la iteración M2 en adelante, y fue
    el cambio con **mayor impacto positivo de todo el laboratorio** (+0.0295 de F1 en el MLP).

### 8.2 Métricas de evaluación

Sobre cada conjunto (validación o test) se calculan cuatro métricas, todas en su variante **macro**:

- **Accuracy**: proporción de predicciones correctas sobre el total. Es intuitiva pero puede
  esconder mal desempeño en clases minoritarias si hay desbalance.
- **Precision** (por clase): de todas las veces que el modelo predijo una clase, ¿qué fracción
  eran realmente de esa clase? `TP / (TP + FP)`.
- **Recall** (por clase): de todos los ejemplos que realmente eran de una clase, ¿qué fracción
  detectó el modelo? `TP / (TP + FN)`.
- **F1-score** (por clase): media armónica de precision y recall, `2·(P·R)/(P+R)`. Penaliza más que
  el promedio simple cuando una de las dos es mucho más baja que la otra.

**Promedio macro** significa: se calcula la métrica **por separado para cada una de las 10 clases**
y luego se promedian esas 10 cifras con el **mismo peso cada una**, sin importar cuántos ejemplos
tenga cada clase. Es la elección correcta cuando, como aquí, hay un desbalance leve entre clases: si
se usara el promedio "micro" (equivalente a la accuracy global), el desempeño en el dígito 5 (la
clase más pequeña) podría pasar desapercibido detrás del desempeño en el dígito 1 (la más grande).

### 8.3 Reproducibilidad

Al inicio de cada iteración se fija la misma semilla (**23236**, el carné) en tres generadores de
números aleatorios distintos: `random` (Python estándar), `numpy` y `torch` (`src/train.py::set_seed`).
Esto asegura que la inicialización de los pesos, el orden de barajado de los datos y cualquier otra
fuente de aleatoriedad sean idénticas entre iteraciones, de modo que **la única diferencia entre dos
configuraciones sea el hiperparámetro que cambió** — condición necesaria para poder atribuir
correctamente el efecto de cada cambio.

---

## 9. La búsqueda de hiperparámetros: las 12 iteraciones

Implementada en `src/experiments.py`. Se hicieron **6 iteraciones por arquitectura** (12 en total,
superando el mínimo de 10 que pide el enunciado), siguiendo una búsqueda **sistemática**: se parte de
una configuración baseline y cada iteración cambia **una sola variable** respecto de su iteración
**base** (no necesariamente la fila anterior de la tabla — ver más abajo).

### 9.1 Por qué "una variable a la vez" y qué significa la columna "Base"

Si se cambiaran varios hiperparámetros al mismo tiempo, y el resultado mejorara, no habría forma de
saber cuál de los cambios fue responsable. Cambiando uno a la vez, cada delta de métrica **es
directamente** el efecto atribuible a ese cambio específico.

La búsqueda de este laboratorio es, técnicamente, un **árbol** y no una cadena lineal. En la CNN, las
iteraciones **C3** (cambio de pooling) y **C4** (agregar BatchNorm) son dos **ramas distintas** que
parten ambas de **C2**, precisamente para no mezclar el efecto del pooling con el de la
normalización en una sola medición. Por eso cada configuración en `src/experiments.py` guarda
explícitamente su `parent` (de qué iteración se deriva), y las tablas de resultados incluyen una
columna `Base` que dice contra cuál iteración se debe comparar cada una — comparar contra la fila
anterior de la tabla habría atribuido a C4 el efecto conjunto de "volver a MaxPool y además agregar
BatchNorm", que son dos cambios, no uno.

### 9.2 Las 6 iteraciones del MLP

Baseline M1: una capa oculta de 128 neuronas, SGD con `lr=0.01`, sin regularización, 10 epochs.

| ID | Base | Cambio aislado | Val. accuracy | Val. F1 | Parámetros | Tiempo |
|---|---|---|---|---|---|---|
| M1 | — | Baseline | 94.47 % | 0.9443 | 101 770 | 89.5 s |
| M2 | M1 | Optimizador SGD → Adam (lr=1e-3) | 97.40 % | 0.9737 | 101 770 | 93.6 s |
| M3 | M2 | Ancho de la capa oculta 128 → 256 | 97.85 % | 0.9784 | 203 530 | 111.3 s |
| M4 | M3 | Profundidad: 1 → 2 capas ocultas (256, 128) | 97.72 % | 0.9769 | 235 146 | 140.0 s |
| M5 | M4 | Regularización: + Dropout 0.3 | 97.78 % | 0.9777 | 235 146 | 126.1 s |
| M6 | M5 | Normalización: + BatchNorm1d | **98.25 %** | **0.9824** | 235 914 | 137.8 s |

**Lectura iteración por iteración:**

- **M1 → M2 (SGD → Adam):** el salto más grande de todo el MLP (+2.93 puntos de accuracy). SGD sin
  momentum, con esta tasa de aprendizaje, converge lento; Adam adapta la tasa por parámetro y
  aprovecha mucho mejor las mismas 10 epochs.
- **M2 → M3 (más ancho):** ganancia moderada (+0.45 pp) al duplicar la capacidad de la única capa
  oculta.
- **M3 → M4 (más profundidad):** aquí el resultado **empeoró** ligeramente (−0.13 pp). Agregar una
  segunda capa oculta no ayudó — probablemente porque, sin regularización todavía, la red más
  profunda empieza a sobreajustar un poco más rápido de lo que gana en capacidad representacional.
- **M4 → M5 (+ Dropout 0.3):** mejora pequeña pero positiva (+0.06 pp), consistente con estar
  corrigiendo algo del sobreajuste introducido en el paso anterior.
- **M5 → M6 (+ BatchNorm1d):** el segundo mayor salto del MLP (+0.47 pp), y la mejor configuración
  final de esta arquitectura.

### 9.3 Las 6 iteraciones de la CNN

Baseline C1: 2 bloques convolucionales (16 y 32 canales), MaxPool, Adam `lr=1e-3`, 8 epochs.

| ID | Base | Cambio aislado | Val. accuracy | Val. F1 | Parámetros | Tiempo |
|---|---|---|---|---|---|---|
| C1 | — | Baseline (2 bloques conv, MaxPool) | 98.78 % | 0.9878 | 206 922 | 149.5 s |
| C2 | C1 | Capacidad: canales (16,32) → (32,64) | 98.73 % | 0.9873 | 421 642 | 230.2 s |
| C3 | C2 | Pooling: MaxPool2d → AvgPool2d | **98.92 %** | **0.9892** | 421 642 | 229.7 s |
| C4 | C2 | Normalización: + BatchNorm2d | 98.62 % | 0.9862 | 421 834 | 266.3 s |
| C5 | C4 | Regularización: + Dropout 0.25 antes de la FC final | 98.92 % | 0.9892 | 421 834 | 264.5 s |
| C6 | C5 | Profundidad: 2 → 3 bloques conv (32, 64, 128) | 98.85 % | 0.9886 | 241 994 | 285.6 s |

**Lectura iteración por iteración:**

- **C1 → C2 (más canales):** el resultado **empeoró** levemente (−0.05 pp) pese a **duplicar** los
  parámetros (206 922 → 421 642). Es la evidencia más directa del laboratorio de que más capacidad no
  es automáticamente mejor: la CNN baseline ya capturaba casi toda la señal disponible en el
  problema, y los canales extra solo agregaron parámetros para ajustar, no señal nueva que aprender.
- **C2 → C3 (MaxPool → AvgPool):** la mejora más grande de la rama CNN a partir de C2 (+0.19 pp), y
  contraria a la intuición estándar (sección 4.3).
- **C2 → C4 (+ BatchNorm2d):** rama alternativa desde C2; el resultado **empeoró** (−0.11 pp). A
  diferencia del MLP, en la CNN BatchNorm no ayudó.
- **C4 → C5 (+ Dropout 0.25):** buena mejora (+0.30 pp), recuperando y superando el nivel de C2.
- **C5 → C6 (+1 bloque conv):** mejora leve en accuracy (+0.06 pp) pero el F1 macro bajó ligeramente
  respecto a C5; además, curiosamente, el tercer bloque **redujo** el total de parámetros de 421 834
  a 241 994 (porque el pooling adicional reduce mucho el tamaño espacial que llega a la capa densa
  final, que es donde vive la mayoría de los parámetros — ver sección 6.2).

### 9.4 Selección de la mejor configuración de cada arquitectura

La mejor configuración de cada arquitectura se elige por **F1-macro de validación** (con desempate
por accuracy de validación), nunca mirando el conjunto de test. Ganadores:

- **MLP: M6** — 256→128 neuronas, Adam, BatchNorm1d, Dropout 0.3.
- **CNN: C3** — canales (32, 64), MaxPool → AvgPool, Adam.

Solo después de fijar estas dos configuraciones se re-entrenan (con la misma semilla) y se evalúan
**una única vez** sobre los 10 000 ejemplos de test — nunca antes.

---

## 10. Resultados finales y comparación

Evaluación sobre el conjunto de **test** (10 000 imágenes, tocado una sola vez):

| Arquitectura | Mejor iteración | Parámetros | Accuracy | Precision | Recall | F1 | Tiempo entren. | Inferencia |
|---|---|---|---|---|---|---|---|---|
| MLP | M6 | 235 914 | 98.19 % | 98.19 % | 98.18 % | 98.18 % | 200.6 s | 0.081 ms/img |
| CNN | C3 | 421 642 | **99.09 %** | 99.09 % | 99.07 % | 99.08 % | 295.7 s | 0.327 ms/img |

En número de errores absolutos sobre las 10 000 imágenes de test: el **MLP se equivocó 181 veces**
y la **CNN se equivocó 91 veces** — la CNN comete **menos de la mitad** de errores que el MLP.

**Relación entre cantidad de parámetros y calidad:** la CNN ganadora usa **1.8× más parámetros** que
el MLP y aun así entrena de forma comparable en pocos minutos. La eficiencia en parámetros de las
convoluciones (sección 6) es real pero no domina el resultado final porque el clasificador denso al
final de la CNN concentra el 95.5 % del costo.

**Tiempo:** la CNN tardó **1.47× más** en entrenar (295.7 s vs 200.6 s) y es **4.0× más lenta en
inferencia** por imagen (0.327 ms vs 0.081 ms) — ambas cifras siguen siendo fracciones de
milisegundo en CPU, muy por debajo de cualquier restricción de latencia realista para este problema.

---

## 11. Discusión: lo que confirmó la intuición y lo que la contradijo

Uno de los aprendizajes más importantes de este laboratorio no es solo "la CNN ganó", sino que
**varias intuiciones estándar sobre estas arquitecturas no se sostuvieron al medir**, y vale la pena
señalarlas explícitamente en vez de forzar una narrativa prolija:

1. **`AvgPool2d` superó a `MaxPool2d`** (C3), contrario a la intuición de que el máximo debería
   preservar mejor los trazos sobre fondo negro.
2. **Duplicar los canales convolucionales empeoró el resultado** (C1 → C2), contrario a la intuición
   de que más capacidad siempre ayuda.
3. **BatchNorm ayudó al MLP pero perjudicó a la CNN.** No hay un método de regularización que gane
   en ambas arquitecturas: BatchNorm dio +0.0046 de F1 en el MLP (M6) pero −0.0011 en la CNN (C4);
   Dropout hizo lo contrario en magnitud (más discreto en el MLP, +0.0030 en la CNN vía C5).
4. **La CNN ganadora tiene más parámetros que el MLP**, no menos — el ahorro de las convoluciones es
   real por capa pero no garantiza que el modelo completo sea más chico (sección 6.2).

**Sobre el punto 3, una nota metodológica honesta:** los deltas de F1 en la CNN son del orden de
milésimas (0.001–0.003), una magnitud comparable al ruido esperable entre corridas con distinta
semilla aleatoria. Con una sola corrida por configuración (el presupuesto de cómputo de este
laboratorio no permitía repetir cada una con varias semillas), esas diferencias pequeñas no son
concluyentes de forma aislada — lo que sí es una conclusión sólida es la ausencia de un patrón
consistente entre arquitecturas, que es justamente lo interesante de reportar.

**Sobre overfitting y underfitting**, identificados con dos señales:

- **Underfitting:** aparece en la baseline del MLP (M1, SGD sin momentum), donde tanto la pérdida de
  entrenamiento como la de validación se quedan relativamente altas y bajan lento — el problema ahí
  no era falta de capacidad, sino velocidad de convergencia del optimizador. Se corrigió cambiando a
  Adam, no agrandando la red.
- **Overfitting:** se detecta con la brecha entre pérdida de validación y de entrenamiento al final
  del entrenamiento (`val_loss − train_loss`); una brecha grande y creciente epoch a epoch es la
  firma de que el modelo está memorizando ejemplos particulares en vez de aprender el patrón general.
  En MNIST, con 54 000 ejemplos limpios frente a modelos de unos cientos de miles de parámetros, el
  margen de overfitting observado fue modesto en general — coherente con que la regularización
  (Dropout, BatchNorm) tuviera efectos pequeños.

---

## 12. Respuestas a las preguntas de discusión del enunciado

**¿Qué cambio de hiperparámetro tuvo el mayor impacto positivo en cada arquitectura? ¿Y el mayor
negativo?**
En el MLP, el mayor impacto positivo fue cambiar SGD por Adam (M1→M2, +0.0295 de F1); el mayor
negativo fue agregar una segunda capa oculta (M3→M4, −0.0015). En la CNN, el mayor impacto positivo
fue agregar Dropout antes de la capa final (C4→C5, +0.0030); el mayor negativo fue agregar
BatchNorm2d (C2→C4, −0.0011).

**¿Observaron overfitting o underfitting? ¿Cómo lo identificaron y qué hicieron para mitigarlo?**
Ver sección 11. Underfitting en la baseline del MLP (se corrigió cambiando el optimizador);
overfitting leve y generalizado dado el tamaño del dataset (se atenuó con Dropout y BatchNorm, con
efecto modesto).

**¿La regularización mejoró el desempeño en validación? ¿Cuál método funcionó mejor para cada
arquitectura y por qué?**
Sí, mejoró en ambas arquitecturas cuando se aplicó el método correcto para cada una: BatchNorm1d
para el MLP, Dropout para la CNN. No hubo un método universalmente mejor — ver el punto 3 de la
sección 11.

**Comparando el MLP y la CNN, ¿cuál obtuvo mejor desempeño en test? ¿Cómo se relaciona esa
diferencia con la cantidad de parámetros y con la forma en que cada arquitectura procesa la
información espacial?**
Ganó la CNN (99.09 % vs 98.19 % de accuracy, 91 vs 181 errores). La diferencia **no** se explica por
tener más parámetros — la CNN ganadora tiene más, pero C1 (con menos parámetros que el mejor MLP) ya
lo supera. Se explica porque el MLP aplana la imagen y pierde toda relación de vecindad entre
píxeles, mientras que la CNN preserva la estructura 2D y comparte filtros entre posiciones, lo que le
da "gratis" una invarianza a pequeñas traslaciones que el MLP tendría que aprender a fuerza de más
pesos y más datos.

**¿En qué tipo de errores se equivocan más el MLP y la CNN?**
Ambos modelos concentran sus errores en pares de dígitos que comparten trazos visuales (4/9, 3/5,
7/2), pero el MLP acumula además errores en dígitos escritos con inclinaciones o desplazamientos
poco comunes — precisamente la variabilidad que pierde al aplanar la imagen. Los pocos errores que le
quedan a la CNN corresponden en su mayoría a dígitos genuinamente ambiguos incluso para un lector
humano (matrices de confusión completas en `results/figures/confusion_mlp.png` y
`confusion_cnn.png`).

**Si tuvieran que desplegar un modelo de producción, priorizando exactitud y eficiencia, ¿qué
arquitectura elegirían y por qué?**
Una CNN, reconociendo que aquí sí existe un trade-off real (la CNN ganadora es más exacta pero
también más pesada y más lenta en inferencia que el MLP). La decisión se inclina por la CNN porque
0.90 puntos porcentuales de accuracy equivalen a ~49 % menos errores, y en un sistema real cada error
mal clasificado suele costar una intervención manual mucho más cara que los 0.327 ms de CPU que toma
una inferencia. Si la restricción vinculante fuera estrictamente la memoria, la configuración a
desplegar no sería el MLP sino **C1**: menos parámetros que el mejor MLP y aun así por encima de él
en validación.

---

## 13. Glosario

- **Accuracy** — proporción de predicciones correctas sobre el total de ejemplos evaluados.
- **Autograd** — el sistema de PyTorch que calcula automáticamente gradientes recorriendo hacia
  atrás el grafo de operaciones que produjo un tensor.
- **Backpropagation** — algoritmo que calcula el gradiente de la pérdida respecto a cada peso de la
  red, propagando el error desde la salida hacia las capas anteriores, aplicando la regla de la
  cadena del cálculo diferencial.
- **Batch** — un subconjunto de ejemplos de entrenamiento procesado junto en un solo paso de
  optimización (aquí, 128 imágenes por batch de entrenamiento).
- **BatchNorm** — ver sección 4.4.
- **Bias (sesgo)** — parámetro constante que se suma a la salida de una neurona o filtro, además del
  producto punto con los pesos.
- **Campo receptivo** — ver sección 5.
- **CNN (Convolutional Neural Network)** — red neuronal que usa capas convolucionales para procesar
  datos con estructura espacial, como imágenes.
- **CrossEntropyLoss** — ver sección 4.6.
- **Data leakage (fuga de datos)** — cuando información del conjunto de validación o test se filtra,
  directa o indirectamente, hacia el proceso de entrenamiento o preprocesamiento, inflando de forma
  artificial las métricas.
- **Dropout** — regularización que apaga aleatoriamente neuronas durante el entrenamiento.
- **Epoch** — una pasada completa por todo el conjunto de entrenamiento.
- **F1-score** — media armónica de precision y recall.
- **Flatten** — ver sección 4.5.
- **Gradiente** — el vector de derivadas parciales de la función de pérdida respecto a cada
  parámetro; indica la dirección de mayor crecimiento de la pérdida (el descenso de gradiente se
  mueve en la dirección opuesta).
- **Kernel (o filtro)** — la pequeña matriz de pesos que una capa convolucional desliza sobre la
  entrada.
- **Learning rate (tasa de aprendizaje)** — el tamaño del paso con el que el optimizador actualiza
  los pesos en cada iteración, en la dirección opuesta al gradiente.
- **Logits** — las salidas crudas de la última capa de una red de clasificación, antes de aplicar
  cualquier normalización tipo softmax.
- **Macro (promedio)** — promediar una métrica calculada por separado para cada clase, dándole el
  mismo peso a cada clase sin importar cuántos ejemplos tenga.
- **MLP (Multi-Layer Perceptron)** — red neuronal compuesta enteramente por capas totalmente
  conectadas (`Linear`).
- **Overfitting (sobreajuste)** — cuando un modelo aprende patrones específicos del conjunto de
  entrenamiento (incluyendo su ruido) que no generalizan a datos nuevos; se detecta cuando la pérdida
  de entrenamiento sigue bajando mientras la de validación se estanca o sube.
- **Padding** — filas/columnas de ceros agregadas alrededor de la entrada de una convolución, para
  controlar el tamaño espacial de la salida.
- **Pooling** — operación de submuestreo espacial (max o average) sin parámetros entrenables.
- **Precision** — de las predicciones positivas de una clase, qué fracción era correcta.
- **Recall** — de los ejemplos reales de una clase, qué fracción fue detectada correctamente.
- **Receptive field** — ver "campo receptivo".
- **ReLU (Rectified Linear Unit)** — función de activación no lineal `f(x) = max(0, x)`.
- **Semilla (seed)** — número que inicializa un generador de números pseudoaleatorios, usado para
  hacer reproducible un experimento.
- **Sesgo inductivo** — la suposición estructural que una arquitectura incorpora "de fábrica" sobre
  cómo están organizados los datos (por ejemplo, que los píxeles vecinos de una imagen están
  relacionados).
- **Stride** — el paso, en píxeles, con el que se desplaza un kernel o una ventana de pooling.
- **Tensor** — ver sección 3.
- **Underfitting (subajuste)** — cuando un modelo no logra capturar ni siquiera el patrón del
  conjunto de entrenamiento; ambas pérdidas (entrenamiento y validación) se mantienen altas.

---

## 14. Mapa del repositorio

| Archivo | Qué contiene |
|---|---|
| `src/data.py` | Carga de MNIST, split estratificado train/val, cálculo de media/desviación solo sobre train, `DataLoader`s. |
| `src/models.py` | Clases `MLP` y `CNN` parametrizables, `count_parameters()`, cálculo del campo receptivo. |
| `src/train.py` | Bucle de entrenamiento/evaluación, métricas macro, fijación de semillas, medición de tiempo de inferencia. |
| `src/experiments.py` | Definición de las 12 configuraciones (con su relación `parent`), ejecución reanudable de la búsqueda, selección de la mejor config y evaluación final en test. |
| `src/figures.py` | Generación de todas las figuras (muestras, distribución de clases, curvas de pérdida, matrices de confusión, parámetros vs. accuracy). |
| `src/report_data.py` | Consolidación de `results/*.json` en las tablas usadas por el notebook y el PDF (una sola fuente de verdad, sin transcripción manual). |
| `notebooks/Lab2_CNN_MNIST_IanCumes_23236.ipynb` | El entregable de notebook: 66 celdas, ejecutado con outputs, cubriendo las 6 secciones del enunciado. |
| `report/build_report.py` | Genera el PDF de 3 páginas a partir de `results/*.json`, con verificación automática del número de páginas. |
| `report/Laboratorio2_CNN_IanCumes_23236.pdf` | El entregable de PDF. |
| `results/iterations.json` | Las 12 iteraciones completas: configuración, historial de pérdida por epoch, métricas de validación, parámetros, tiempo. |
| `results/final_test.json` | Evaluación final única sobre test de ambos modelos ganadores, con sus matrices de confusión. |
| `results/data_meta.json` | Metadatos del split y de la normalización (tamaños, media, desviación, conteos por clase). |
| `results/figures/*.png` | Todas las gráficas generadas, en alta resolución. |
| `run_all.sh` | Script que reproduce todo el laboratorio de punta a punta desde cero. |
