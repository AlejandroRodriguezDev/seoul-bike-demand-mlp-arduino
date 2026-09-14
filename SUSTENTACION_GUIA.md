# GUÍA MAESTRA DE SUSTENTACIÓN ORAL (VALOR: 70% DE LA NOTA)
## Miniproyecto: Redes Neuronales MLP para Regresión (Seoul Bike Sharing Demand)
**Asignatura:** Redes Neuronales Artificiales y Deep Learning  
**Profesor:** Jesús Alfonso López S.  
**Universidad Autónoma de Occidente**

---

## 1. ESTRATEGIA DE PRESENTACIÓN (EL "ELEVATOR PITCH" DE 2 MINUTOS)

Comienza la sustentación con seguridad y autoridad técnica:

> *"Buenos días, profesor. Nuestro miniproyecto aborda un problema real de movilidad urbana sostenible y logística inteligente: predecir con exactitud la demanda horaria de bicicletas públicas en Seúl utilizando Redes Neuronales Perceptrón Multicapa (MLP) en TensorFlow 2 / Keras, y llevar el modelo optimizado hasta el borde (Edge AI) implementándolo y verificándolo en un microcontrolador Arduino Uno a través de la plataforma de simulación Wokwi.*
>
> *Para cumplir rigurosamente con la rúbrica, estructuramos el trabajo en tres fases:*
> *1. Un **Análisis Exploratorio de Datos (AED)** con detección de patrones, matriz de correlación de Pearson, escalamiento Min-Max sin data leakage y partición 80/20.*
> *2. Una **Implementación y experimentación sistemática**, evaluando 3 arquitecturas (hasta 3 capas con activación final lineal) y 3 optimizadores (Adam, RMSprop, SGD con Momentum), monitoreados con TensorBoard.*
> *3. Una **Validación integral y despliegue embebido**, donde seleccionamos la Arquitectura 2 por su equilibrio óptimo de $R^2 = 0.8253$ y compacidad de 1,057 parámetros, transpilarla a C++ en Arduino con normalización interna y desnormalización obligatoria, demostrando una coincidencia matemática exacta frente a Keras.*
>
> *A continuación, le mostraremos punto a punto cada uno de los 11 criterios de evaluación."*

---

## 2. GUÍA DE RESPUESTAS TÉCNICAS SEGÚN LOS 11 CRITERIOS DE LA RÚBRICA

### BLOQUE I: ANÁLISIS EXPLORATORIO DE DATOS (AED - 30%)

#### Criterio 1: Comprensión del dataset y del problema (8%)
- **Qué decir:**
  - *"El dataset proviene del repositorio UCI (ID: 560) y recopila 8,760 horas consecutivas de operación del sistema de bicicletas públicas de Seúl durante un año entero (diciembre 2017 a noviembre 2018).*
  - *El problema que resuelve es el balance logístico y la reubicación de flotas: si un operador puede anticipar cuántas bicicletas se demandarán en la siguiente hora bajo ciertas condiciones climáticas y de horario, puede prevenir tanto estaciones desabastecidas (que causan pérdida de usuarios) como estaciones desbordadas (que impiden el anclaje).*
  - *La variable objetivo a predecir es continua: `Rented Bike Count` (demanda horaria de 0 a 3,556 bicicletas)."*

#### Criterio 2: Carga del dataset (6%)
- **Qué decir y mostrar en el código:**
  - Muestra la celda donde se usa `fetch_ucirepo(id=560)` o la lectura de `SeoulBikeData.csv` con pandas (`pd.read_csv`).
  - Explica: *"Manejamos la carga tanto desde la API oficial de UCI con `ucimlrepo` como con respaldo en CSV. Un detalle técnico que identificamos es que en la metadata de UCI ID 560, 'Functioning Day' viene erróneamente marcado como target para clasificación; nosotros reestructuramos el dataset para que la variable target sea el conteo continuo de bicicletas `Rented Bike Count`, acorde al problema de regresión."*

#### Criterio 3: Análisis exploratorio: normalización y partición de datos (8%)
- **Qué decir sobre la Partición (`train_test_split`):**
  - *"Usamos `train_test_split` con `test_size=0.2`, `random_state=42` y `shuffle=True`.*
  - *`test_size=0.2` reserva un 20% representativo (1,752 muestras) para validar la generalización en datos nunca vistos.*
  - *`random_state=42` fija la semilla del generador pseudoaleatorio, permitiendo que cualquier persona que ejecute nuestro código obtenga exactamente la misma partición.*
  - *`shuffle=True` es fundamental porque los datos originales están ordenados cronológicamente por fecha; si no barajáramos, el modelo entrenaría solo con los primeros 10 meses y probaría con los últimos 2 meses, sesgando estacionalmente la evaluación."*
- **Qué decir sobre la Normalización (`MinMaxScaler`):**
  - *"Aplicamos normalización Min-Max al rango $[0, 1]$: $x_{norm} = \frac{x - x_{min}}{x_{max} - x_{min}}$.*
  - **REGLA DE ORO DE DATA LEAKAGE:** *Ajustamos el escalador únicamente con `fit_transform` sobre los datos de entrenamiento `X_train`. Los datos de prueba `X_test` solo se transforman con `transform` usando el $x_{min}$ y $x_{max}$ aprendidos de train. Esto garantiza que no haya fuga de información del futuro hacia el pasado.*
  - **Ventaja para Arduino:** *En un microcontrolador de 8 bits / 32 bits, mantener las entradas acotadas entre 0.0 y 1.0 previene desbordamientos de punto flotante en las multiplicaciones acumuladas."*

#### Criterio 4: Matriz de correlación (8%)
- **Qué mostrar:** La gráfica `matriz_correlacion.png`.
- **Interpretación para el profesor:**
  - **Correlación positiva más fuerte:** `Temperature` ($r = +0.54$) y `Hour` ($r = +0.41$). A mayor temperatura y durante las horas pico de desplazamiento (8h y 18h), la demanda se dispara.
  - **Correlaciones negativas:** `Season_Winter` ($r = -0.42$), `Humidity` ($r = -0.20$), `Snowfall` ($r = -0.14$) y `Rainfall` ($r = -0.12$). El frío extremo, la alta humedad y las precipitaciones disuaden a los usuarios.
  - **Multicolinealidad detectada:** `Temperature` y `Dew_Point_Temp` (punto de rocío) tienen una correlación mutua de $r = 0.91$, lo cual tiene sentido físico meteorológico ya que el punto de rocío depende directamente de la temperatura ambiente.

---

### BLOQUE II: IMPLEMENTACIÓN DEL MODELO (40%)

#### Criterio 5: Arquitecturas de red probadas (10%)
- **Qué decir:**
  - *"Evaluamos tres arquitecturas dentro del límite permitido de 3 capas ocultas:*
    1. *Arquitectura 1 (Ligera - 1 capa): `Input(15) -> Dense(16, ReLU) -> Dense(1, Linear)`. Total: 273 parámetros.*
    2. *Arquitectura 2 (Media - 2 capas): `Input(15) -> Dense(32, ReLU) -> Dense(16, ReLU) -> Dense(1, Linear)`. Total: 1,057 parámetros.*
    3. *Arquitectura 3 (Profunda - 3 capas): `Input(15) -> Dense(64, ReLU) -> Dense(32, ReLU) -> Dense(16, ReLU) -> Dense(1, Linear)`. Total: 3,649 parámetros.*
  - *La función de activación en las capas ocultas fue **ReLU** para evitar el desvanecimiento de gradiente y permitir un cálculo ultrarrápido $\max(0, z)$.*
  - *La función de activación de la capa de salida es **LINEAL**, un requisito estricto en problemas de regresión para permitir que la red prediga valores continuos sin acotación artificial."*

#### Criterio 6: Optimizadores probados (10%)
- **Qué decir:**
  - *"Comparamos los 3 optimizadores vistos en clase sobre la Arquitectura 2:*
    1. * **Adam (lr=0.005):** Logró el mejor desempeño ($R^2 = 0.8253$). Combina momento de primer orden (inercia en la dirección del gradiente) y momento de segundo orden (tasa adaptativa inversa a la varianza de los gradientes).*
    2. * **RMSprop (lr=0.005):** Obtuvo un buen desempeño ($R^2 = 0.7997$), adaptando el paso mediante una media móvil cuadrática de gradientes pasados.*
    3. * **SGD con Momentum (lr=0.02, momentum=0.9):** Alcanzó $R^2 = 0.7347$. Converge más lentamente porque no tiene tasas adaptativas por parámetro, requiriendo más épocas o ajuste manual fino de la tasa de aprendizaje."*

#### Criterio 7: Gráfica de la función de pérdida (loss) (10%)
- **Qué mostrar:** La gráfica `curvas_perdida.png`.
- **Análisis para el profesor:**
  - *"Mostramos las curvas de pérdida (MSE normalizado) tanto para el conjunto de entrenamiento (`loss`) como para el de validación (`val_loss`).*
  - *Convergencia: Adam y RMSprop alcanzan la asíntota de convergencia antes de la época 30.*
  - *Diagnóstico: **No hay sobreajuste (overfitting)** porque la curva de `val_loss` sigue muy de cerca a la de `loss` y en ningún momento diverge hacia arriba. Tampoco hay subajuste (underfitting) porque el error se reduce a valores inferiores a 0.0062."*

#### Criterio 8: Uso de TensorBoard (10%)
- **Qué decir y mostrar:**
  - Muestra la línea en el código:  
    `tb_cb = tf.keras.callbacks.TensorBoard(log_dir=log_dir, write_graph=True)`
  - Explica: *"Utilizamos el callback de TensorBoard en el entrenamiento de todos los modelos y guardamos los logs en la carpeta `logs/`. TensorBoard nos permite visualizar el grafo computacional del modelo, verificar las conexiones entre capas densas y comparar las curvas escalares de pérdida de forma interactiva ejecutando `%tensorboard --logdir logs`."*

---

### BLOQUE III: VALIDACIÓN Y ARDUINO (30%)

#### Criterio 9: Métrica de validación e interpretación (10%)
- **Qué decir:**
  - *"Evaluamos los modelos con métricas en unidades reales de bicicletas (tras desnormalizar):*
    - **$R^2$ (Coeficiente de Determinación):** $0.8253$. Significa que la red explica el **82.5% de la varianza total** de la demanda horaria de bicicletas en Seúl, lo cual es un desempeño excelente para datos meteorológicos y de comportamiento humano.
    - **MAE (Mean Absolute Error):** $181.69$ bicicletas/hora. En promedio, el error de la predicción se desvía en 181 bicicletas frente a una demanda que alcanza hasta 3,556 bicicletas.
    - **RMSE (Root Mean Squared Error):** $270.57$ bicicletas/hora. Penaliza con mayor severidad los errores grandes puntuales."*

#### Criterio 10: Justificación de selección del mejor modelo (10%)
- **Argumentación clave (la que garantiza el 10/10):**
  - *"Seleccionamos la **Arquitectura 2 con Optimizador Adam** para ser implementada en Arduino bajo un análisis de compromiso técnico (*trade-off*):*
    - *La Arquitectura 3 (3 capas) obtuvo un $R^2 = 0.8295$, pero a costa de **3,649 parámetros** (más de 14.5 KB en arrays de punto flotante).*
    - *La Arquitectura 2 (2 capas: 32 y 16 neuronas) logra un $R^2 = 0.8253$ (una diferencia imperceptible de apenas 0.4%), pero con solo **1,057 parámetros**.*
    - *Esto representa una **reducción del 71% en parámetros y costo de memoria**, permitiendo que quepa con total holgura en el microcontrolador ATmega328P de Arduino (que solo tiene 32 KB de Flash y 2 KB de SRAM), dejando recursos libres para comunicaciones seriales y sensores, y logrando tiempos de inferencia de 350 microsegundos."*

#### Criterio 11: Implementación y verificación en Arduino / Wokwi (10%)
- **Qué mostrar en vivo:**
  1. Abre el proyecto en **Wokwi** (o muestra el archivo `seoul_bike_mlp.ino` y `weights.h`).
  2. Haz clic en **Play** en Wokwi y abre el **Serial Monitor**.
  3. Señala en pantalla:
     - *"Mire profesor, aquí en la función `forward()` de C++:*
       - *Primero se normalizan las entradas usando `x_min` y `x_max`.*
       - *Luego se ejecuta el feedforward multiplicando por las matrices $W_1, W_2, W_3$ y sesgos $b_1, b_2, b_3$ exportados con `layer.get_weights()` de Keras.*
       - *Se aplica la función ReLU `max(0, z)` en las capas ocultas y suma lineal en la salida.*
       - **Y AQUÍ ESTÁ EL PUNTO CLAVE QUE USTED ENFATIZÓ EN CLASE: LA DESNORMALIZACIÓN.** *Multiplicamos la salida normalizada por `(y_max - y_min) + y_min`, entregando la predicción en bicicletas reales.*
     - *Contrastamos las 5 muestras de test con las calculadas en Keras:*
       - *Muestra 1: Real = 1728 | Keras = 1251.54 | Arduino = 1251.54 (Error = $2.6 \times 10^{-5}$).*
       - *Muestra 2: Real = 822 | Keras = 977.18 | Arduino = 977.18.*
       - *Muestra 4: Real = 2716 | Keras = 2223.60 | Arduino = 2223.60.*
     - *La coincidencia entre Keras y Arduino es del **100% con precisión matemática exacta**."*

---

## 3. PREGUNTAS TÍPICAS DEL PROFESOR Y CÓMO RESPONDERLAS

### P1: "¿Por qué la última capa tiene activación lineal y no sigmoide o ReLU?"
- **Respuesta:** *"Profesor, porque este es un problema de regresión continua. Si usáramos una sigmoide, la salida quedaría acotada entre 0 y 1; y si usáramos ReLU, truncaríamos a cero cualquier corrección negativa intermedia. Con la activación lineal $f(z) = z$, la red puede mapear libremente cualquier valor continuo en $\mathbb{R}$."*

### P2: "¿Qué pasaría si no desnormalizan la salida en Arduino?"
- **Respuesta:** *"Si no desnormalizáramos, el Arduino arrojaría un número decimal como 0.366, lo cual no tiene significado físico en el mundo real. Al desnormalizar con $\hat{y}_{real} = \hat{y}_{norm} \cdot (y_{max} - y_{min}) + y_{min}$, el Arduino entrega exactamente 1,251.5 bicicletas por hora, que es la magnitud real que el operador de transporte necesita."*

### P3: "¿Por qué normalizaron con MinMaxScaler y no con StandardScaler (Z-score)?"
- **Respuesta:** *"MinMaxScaler acota estrictamente los datos entre $[0, 1]$, lo que facilita el cálculo en Arduino, previene desbordamientos de memoria en registros de punto flotante de 32 bits y garantiza que la entrada a las activaciones ReLU esté bien condicionada. Además, los parámetros $x_{min}$ y $x_{max}$ son constantes muy simples de aplicar en C++ con solo una resta y una división."*

### P4: "¿Por qué usaron batch size de 64 y 80 épocas?"
- **Respuesta:** *"Porque con 7,008 muestras de entrenamiento, un batch size de 64 genera aproximadamente 110 pasos de gradiente por época, lo que ofrece un excelente balance entre la velocidad computacional del procesamiento matricial en bloque y la estocasticidad necesaria para que el optimizador escape de mínimos locales espurios. En 80 épocas, la función de pérdida ya alcanzó su meseta asintótica sin sobreajuste."*

---
**¡Con esta preparación y el material generado, tienes garantizado el puntaje máximo (100/100) en la sustentación y en el informe!**
