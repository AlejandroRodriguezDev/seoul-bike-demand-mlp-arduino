# GUÍA MAESTRA DE SUSTENTACIÓN ORAL (VALOR: 70% DE LA NOTA)
## Miniproyecto: Redes Neuronales MLP para Regresión (Seoul Bike Sharing Demand)
**Asignatura:** Redes Neuronales Artificiales y Deep Learning  
**Profesor:** Gilber Alexis Corrales Gallego  
**Integrantes:**
- Alejandro Rodríguez Cortés (Cód. 2225645)
- Nicolás Mejía Ochoa (Cód. 2205076)  
**Institución:** Universidad Autónoma de Occidente (UAO), Cali, Colombia  

---

## 1. ESTRATEGIA DE PRESENTACIÓN (EL "ELEVATOR PITCH" DE 2 MINUTOS)

Comienza la sustentación con máxima seguridad, autoridad técnica y fluidez:

> *"Buenos días, profesor Gilber. Nuestro miniproyecto aborda un problema real de movilidad urbana sostenible y optimización logística: predecir con exactitud la demanda horaria de bicicletas públicas en la ciudad de Seúl (*Seoul Bike Sharing Demand*) utilizando Redes Neuronales Perceptrón Multicapa (MLP) en TensorFlow 2 / Keras, y llevar el modelo optimizado hasta el borde (*Edge AI*) implementándolo y verificándolo en un microcontrolador Arduino simulado en Wokwi.*
>
> *Para cumplir con los 11 criterios de su rúbrica de evaluación, estructuramos el proyecto en tres fases:*
> *1. **Análisis Exploratorio de Datos (AED):** Comprensión a fondo de las 14 variables originales de Seúl, matriz de correlación de Pearson, tratamiento de variables temporales/categóricas, escalamiento Min-Max sin data leakage y partición 80/20.*
> *2. **Implementación y experimentación sistemática:** Evaluación de 3 arquitecturas MLP (hasta 3 capas con activación final estrictamente lineal) y 3 optimizadores (Adam, RMSprop, SGD con Momentum), monitoreados mediante TensorBoard.*
> *3. **Validación y despliegue en Arduino:** Selección justificada de la Arquitectura 2 (1,057 parámetros y $R^2 = 0.8253$), transpilación a C++ nativo sin librerías externas, normalización de entradas, **desnormalización obligatoria de la salida** y demostración de coincidencia matemática exacta frente a Keras.*
>
> *A continuación, le presentamos detalladamente cada bloque de la rúbrica."*

---

## 2. GUÍA DE RESPUESTAS TÉCNICAS SEGÚN LOS 11 CRITERIOS DE LA RÚBRICA

### BLOQUE I: ANÁLISIS EXPLORATORIO DE DATOS (AED - 30%)

#### Criterio 1: Comprensión del dataset y del problema (8%)
- **Origen y contexto geográfico:**
  - *"El dataset proviene del repositorio UCI Machine Learning Repository (ID: 560) y corresponde al sistema público de alquiler de bicicletas de la ciudad de **Seúl, Corea del Sur** (conocido localmente como 'Ddareungi' o Seoul Bike)."*
  - *"Registra **8,760 horas consecutivas de operación**, que corresponden exactamente a las 24 horas de los 365 días de un año calendario completo (del 1 de diciembre de 2017 al 30 de noviembre de 2018)."*
- **Problema real de ingeniería que resuelve:**
  - *"Resuelve el desafío crítico de la **logística de reubicación y balance dinámico de flotas** en sistemas de transporte compartido.*
  - *Si un operador no puede anticipar la demanda de la siguiente hora, se generan dos fallas graves de servicio:*
    1. ***Estaciones desabastecidas (vacías):*** *Los usuarios que llegan no encuentran bicicleta disponible, generando pérdida económica, insatisfacción y migración hacia transporte fósil contaminante.*
    2. ***Estaciones desbordadas (saturadas):*** *Los usuarios no encuentran anclajes libres para devolver la bicicleta, obligándolos a buscar otra estación lejana.*
  - *Predecir la demanda horaria en función del clima y la hora permite a la empresa despachar camiones de redistribución de forma proactiva y eficiente."*
- **Variable objetivo (Target):**
  - *"La variable a predecir es continua: `Rented Bike Count` (número de bicicletas alquiladas en esa hora). Su valor oscila entre 0 y un máximo registrado de 3,556 bicicletas/hora, con una media anual de 704.6 bicicletas."*

---

#### Criterio 2: Carga, estructura y tratamiento de las variables (6%)
- **¿Por qué se verifican tipos de datos y valores nulos?**
  - *"Una red neuronal es una calculadora de álgebra lineal. Si una columna tiene datos vacíos (`NaN`), cualquier multiplicación por `NaN` corrompe los pesos y hace que el gradiente diverja. Al ejecutar `df.isnull().sum()`, verificamos que el dataset tiene **cero valores nulos** en sus 8,760 filas."*
- **Estructura original (14 variables):**
  1. *Temporales:* `Date` (DD/MM/YYYY), `Hour` (0-23h), `Seasons` (4 estaciones), `Holiday` (festivo/no), `Functioning Day` (operativo/mantenimiento).
  2. *Meteorológicas continuas:* `Temperature` ($-17.8$ a $+39.4$ °C), `Humidity` ($0-98\%$), `Wind speed` ($0-7.4$ m/s), `Visibility` ($27-2000$, hasta 20 km), `Dew point temperature` ($-30.5$ a $+27.2$ °C), `Solar Radiation` ($0-3.52$ MJ/m²).
  3. *Precipitaciones:* `Rainfall` ($0-35$ mm) y `Snowfall` ($0-8.8$ cm).
- **Tratamiento y transformación a las 15 entradas de la red:**
  - **¿Por qué descartamos `Date`?** *"La fecha en texto ('01/12/2017') tiene alta cardinalidad y provocaría memorización espuria. Toda la información útil ya está extraída en `Hour`, `Seasons` y `Holiday`. Además, los eventos culturales y días especiales quedan perfectamente cubiertos por la variable de festivos (`Holiday`)."*
  - **Binarización:** `Holiday` se convirtió a $1$ (festivo) y $0$ (laboral). `Functioning Day` a $1$ (operando) y $0$ (cerrado).
  - **One-Hot Encoding para `Seasons`:** *"No usamos números 1, 2, 3, 4 porque la red asumiría un orden falso (que el invierno vale el cuádruple que la primavera). Creamos 4 columnas binarias (`Season_Spring`, `Season_Summer`, `Season_Autumn`, `Season_Winter`), donde cada estación tiene su propio canal con valor 1 ó 0."*
  - **Total de entradas numéricas resultantes:** **15 entradas**.

---

#### Criterio 3: Análisis exploratorio: normalización y partición de datos (8%)
- **Partición (`train_test_split`):**
  - *"Aplicamos `train_test_split(X, y, test_size=0.2, random_state=42, shuffle=True)`.*
  - *`test_size=0.2`: Separa un 20% (1,752 registros) como conjunto de prueba ciego nunca visto durante el ajuste, dejando el 80% (7,008 registros) para entrenar.*
  - *`random_state=42`: Fija la semilla pseudoaleatoria garantizando reproducibilidad experimental exacta para cualquier persona que ejecute el código.*
  - *`shuffle=True`: **Punto crítico.** Los datos originales vienen en orden cronológico estricto (de diciembre a noviembre). Si no barajáramos aleatoriamente, el modelo entrenaría solo con los primeros 10 meses y evaluaría exclusivamente con los últimos dos meses de invierno, sufriendo un sesgo estacional fatal."*
- **Normalización Min-Max ($[0, 1]$):**
  - *"Fórmula: $x_{norm} = \frac{x - x_{min}}{x_{max} - x_{min}}$ aplicada a las 15 entradas y a la salida.*
  - **¿Por qué normalizar?** *Si no normalizáramos, la variable Visibilidad (hasta 2000) dominaría a la Radiación Solar (hasta 3.5) solo por magnitud numérica. Al normalizar, todas compiten en igualdad de condiciones entre 0 y 1.*
  - **REGLA DE ORO DE DATA LEAKAGE (Fuga de Información):** *Ajustamos el escalador únicamente con `fit_transform` sobre los datos de entrenamiento (`X_train`). Los datos de prueba (`X_test`) solo se procesan con `transform` usando los valores mínimos y máximos aprendidos en entrenamiento. Nunca se toca el conjunto de prueba para definir la escala.*
  - **Ventaja para Arduino:** *Mantener los datos acotados entre 0.0 y 1.0 previene desbordamientos de punto flotante de 32 bits (`float` en microcontrolador ATmega) y asegura que las activaciones ReLU no se saturen."*

---

#### Criterio 4: Matriz de correlación y análisis profundo de variables (8%)
- **Qué mostrar:** La gráfica [`matriz_correlacion.png`](file:///c:/Users/aleja/OneDrive/Escritorio/Redes%20Neuronales/proyecto_seoul_bike/informe/matriz_correlacion.png).
- **Línea de código exacta:**
  ```python
  df_corr = pd.concat([y, X], axis=1).corr()
  ```
  *Junta la variable target (`y`) con las 15 entradas (`X`) y `.corr()` calcula la correlación de Pearson para todas las parejas.*
- **¿Qué es la correlación de Pearson y cómo se comparan 'horas' con 'bicicletas'?**
  - *"Es una medida adimensional entre $-1$ y $+1$ que evalúa la covarianza estandarizada. No compara unidades físicas (minutos vs pedales), sino **simultaneidad de variación**: cuando la hora del día se mueve hacia la jornada laboral activa (8h y 18h), la cantidad de bicicletas se eleva simultáneamente, dando un coeficiente positivo de $r = +0.41$."*
- **Tabla de correlaciones con la demanda:**

| Variable | Correlación ($r$) | Comportamiento e Impacto Físico |
|---|:---:|---|
| `Temperature` | **$+0.54$** | **Predictor positivo dominante.** Clima agradable estimula masivamente el transporte en bicicleta. |
| `Hour` | **$+0.41$** | **Predictor circadiano.** Patrón bimodal: picos a las 8:00 AM (entrada oficina) y 18:00 PM (salida laboral). |
| `Dew_Point_Temp` | **$+0.38$** | Sigue la tendencia térmica veraniega. |
| `Season_Summer` | **$+0.30$** | Temporada con mayor aprovechamiento de horas de luz. |
| `Solar_Radiation` | **$+0.26$** | Días despejados y soleados favorecen el pedaleo. |
| `Functioning_Binary` | **$+0.20$** | Filtro determinante: si el sistema no opera (0), la demanda es forzosamente 0. |
| `Visibility` | **$+0.20$** | Buena visibilidad aumenta la seguridad percibida. |
| `Wind_Speed` | **$+0.12$** | Vientos moderados urbanos no frenan a los usuarios. |
| `Holiday_Binary` | **$-0.05$** | Correlación lineal cercana a cero frente al total anual. |
| `Rainfall` | **$-0.12$** | La lluvia disuade de inmediato por riesgo de mojado y caídas. |
| `Snowfall` | **$-0.14$** | La nieve congela el asfalto e inhabilita las ciclorrutas. |
| `Humidity` | **$-0.20$** | Humedad alta produce incomodidad térmica y sofocación. |
| `Season_Winter` | **$-0.42$** | **Predictor negativo dominante.** Heladas invernales de hasta $-17.8$ °C desploman el uso. |

- **ANÁLISIS PROFUNDO DE DECISIONES SOBRE VARIABLES:**
  
  > *"Profesor Gilber, tomamos tres decisiones críticas de ingeniería:*
  >
  > *1. **¿Por qué dejamos `Holiday_Binary` si su correlación es de apenas $r = -0.05$?**  
  > Porque los festivos son menos del 5% del año y Pearson diluye su efecto en el promedio global. Pero en esos días puntuales, el patrón bimodal de oficina (8h y 18h) se destruye y la demanda pasa a la tarde. La red neuronal aprovecha esa entrada binaria para ajustar los sesgos en festivos.*
  >
  > *2. **El engaño de Pearson en lluvia (`Rainfall`, $-0.12$) y nieve (`Snowfall`, $-0.14$):**  
  > Aparentan correlación baja porque en Seúl **no llueve ni nieva el 92% de las horas del año** (casi todo son ceros). Pero cuando llueve 5 mm, la demanda en la calle colapsa a cero. Una regresión lineal simple falla aquí, pero nuestra **red neuronal MLP con activaciones ReLU detecta el umbral (si lluvia > 0 $\rightarrow$ resta fuertemente la salida)**. Por eso era obligatorio mantenerlas.*
  >
  > *3. **Multicolinealidad: Temperatura vs Punto de Rocío ($r = 0.91$):**  
  > En estadística clásica (OLS) se eliminaría una por colinealidad. **En redes neuronales NO se elimina porque:**  
  > a) Las capas densas distribuyen los pesos sin matrices singulares.  
  > b) **Fisiología humana y sensación térmica:** La combinación no lineal de Temperatura, Humedad y Punto de Rocío le permite a la red aprender internamente el **índice de calor**. Dos días a 28 °C se sienten completamente distintos si el punto de rocío es bajo (aire fresco y seco) vs alto (bochorno insoportable). Eliminar el punto de rocío le quitaría esa discriminación climática a la red."*

---

### BLOQUE II: IMPLEMENTACIÓN DEL MODELO (40%)

#### Criterio 5: Arquitecturas de red probadas y anatomía de parámetros (10%)
- **Qué decir:**
  - *"Evaluamos 3 arquitecturas dentro del límite permitido de 3 capas ocultas:*
    1. ***Arquitectura 1 (Ligera - 1 capa):*** `Input(15) -> Dense(16, ReLU) -> Dense(1, Linear)`. Total: **273 parámetros** ($R^2 = 0.7295$).
    2. ***Arquitectura 2 (Media - 2 capas):*** `Input(15) -> Dense(32, ReLU) -> Dense(16, ReLU) -> Dense(1, Linear)`. Total: **1,057 parámetros** ($R^2 = 0.8253$).
    3. ***Arquitectura 3 (Profunda - 3 capas):*** `Input(15) -> Dense(64, ReLU) -> Dense(32, ReLU) -> Dense(16, ReLU) -> Dense(1, Linear)`. Total: **3,649 parámetros** ($R^2 = 0.8295$).
  - *La función de activación oculta fue **ReLU** $f(z) = \max(0, z)$ para evitar el desvanecimiento de gradiente y garantizar cálculo ultrarrápido en microcontroladores.*
  - *La activación de salida fue **LINEAL** $f(z) = z$, porque es un problema de regresión continua sin límites artificiales."*
- **¿De dónde salen exactamente los 273 parámetros de la Arquitectura 1?**
  - Capa Oculta 1: 15 entradas conectadas a 16 neuronas = $15 \times 16 = 240$ pesos ($W$). Cada neurona tiene 1 sesgo ($b$) = $+16$ bias. Subtotal = $256$.
  - Capa de Salida: 16 neuronas conectadas a 1 neurona = $16 \times 1 = 16$ pesos ($W$) $+ 1$ bias. Subtotal = $17$.
  - **Total:** $256 + 17 = 273$ parámetros.
- **¿De dónde salen los 1,057 parámetros de la Arquitectura 2?**
  - Capa 1: $15 \times 32 + 32 = 512$
  - Capa 2: $32 \times 16 + 16 = 528$
  - Capa de Salida: $16 \times 1 + 1 = 17$
  - **Total:** $512 + 528 + 17 = 1,057$ parámetros.

#### Criterio 6: Optimizadores probados (10%)
- **¿Qué es un optimizador?**  
  *"Es el algoritmo matemático que calcula hacia dónde y con qué magnitud actualizar los pesos de la red durante el backpropagation para minimizar la función de costo."*
- **Resultados sobre la Arquitectura 2:**
  1. ***Adam (lr=0.005) ⭐:*** Ganador absoluto ($R^2 = 0.8253$). Combina momento de primer orden (inercia del gradiente) y momento de segundo orden (tasa adaptativa inversa a la varianza), acelerando en valles planos y frenando en curvas pronunciadas.
  2. ***RMSprop (lr=0.005):*** Buen desempeño ($R^2 = 0.7997$), divide el gradiente entre la raíz del promedio móvil de sus cuadrados.
  3. ***SGD con Momentum (lr=0.02, momentum=0.9):*** Desempeño inferior ($R^2 = 0.7347$), al usar una tasa fija para todos los parámetros desciende lentamente y requiere cientos de épocas más.

#### Criterio 7: Gráfica de la función de pérdida (loss) (10%)
- **Qué mostrar:** La gráfica [`curvas_perdida.png`](file:///c:/Users/aleja/OneDrive/Escritorio/Redes%20Neuronales/proyecto_seoul_bike/informe/curvas_perdida.png).
- **Diagnóstico técnico de convergencia:**
  - *"Monitoreamos el MSE normalizado de entrenamiento (`loss`) vs validación (`val_loss`).*
  - *Adam y RMSprop alcanzan su asíntota estable antes de la época 30.*
  - ***Ausencia total de Sobreajuste (Overfitting):*** *La curva de validación acompaña en paralelo a la de entrenamiento sin despegarse jamás hacia arriba.*
  - ***Ausencia de Subajuste (Underfitting):*** *El error cuadrático medio cae por debajo de 0.0062, confirmando que la red aprendió la complejidad del dataset."*

#### Criterio 8: Uso e integración con TensorBoard (10%)
- **Qué mostrar:**
  - Enseña la línea: `TensorBoard(log_dir=log_dir, write_graph=True)`.
  - Muestra la carpeta `logs/` con los registros exportados.
  - Abre la interfaz (`localhost:6006`): pestaña **Graphs** con el diagrama de flujo de tensores y pestaña **Scalars** con las curvas interactivas.

---

### BLOQUE III: VALIDACIÓN Y DESPLIEGUE EN ARDUINO (30%)

#### Criterio 9: Métricas de validación e interpretación (10%)
- **Métricas en escala real (tras desnormalizar):**
  - **$R^2 = 0.8253$:** El modelo explica el **82.53% de la varianza total** de la demanda de bicicletas en Seúl sobre datos de prueba independientes (1,752 horas no vistas).
  - **MAE = $174.93$ bicicletas/hora:** En promedio, la red se desvía en 174 bicicletas respecto a la demanda real.
  - **RMSE = $269.82$ bicicletas/hora:** Mide la dispersión cuadrática penalizando desviaciones atípicas.

#### Criterio 10: Justificación de selección del mejor modelo (10%)
- **El argumento de ingeniería (Trade-off):**
  - *"Seleccionamos la **Arquitectura 2 con Optimizador Adam** bajo un balance riguroso entre precisión y costo de memoria para sistemas embebidos:*
    - *La Arquitectura 3 (3 capas) alcanza un $R^2 = 0.8295$, pero a costa de **3,649 parámetros** (más de 14.6 KB de Flash).*
    - *La Arquitectura 2 (2 capas) alcanza un $R^2 = 0.8253$ (diferencia insignificante de apenas 0.42%), pero con solo **1,057 parámetros**.*
    - *Esto representa una **reducción del 71.0% en memoria y complejidad de cómputo**.*
    - *Permite que el modelo ocupe únicamente 4.2 KB de Flash en el Arduino (dejando más del 85% libre para periféricos y comunicaciones) con un tiempo de inferencia de solo **350 microsegundos**."*

#### Criterio 11: Implementación y verificación en Arduino / Wokwi (10%)
- **La relación entre Python (`compare_keras_arduino.py`) y Wokwi (`seoul_bike_mlp.ino`):**
  - *"En Python primero construimos una emulación matemática de la función `forward()` para validar que el producto punto matricial, las activaciones ReLU y la **desnormalización obligatoria** coincidieran al 100% con `model.predict()` de Keras.*
  - *Luego llevamos esa misma lógica en C++ nativo (sin librerías pesadas ni TensorFlow Lite) a Arduino en Wokwi.*
  - *Al darle Play en Wokwi, el Serial Monitor evalúa las mismas 5 muestras de prueba con coincidencia matemática exacta frente a Keras ($error < 10^{-4}$):*
    - Muestra 1: Real = 1728.0 | Keras = 1251.54 | Arduino = 1251.54.
    - Muestra 2: Real = 822.0 | Keras = 977.18 | Arduino = 977.18.
    - Muestra 4: Real = 2716.0 | Keras = 2223.60 | Arduino = 2223.60."*

---

## 3. BANCO ANTI-CORCHADAS: PREGUNTAS TÍPICAS DEL PROFESOR GILBER

### P1: "Frente a 822 bicis reales en la Muestra 2, el modelo predice 977. ¿Ese error de 155 bicicletas no es muy grande?"
- **Respuesta contundente:**
  > *"No, profesor. La demanda en Seúl va de 0 a 3,556 bicicletas por hora. Un error de 155 bicicletas representa apenas el **4.3% del rango total** del problema y está por debajo de nuestro MAE promedio de 174.9 bicicletas.  
  > Además, no estamos modelando un circuito electrónico determinista con leyes de Kirchhoff, sino el **comportamiento libre de 10 millones de personas**: en dos días con el mismo clima exacto, la demanda puede variar en 200 personas por un partido de fútbol, una obra vial o un concierto.  
  > Si nuestro modelo diera 822.0 exacto sobre datos de prueba ciegos, tendría un **sobreajuste (overfitting)** fatal. Predecir 977 confirma que la red aprendió la tendencia general y es perfectamente útil para la logística: el operador sabe que se demandarán cerca de 1,000 bicicletas y envía un camión a abastecer la estación antes de que se quede vacía."*

### P2: "¿Qué es Keras? ¿Dónde se guarda el modelo? ¿Es esto una sola neurona?"
- **Respuesta:**
  > *"Keras es la API de alto nivel de TensorFlow para construir y entrenar redes neuronales profundas. El modelo entrenado se guarda en disco en el archivo `models/mejor_modelo_seoul_bike.keras`, que almacena la arquitectura, los 1,057 pesos entrenados y la configuración del optimizador.  
  > No es una sola neurona: es un Perceptrón Multicapa (MLP) con **49 neuronas artificiales interconectadas** (32 en la primera capa oculta, 16 en la segunda y 1 neurona en la salida).  
  > Sabemos que está bien entrenada porque su $R^2$ es de 0.8253 en datos de prueba no vistos, sus curvas de pérdida no presentan divergencia y su inferencia en Arduino coincide hasta el cuarto decimal con Keras."*

### P3: "¿Por qué la última capa tiene activación lineal y no ReLU o Sigmoide?"
- **Respuesta:**
  > *"Porque es un problema de regresión continua. Si pusiéramos una función Sigmoide, la red quedaría confinada entre 0 y 1; y si usáramos ReLU, truncaríamos a cero cualquier cálculo negativo intermedio antes de desnormalizar. La función lineal $f(z) = z$ permite a la neurona proyectar en toda la recta numérica real $\mathbb{R}$."*

### P4: "¿Qué pasaría si no desnormalizan la salida en el Arduino?"
- **Respuesta:**
  > *"El Arduino arrojaría un número adimensional como `0.3519`, lo cual carece de sentido físico para el operador de transporte. Al aplicar la desnormalización matemática $\hat{y}_{real} = \hat{y}_{norm} \cdot (y_{max} - y_{min}) + y_{min}$, el Arduino entrega exactamente `1,251.5 bicicletas/hora`, que es la magnitud real requerida para la toma de decisiones."*

### P5: "¿Por qué usaron MinMaxScaler y no StandardScaler (Z-score)?"
- **Respuesta:**
  > *"MinMaxScaler acota rígidamente las entradas en el intervalo $[0, 1]$. En microcontroladores embebidos como el ATmega de Arduino, mantener los números entre 0 y 1 previene desbordamientos de registros de punto flotante de 32 bits y asegura que las sumas ponderadas no saturen. Además, en C++ solo requiere dos constantes ($x_{min}$ y $x_{max}$) y una resta y división simples, mientras que Z-score requeriría medias y desviaciones estándar más complejas."*

### P6: "¿Por qué usaron batch_size = 64 y epochs = 80?"
- **Respuesta:**
  > *"Con 7,008 muestras de entrenamiento, un batch size de 64 genera aproximadamente 110 pasos de gradiente por época, equilibrando la velocidad de vectorización matricial en la GPU/CPU con el ruido estocástico suficiente para escapar de mínimos locales. En 80 épocas, observamos que las curvas de pérdida ya se estabilizaron en su meseta asintótica sin sobreajustarse."*
