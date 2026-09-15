# GUÍA MAESTRA DE SUSTENTACIÓN ORAL (VALOR: 70% DE LA NOTA)
## Miniproyecto: Redes Neuronales MLP para Regresión (Seoul Bike Sharing Demand)
**Asignatura:** Redes Neuronales Artificiales y Deep Learning  
**Profesor:** Jesús Alfonso López S.  
**Integrantes:**
- Alejandro Rodríguez Cortés (Cód. 2225645)
- Nicolás Mejía Ochoa (Cód. 2205076)  
**Institución:** Universidad Autónoma de Occidente (UAO), Cali, Colombia  

---

## 1. ESTRATEGIA DE PRESENTACIÓN (EL "ELEVATOR PITCH" DE 2 MINUTOS)

Comienza la sustentación con máxima seguridad, autoridad técnica y fluidez:

> *"Buenos días, profesor. Nuestro miniproyecto aborda un problema real de movilidad urbana sostenible y optimización logística: predecir con exactitud la demanda horaria de bicicletas públicas en la ciudad de Seúl (*Seoul Bike Sharing Demand*) utilizando Redes Neuronales Perceptrón Multicapa (MLP) en TensorFlow 2 / Keras, y llevar el modelo optimizado hasta el borde (*Edge AI*) implementándolo y verificándolo en un microcontrolador Arduino Uno simulado en Wokwi.*
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
- **Qué mostrar en el código:**
  - Muestra la lectura con pandas (`pd.read_csv('SeoulBikeData.csv')` o `fetch_ucirepo(id=560)`).
  - Explica la corrección del target: *"En los metadatos de UCI ID 560, 'Functioning Day' viene erróneamente sugerido como etiqueta de clasificación; nosotros corregimos esto para modelar `Rented Bike Count` como un problema riguroso de regresión no lineal continua."*
- **Estructura y radiografía completa de las variables originales:**
  - *"El dataset cuenta originalmente con 14 columnas que clasificamos en tres naturalezas:*
    1. ***Temporales y calendarias:***
       - `Date` (DD/MM/YYYY): Fecha del registro.
       - `Hour` (0 a 23 h): Hora del día.
       - `Seasons` (Winter, Spring, Summer, Autumn): Estación meteorológica.
       - `Holiday` ('Holiday' vs 'No Holiday'): Si el día es festivo o laborable.
       - `Functioning Day` ('Yes' vs 'No'): Si el sistema estuvo operativo o cerrado por mantenimiento.
    2. ***Meteorológicas continuas:***
       - `Temperature` (°C): Temperatura del aire (de $-17.8$ a $+39.4$ °C).
       - `Humidity` (%): Humedad relativa del aire (0 a 98%).
       - `Wind speed` (m/s): Velocidad del viento (0 a 7.4 m/s).
       - `Visibility` (10m): Visibilidad horizontal (27 a 2,000 unidades, es decir, hasta 20 km).
       - `Dew point temperature` (°C): Temperatura del punto de rocío ($-30.5$ a $+27.2$ °C).
       - `Solar Radiation` (MJ/m²): Radiación solar incidente (0 a 3.52 MJ/m²).
    3. ***Precipitaciones:***
       - `Rainfall` (mm): Lluvia acumulada en la hora (0 a 35 mm).
       - `Snowfall` (cm): Nieve acumulada en la hora (0 a 8.8 cm).*
- **¿Cómo se trataron las variables para la red neuronal? (15 variables de entrada):**
  - **Variable eliminada:** `Date`. Se descartó de las entradas matriciales numéricas porque es una cadena cronológica de alta cardinalidad no generalizable; toda su información temporal útil ya está capturada matemáticamente por `Hour`, `Seasons` y `Holiday`.
  - **Binarización:** `Holiday` se transformó en `Holiday_Binary` (1 si es festivo, 0 si no). `Functioning Day` se transformó en `Functioning_Binary` (1 si operó, 0 si estuvo inactivo).
  - **One-Hot Encoding:** `Seasons` tenía 4 categorías cualitativas nominales. Se convirtió en 4 columnas numéricas binarias independientes (`Season_Spring`, `Season_Summer`, `Season_Autumn`, `Season_Winter`).
  - **Total de entradas de la red:** $15$ neuronas de entrada.

---

#### Criterio 3: Análisis exploratorio: normalización y partición de datos (8%)
- **Partición (`train_test_split`):**
  - *"Aplicamos `train_test_split(X, y, test_size=0.2, random_state=42, shuffle=True)`.*
  - *`test_size=0.2`: Separa un 20% (1,752 registros) como conjunto de prueba ciego nunca visto durante el ajuste, dejando el 80% (7,008 registros) para entrenar.*
  - *`random_state=42`: Fija la semilla pseudoaleatoria garantizando reproducibilidad experimental exacta para cualquier persona que ejecute el código.*
  - *`shuffle=True`: **Punto crítico.** Los datos originales vienen en orden cronológico estricto (de diciembre a noviembre). Si no barajáramos aleatoriamente, el modelo entrenaría solo con los primeros 10 meses y evaluaría exclusivamente con los últimos dos meses de invierno, sufriendo un sesgo estacional fatal."*
- **Normalización Min-Max ($[0, 1]$):**
  - *"Fórmula: $x_{norm} = \frac{x - x_{min}}{x_{max} - x_{min}}$ aplicada a todas las variables de entrada y a la variable objetivo.*
  - **REGLA DE ORO DE DATA LEAKAGE (Fuga de Información):** *Ajustamos el escalador únicamente con `fit_transform` sobre los datos de entrenamiento (`X_train`). Los datos de prueba (`X_test`) solo se procesan con `transform` usando los valores mínimos y máximos aprendidos en entrenamiento. Nunca se toca el conjunto de prueba para definir la escala.*
  - **Ventaja para Arduino:** *Mantener los datos acotados entre 0.0 y 1.0 previene desbordamientos de punto flotante de 32 bits (`float` en microcontrolador ATmega328P) y asegura que las activaciones ReLU no saturen."*

---

#### Criterio 4: Matriz de correlación y análisis profundo de variables (8%)
- **Qué mostrar:** La gráfica [`matriz_correlacion.png`](file:///c:/Users/aleja/OneDrive/Escritorio/Redes%20Neuronales/proyecto_seoul_bike/informe/matriz_correlacion.png).
- **Interpretación física de las correlaciones con la demanda (`Rented Bike Count`):**

| Variable | Correlación ($r$) | Impacto y Comportamiento Físico |
|---|:---:|---|
| `Temperature` | **$+0.54$** | **Predictor positivo dominante.** El clima cálido y templado estimula masivamente el uso de bicicletas recreativas y de transporte. |
| `Hour` | **$+0.41$** | **Predictor circadiano.** Comportamiento bimodal con dos picos diarios claros de alta demanda: 8:00 AM (ingreso laboral/académico) y 18:00 PM (retorno a casa). |
| `Dew_Point_Temp` | **$+0.38$** | Correlación positiva, pero condicionada térmicamente (punto de rocío alto acompaña días cálidos). |
| `Season_Summer` | **$+0.30$** | Máximo aprovechamiento de horas de luz solar y clima estival. |
| `Solar_Radiation` | **$+0.26$** | Días despejados y soleados aumentan la disposición al pedaleo. |
| `Functioning_Binary` | **$+0.20$** | Variable determinante: si el sistema no funciona (0), la demanda es forzosamente 0. |
| `Visibility` | **$+0.20$** | Mayor visibilidad genera sensación de seguridad vial. |
| `Wind_Speed` | **$+0.12$** | Correlación baja pero positiva: vientos moderados no disuaden el uso en la ciudad. |
| `Holiday_Binary` | **$-0.05$** | Correlación lineal cercana a cero frente al total anual. |
| `Rainfall` | **$-0.12$** | La lluvia disuade de inmediato el alquiler por mojado y riesgo de resbalamiento. |
| `Snowfall` | **$-0.14$** | La nieve congela la calzada e inhabilita el transporte en bicicleta. |
| `Humidity` | **$-0.20$** | Humedades relativas altas provocan sofocación e incomodidad física. |
| `Season_Winter` | **$-0.42$** | **Predictor negativo dominante.** El invierno en Seúl alcanza hasta $-17.8$ °C con heladas severas, provocando el desplome más drástico de la demanda. |

- **ANÁLISIS PROFUNDO: ¿Se eliminó alguna variable? ¿Cuáles sirven y cuáles parecieran no tener nada?**
  *(Dilo con este nivel de argumentación para asombrar al docente):*
  
  > *"Profesor, realizamos un análisis crítico sobre la pertinencia de cada variable y tomamos decisiones sustentadas:*
  >
  > *1. **¿Qué variable parecía 'no tener nada' linealmente?**  
  > `Holiday_Binary` tiene un coeficiente de correlación de apenas $r = -0.05$. Cualquiera pensaría en eliminarla. Sin embargo, **NO se eliminó porque aporta una no linealidad clave:** los festivos representan menos del 5% del año, por lo que Pearson diluye su efecto global; pero en esos días puntuales, el patrón horario bimodal (8h y 18h) se destruye y la demanda se corre hacia horas del mediodía y la tarde. La red neuronal aprovecha esa variable para ajustar el comportamiento de las neuronas en días festivos.*
  >
  > *2. **El dilema de la lluvia (`Rainfall`, $r = -0.12$) y la nieve (`Snowfall`, $r = -0.14$):**  
  > Aparentan tener una correlación baja en magnitud. ¿Por qué? Porque en Seúl no llueve ni nieva en el 92% de las horas del año (la gran mayoría de valores son cero). Pero cuando llueve aunque sea 5 mm, el impacto en la realidad es un 'apagón' inmediato de la demanda. Una regresión lineal simple no puede ver esto, pero una **Red Neuronal MLP con funciones ReLU sí detecta el umbral de activación (si lluvia > 0 $\rightarrow$ penalizar fuertemente la salida)**. Por eso era obligatorio conservarlas.*
  >
  > *3. **El fenómeno de multicolinealidad: Temperatura vs Punto de Rocío ($r = 0.91$):**  
  > Detectamos una fuerte colinealidad física entre `Temperature` y `Dew_Point_Temp` ($r = 0.91$). En modelos estadísticos lineales tradicionales (como OLS), la teoría exige eliminar una de las dos para evitar matrices singulares o inestabilidad en los coeficientes.*  
  > ***¿Por qué decidimos NO eliminarla en nuestra Red Neuronal?***  
  > *Primero, porque las redes MLP no sufren el problema de inversión de matrices de OLS; los pesos en las capas ocultas distribuyen la carga sin colapsar.*  
  > *Segundo, y más importante: **la combinación no lineal de Temperatura, Humedad y Punto de Rocío le permite a la red calcular internamente el índice de confort térmico humano (sensación térmica)**. Un día con 28 °C y punto de rocío bajo se siente fresco y agradable para montar bicicleta, mientras que a 28 °C con punto de rocío alto se siente un bochorno insoportable. Mantener ambas variables le otorga mayor capacidad discriminativa al modelo con un costo de cómputo despreciable (solo 16 multiplicaciones más en Arduino)."*

---

### BLOQUE II: IMPLEMENTACIÓN DEL MODELO (40%)

#### Criterio 5: Arquitecturas de red probadas (10%)
- **Qué decir:**
  - *"Evaluamos 3 arquitecturas respetando el tope de 3 capas ocultas de la guía:*
    1. ***Arquitectura 1 (Ligera - 1 capa):*** `Input(15) -> Dense(16, ReLU) -> Dense(1, Linear)`. Total: **273 parámetros** ($R^2 = 0.7295$).
    2. ***Arquitectura 2 (Media - 2 capas):*** `Input(15) -> Dense(32, ReLU) -> Dense(16, ReLU) -> Dense(1, Linear)`. Total: **1,057 parámetros** ($R^2 = 0.8253$).
    3. ***Arquitectura 3 (Profunda - 3 capas):*** `Input(15) -> Dense(64, ReLU) -> Dense(32, ReLU) -> Dense(16, ReLU) -> Dense(1, Linear)`. Total: **3,649 parámetros** ($R^2 = 0.8295$).
  - *La función de activación oculta fue **ReLU** $f(z) = \max(0, z)$ para evitar el desvanecimiento de gradiente y garantizar cálculo eficiente en microcontrolador.*
  - *La activación de salida fue **LINEAL** $f(z) = z$, porque es un problema de regresión continua sin límites artificiales."*

#### Criterio 6: Optimizadores probados (10%)
- **Qué decir sobre los 3 optimizadores evaluados en la Arquitectura 2:**
  1. ***Adam (lr=0.005) ⭐:*** Mejor desempeño ($R^2 = 0.8253$). Combina momento de primer orden (media de gradientes pasados) y momento de segundo orden (varianza de gradientes), adaptando la tasa de aprendizaje independientemente para cada peso.
  2. ***RMSprop (lr=0.005):*** Buen desempeño ($R^2 = 0.7997$), divide el gradiente entre la raíz cuadrada del promedio móvil de cuadrados, amortiguando oscilaciones en direcciones de alta curvatura.
  3. ***SGD con Momentum (lr=0.02, momentum=0.9):*** Desempeño inferior ($R^2 = 0.7347$). Al usar una tasa fija para todos los parámetros, desciende lentamente y requiere un ajuste mucho más minucioso."*

#### Criterio 7: Gráfica de la función de pérdida (loss) (10%)
- **Qué mostrar:** La gráfica [`curvas_perdida.png`](file:///c:/Users/aleja/OneDrive/Escritorio/Redes%20Neuronales/proyecto_seoul_bike/informe/curvas_perdida.png).
- **Diagnóstico técnico:**
  - *"Curvas de entrenamiento (`loss`) vs validación (`val_loss`) en escala de MSE normalizado.*
  - *Convergencia: Adam y RMSprop alcanzan su asíntota estable antes de la época 30.*
  - ***Ausencia total de Sobreajuste (Overfitting):*** *La curva de validación acompaña en paralelo a la de entrenamiento sin divergir jamás hacia arriba.*
  - ***Ausencia de Subajuste (Underfitting):*** *El error cuadrático medio desciende por debajo de 0.0062, capturando la dinámica no lineal."*

#### Criterio 8: Uso e integración con TensorBoard (10%)
- **Qué mostrar:**
  - Enseña la línea de código: `TensorBoard(log_dir=log_dir, write_graph=True)`.
  - Muestra la carpeta `logs/` con los 9 subdirectorios generados.
  - Abre la pestaña de TensorBoard (`localhost:6006`):
    - Muestra la pestaña **Graphs** con el grafo computacional de capas densas y tensores.
    - Muestra la pestaña **Scalars** con la comparación en vivo de las curvas de convergencia.

---

### BLOQUE III: VALIDACIÓN Y DESPLIEGUE EN ARDUINO (30%)

#### Criterio 9: Métricas de validación e interpretación (10%)
- **Qué decir con las métricas en escala real (tras desnormalizar):**
  - **$R^2 = 0.8253$:** El modelo explica el **82.53% de la variabilidad total** de la demanda de bicicletas en Seúl sobre datos de prueba independientes (1,752 horas no vistas).
  - **MAE = $174.93$ bicicletas/hora:** En promedio, la red se desvía en 174 bicicletas respecto a la demanda real, lo cual es un margen de tolerancia excelente para gestionar flotas de hasta 3,556 bicicletas.
  - **RMSE = $269.82$ bicicletas/hora:** Mide la dispersión cuadrática penalizando desviaciones atípicas en eventos meteorológicos extremos.

#### Criterio 10: Justificación de selección del mejor modelo (10%)
- **El argumento ganador (Trade-off de Ingeniería):**
  - *"Seleccionamos la **Arquitectura 2 con Optimizador Adam** para implementarla en Arduino mediante un análisis riguroso de compromiso entre precisión y consumo de recursos:*
    - *La Arquitectura 3 (3 capas) alcanza un $R^2 = 0.8295$, pero exige **3,649 parámetros** (más de 14.6 KB de memoria Flash).*
    - *La Arquitectura 2 (2 capas) alcanza un $R^2 = 0.8253$, lo que representa una **diferencia mínima de apenas 0.42%**, pero utilizando únicamente **1,057 parámetros**.*
    - *Logramos una **reducción del 71.0% en memoria y complejidad de cómputo**.*
    - *En un microcontrolador ATmega328P de Arduino Uno (que solo tiene 32 KB de Flash y 2 KB de SRAM), la Arquitectura 2 consume apenas 4.2 KB (menos del 14% de la memoria), dejando libre el 86% para el programa principal, sensores y puertos seriales, con un tiempo de inferencia de solo **350 microsegundos**."*

#### Criterio 11: Implementación y verificación en Arduino / Wokwi (10%)
- **Qué mostrar en pantalla:**
  1. Abre Wokwi y muestra el código [`seoul_bike_mlp.ino`](file:///c:/Users/aleja/OneDrive/Escritorio/Redes%20Neuronales/proyecto_seoul_bike/arduino_wokwi/seoul_bike_mlp.ino) y [`weights.h`](file:///c:/Users/aleja/OneDrive/Escritorio/Redes%20Neuronales/proyecto_seoul_bike/arduino_wokwi/weights.h).
  2. Haz clic en **Play** e inspecciona el **Serial Monitor** (115200 baudios).
  3. Explica los 3 pasos de la función `forward()` en C++:
     - Normalización de las 15 entradas con `(raw - x_min) / (x_max - x_min)`.
     - Multiplicación matricial con pesos y sesgos reales extraídos de Keras y aplicación de ReLU `max(0, z)`.
     - **DESNORMALIZACIÓN OBLIGATORIA:**  
       `pred_real = (y_norm * (y_max - y_min)) + y_min;`
  4. Muestra la coincidencia exacta de las 5 muestras de prueba:
     - **Muestra 1 (Hora 8h, 27.2°C, 69% Hum):** Real = 1,728.0 | Keras = 1,251.54 | Arduino = 1,251.54 | Error = $2.63 \times 10^{-5}$ (**Coincidencia 100%**).
     - **Muestra 2 (Hora 12h, 32.6°C):** Real = 822.0 | Keras = 977.18 | Arduino = 977.18.
     - **Muestra 4 (Hora 18h pico, 16.9°C):** Real = 2,716.0 | Keras = 2,223.60 | Arduino = 2,223.60.
  5. Concluye: *"El error entre Arduino y Keras es menor a $10^{-4}$ bicicletas, demostrando que la emulación en C++ es matemáticamente idéntica al modelo entrenado en TensorFlow."*

---

## 3. BANCO DE PREGUNTAS TÍPICAS DEL PROFESOR Y CÓMO RESPONDERLAS

### P1: "¿Por qué no eliminaron la variable Dew Point Temperature si tenía correlación de 0.91 con la temperatura?"
- **Respuesta:** *"Profesor, porque las Redes Neuronales MLP con capas densas y regularización no sufren el problema de matriz singular que afecta a los modelos de regresión lineal OLS. Además, físicamente, la temperatura sola no indica el confort térmico; al combinar Temperatura, Humedad y Punto de Rocío de forma no lineal, la red es capaz de aprender el índice de sensación térmica humana, distinguiendo entre un calor seco agradable para pedalear y un calor bochornoso insoportable."*

### P2: "¿Por qué la variable Holiday tiene correlación tan baja (-0.05) y aun así la dejaron?"
- **Respuesta:** *"Porque el coeficiente de Pearson solo mide relaciones lineales globales a lo largo de todo el año, y los días festivos representan menos del 5% del total de muestras. Sin embargo, en esos días festivos la curva horaria cambia radicalmente: se eliminan los picos de las 8 AM y 6 PM y la gente sale a montar bicicleta en horas de la tarde. La red utiliza ese 1 binario para modular los sesgos de las neuronas en días festivos."*

### P3: "¿Por qué la última capa tiene activación lineal y no ReLU o Sigmoide?"
- **Respuesta:** *"Porque estamos resolviendo un problema de regresión continua. Si pusiéramos una sigmoide, la red solo podría generar salidas entre 0 y 1; y si pusiéramos ReLU, truncaríamos a cero cualquier cálculo negativo intermedio antes de desnormalizar. La función lineal $f(z) = z$ permite a la neurona proyectar en toda la recta numérica real $\mathbb{R}$."*

### P4: "¿Qué sucedería si olvidan desnormalizar la salida en el Arduino?"
- **Respuesta:** *"El Arduino arrojaría un número adimensional como `0.3519`, lo cual no tiene ninguna utilidad práctica para el operador del sistema de transporte. Al aplicar la desnormalización matemática $\hat{y}_{real} = \hat{y}_{norm} \cdot (y_{max} - y_{min}) + y_{min}$, el Arduino entrega exactamente `1,251.5 bicicletas/hora`, permitiendo una toma de decisiones física real."*

### P5: "¿Por qué usaron MinMaxScaler y no StandardScaler (Z-score)?"
- **Respuesta:** *"MinMaxScaler acota rígidamente los valores en el intervalo $[0, 1]$. En sistemas embebidos como el ATmega328P de Arduino, mantener las entradas entre 0 y 1 previene desbordamientos de registros de punto flotante de 32 bits y asegura que los pesos no generen valores extremos en las multiplicaciones acumuladas. Además, la normalización y desnormalización solo requieren constantes simples ($x_{min}, x_{max}$), mientras que Z-score requeriría medias y desviaciones estándar más costosas de computar."*
