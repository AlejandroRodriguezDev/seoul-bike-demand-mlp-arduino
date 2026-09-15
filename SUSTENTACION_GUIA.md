# GUION MAESTRO DE SUSTENTACIÓN ORAL - PROYECTO SEOUL BIKE MLP

**Universidad Autónoma de Occidente (UAO)**  
**Materia:** Redes Neuronales Artificiales y Deep Learning  
**Docente:** Prof. Gilber Alexis Corrales Gallego  
**Estudiantes:**  
- **Parte 1 (Desde el inicio hasta TensorBoard):** Alejandro Rodríguez Cortés (Cód. 2225645)  
- **Parte 2 (Métricas, Justificación y Arduino Wokwi):** Nicolás Mejía Ochoa (Cód. 2205076)  

---

## ESTRUCTURA DE LA SUSTENTACIÓN Y TIEMPOS ESTIMADOS (Total: ~15 a 20 min)
* **Parte 1 - Alejandro (~10 min):**
  - Introducción y contexto del problema real (Seúl).
  - Bloque I (AED): Carga, nulos, codificación de variables, correlación, partición y normalización.
  - Bloque II (Modelado): Definición de 3 arquitecturas, entrenamiento con bucles `for`, optimizadores, curvas de pérdida y panel interactivo de TensorBoard.
* **Parte 2 - Nicolás (~7 min):**
  - Bloque III (Validación y Arduino): Métricas desnormalizadas ($R^2$, MAE, RMSE), trade-off de memoria Flash para microcontrolador, código C++ en Wokwi y verificación exacta frente a Keras.
* **Preguntas del profesor (~3 min).**

---

# PARTE 1: GUION DE ALEJANDRO RODRÍGUEZ
*(Desde la introducción hasta el visor de TensorBoard)*

---

### PASO 0: INTRODUCCIÓN Y PORTADA
* **QUÉ MOSTRAR EN PANTALLA:**
  - Abre el cuaderno de Google Colab en la parte superior (título del proyecto y nombres).
* **QUÉ DECIR TEXTUALMENTE:**
  > *"Buenos días, profesor Gilber. Nuestro miniproyecto consiste en predecir la demanda horaria de alquiler de bicicletas públicas en Seúl, Corea del Sur, utilizando Redes Neuronales Perceptrón Multicapa (MLP) en TensorFlow y Keras, y posteriormente desplegar este modelo optimizado en un microcontrolador Arduino simulado en Wokwi.  
  > Nos dividimos la presentación en dos partes: yo explicaré todo el Bloque I de Análisis Exploratorio de Datos y el Bloque II de Implementación, Modelado y TensorBoard; luego mi compañero Nicolás explicará el Bloque III de Validación, Métricas en escala real y la implementación física en C++ para Arduino."*

---

### PASO 1: CELDA 1 - IMPORTACIONES Y REPRODUCIBILIDAD
* **QUÉ MOSTRAR EN PANTALLA:**
  - Celda 1 de código (`import tensorflow... SEED = 42`).
* **QUÉ DECIR TEXTUALMENTE:**
  > *"Aquí importamos las librerías fundamentales: TensorFlow y Keras para las redes neuronales, Scikit-learn para el preprocesamiento y Pandas/NumPy para el tratamiento tabular.  
  > **Un detalle clave aquí:** en las líneas 72 a 74 fijamos la semilla de reproducibilidad `SEED = 42` tanto en NumPy como en TensorFlow. Esto garantiza que la inicialización aleatoria de los pesos sinápticos y la partición de los datos sean 100% reproducibles si usted ejecuta este cuaderno en su equipo."*

---

### PASO 2: CELDA 2 - CARGA ROBUSTA DEL DATASET
* **QUÉ MOSTRAR EN PANTALLA:**
  - Celda 2 (`DATA_URL = ... df = pd.read_csv(...)`) y la salida con `df.head()`.
* **QUÉ DECIR TEXTUALMENTE:**
  > *"El dataset proviene del repositorio UCI Machine Learning con ID 560. Registra 8,760 horas continuas (un año completo de Seúl) con 14 variables originales.  
  > Implementamos una carga robusta con bloques `try-except`: si no encuentra el archivo CSV local en la máquina, lo descarga automáticamente desde nuestro repositorio de GitHub, permitiendo que el cuaderno corra en Google Colab de forma totalmente autónoma sin requerir subir archivos a mano."*

---

### PASO 3: CELDA 3 - EXPLORACIÓN, TIPOS Y VALORES NULOS (Criterio 2)
* **QUÉ MOSTRAR EN PANTALLA:**
  - Celda 3 (`df.isnull().sum()` y `df.describe().T`).
* **QUÉ DECIR TEXTUALMENTE:**
  > *"Verificamos la calidad del dataset. Con `df.isnull().sum()` confirmamos que existen **cero valores nulos** en todas las columnas, por lo que no fue necesario imputar datos faltantes.  
  > En la tabla descriptiva observamos la gran disparidad de escalas: la visibilidad llega a 2,000 metros, la humedad va de 0 a 100%, la temperatura de -17 a 39°C, y la variable objetivo `Rented_Bike_Count` tiene una media de 704 bicicletas y un valor máximo de 3,556 bicicletas por hora."*

---

### PASO 4: CELDA 4 - TRATAMIENTO DE VARIABLES CATEGÓRICAS (Criterio 2)
* **QUÉ MOSTRAR EN PANTALLA:**
  - Celda 4 (`Holiday_Binary`, `Functioning_Binary`, `pd.get_dummies` y la lista `feature_cols`).
* **QUÉ DECIR TEXTUALMENTE:**
  > *"Las redes neuronales solo procesan números continuos. Por eso transformamos las variables cualitativas:  
  > 1. Eliminamos la columna de texto `Date` porque el patrón temporal ya está capturado en la hora y la estación.  
  > 2. Convertimos `Holiday` y `Functioning_Day` a binarias `0` o `1` mediante comparaciones booleanas.  
  > 3. En la variable `Seasons` (las 4 estaciones) aplicamos **One-Hot Encoding** con `pd.get_dummies(..., dtype=int)`. Esto genera 4 columnas mutuamente excluyentes (Primavera, Verano, Otoño, Invierno).  
  > 4. Al concatenar con `axis=1`, obtenemos un vector final de **15 variables predictoras de entrada** para nuestra red."*

---

### PASO 5: CELDA 5 - MATRIZ DE CORRELACIÓN DE PEARSON (Criterio 4)
* **QUÉ MOSTRAR EN PANTALLA:**
  - Celda 5 (El mapa de calor en azul y rojo `sns.heatmap` y la lista ordenada).
* **QUÉ DECIR TEXTUALMENTE:**
  > *"Aquí generamos la matriz de correlación lineal de Pearson con `pd.concat([y, X], axis=1).corr()`.  
  > **Interpretación técnica:**  
  > - Las variables con mayor correlación positiva son la **Temperatura (+0.54)** y la **Hora del día (+0.41)**: a mayor calor ambiental y durante las horas pico de la tarde (18:00 horas), el alquiler de bicicletas se dispara.  
  > - La correlación negativa más fuerte es el **Invierno (-0.42)** y la **Humedad (-0.20)**, lo cual es físicamente lógico por el frío extremo y el pavimento resbaloso.  
  > - Entre `Temperature` y `Dew_Point_Temp` (punto de rocío) hay una correlación de **0.91**. En regresión lineal clásica esto causaría problemas de singularidad, pero en redes neuronales la red aprovecha ambas entradas para inferir la **sensación térmica humana**, por lo que decidimos conservar ambas."*

---

### PASO 6: CELDA 6 - PARTICIÓN Y NORMALIZACIÓN SIN DATA LEAKAGE (Criterio 3)
* **QUÉ MOSTRAR EN PANTALLA:**
  - Celda 6 (`train_test_split`, `MinMaxScaler`, `fit_transform` y `transform`).
* **QUÉ DECIR TEXTUALMENTE:**
  > *"Realizamos la partición de datos con `train_test_split`: **80% entrenamiento (7,008 muestras)** y **20% prueba (1,752 muestras)** con `shuffle=True` para garantizar que todas las estaciones del año queden representadas en ambos grupos.  
  > **Justificación de MinMaxScaler al rango [0, 1]:**  
  > Elegimos Min-Max porque acota estrictamente los datos entre 0 y 1. Esto es ideal para microcontroladores como Arduino, ya que evita desbordamientos (*overflow*) en la aritmética de punto flotante de 32 bits y reduce el cómputo en C++ a una resta y división simples.  
  > **Prevención de Data Leakage:**  
  > Note que en `X_train` usamos `fit_transform()` para calcular el mínimo y máximo de entrenamiento, mientras que en `X_test` usamos **únicamente `transform()`**. Esto garantiza que el conjunto de prueba sea completamente ciego y no contamine el aprendizaje."*

---

### PASO 7: CELDA 7 - DISEÑO DE LAS 3 ARQUITECTURAS MLP (Criterio 5)
* **QUÉ MOSTRAR EN PANTALLA:**
  - Celda 7 (`build_arch_1`, `build_arch_2`, `build_arch_3` y la tabla `m2.summary()`).
* **QUÉ DECIR TEXTUALMENTE:**
  > *"Siguiendo las pautas del curso de evaluar hasta 3 capas ocultas, diseñamos 3 arquitecturas progresivas:  
  > - **Arquitectura 1 (Ligera):** 1 capa oculta de 16 neuronas (273 parámetros).  
  > - **Arquitectura 2 (Media):** 2 capas ocultas (32 y 16 neuronas, 1,057 parámetros).  
  > - **Arquitectura 3 (Profunda):** 3 capas ocultas (64, 32 y 16 neuronas, 3,649 parámetros).  
  > **Aspecto fundamental de regresión:** Todas las capas ocultas usan activación no lineal **ReLU**, pero la capa de salida tiene **1 sola neurona con activación estrictamente LINEAL (`activation='linear'`)**, lo que le permite a la red proyectar cualquier valor continuo en los reales sin truncar ni saturar.  
  > En pantalla vemos el `summary()` de la Arquitectura 2: $15 \times 32 + 32 = 512$ pesos en la capa 1; $32 \times 16 + 16 = 528$ en la capa 2; y $16 \times 1 + 1 = 17$ en la salida. Total exacto: **1,057 parámetros entrenables**."*

---

### PASO 8: CELDA 8 - PREPARACIÓN DE TENSORBOARD (Criterio 8)
* **QUÉ MOSTRAR EN PANTALLA:**
  - Celda 8 (`%load_ext tensorboard`).
* **QUÉ DECIR TEXTUALMENTE:**
  > *"En esta celda ejecutamos `%load_ext tensorboard` para cargar la extensión en el entorno. Esto deja el sistema listo para que los entrenamientos que vienen a continuación registren automáticamente sus grafos y curvas en la carpeta `logs/`."*

---

### PASO 9: CELDA 9 - COMPARACIÓN DE ARQUITECTURAS (Criterio 5 y Código Celda 9)
* **QUÉ MOSTRAR EN PANTALLA:**
  - Celda 9 de código completa y la tabla final `pd.DataFrame(results_arch).T`.
* **QUÉ DECIR TEXTUALMENTE:**
  > *"Aquí ejecutamos el primer experimento comparativo manteniendo fijo el optimizador Adam con tasa de aprendizaje de 0.005.  
  > **¿Por qué usamos un ciclo `for`?**  
  > Para aplicar el principio de ingeniería DRY (*Don't Repeat Yourself*). En lugar de copiar y pegar el mismo código de entrenamiento 3 veces, iteramos sobre el diccionario `architectures.items()`, asegurando que las 3 redes se entrenen bajo exactamente las mismas 80 épocas, con lotes de 64 muestras y registrándose en TensorBoard mediante el callback `TensorBoard(write_graph=True)`.  
  > **Línea de desnormalización:**  
  > En la línea 388 ejecutamos `scaler_y.inverse_transform(model.predict(X_test))`. Esto es crucial: la red predice números normalizados de 0 a 1, y con esta línea los regresamos a **unidades reales de bicicletas**.  
  > **Resultado en pantalla:**  
  > La Arquitectura 1 se queda corta con $R^2 = 0.77$. La Arquitectura 2 sube a **$R^2 = 0.8253$**, y la Arquitectura 3 alcanza $R^2 = 0.8295$. Más adelante mi compañero Nicolás sustentará por qué la Arquitectura 2 es la ganadora para el Arduino."*

---

### PASO 10: CELDA 10 - COMPARACIÓN DE OPTIMIZADORES (Criterio 6 y Código Celda 10)
* **QUÉ MOSTRAR EN PANTALLA:**
  - Celda 10 de código completa y la tabla final `pd.DataFrame(results_opt).T`.
* **QUÉ DECIR TEXTUALMENTE:**
  > *"En el Experimento 2 evaluamos los 3 optimizadores sobre la Arquitectura 2: **Adam**, **RMSprop** y **SGD con Momentum**.  
  > **Detalle técnico de vida o muerte en este bucle:**  
  > La línea `model = build_arch_2(input_dim)` está **DENTRO** del ciclo `for`. Esto es indispensable para que cada optimizador arranque con pesos nuevos y frescos desde cero; si estuviera afuera, el segundo optimizador entrenaría sobre los pesos ya modificados por el primero.  
  > **Comportamiento:**  
  > - Adam y RMSprop utilizan tasas adaptativas ($\eta = 0.005$) y logran un desempeño sobresaliente ($R^2$ de 0.8253 y 0.8140).  
  > - SGD con Momentum usa una tasa mayor ($\eta = 0.02$) e inercia de $\gamma = 0.9$ para amortiguar oscilaciones transversales, alcanzando un $R^2 = 0.7380$."*

---

### PASO 11: CELDA 11 - CURVAS DE PÉRDIDA Y CONVERGENCIA (Criterio 7)
* **QUÉ MOSTRAR EN PANTALLA:**
  - Celda 11 con los dos gráficos de Matplotlib (`Train` vs `Val`).
* **QUÉ DECIR TEXTUALMENTE:**
  > *"Aquí graficamos la función de pérdida MSE normalizada a lo largo de las 80 épocas:  
  > 1. **Convergencia rápida:** Adam y RMSprop alcanzan su asíntota horizontal antes de la época 30.  
  > 2. **¿Por qué 80 épocas?** Porque entre la época 40 y 80 la pérdida ya no disminuye más y se estabiliza por debajo de 0.0062. Dejarlo hasta 80 épocas nos permitió verificar estabilidad asintótica total.  
  > 3. **Ausencia total de Sobreajuste (Overfitting):** Fíjense en la curva discontinua de validación (`val_loss`): corre totalmente paralela a la curva de entrenamiento sin despegarse jamás hacia arriba, lo que demuestra excelente capacidad de generalización."*

---

### PASO 12: CELDA 12 - PANEL INTERACTIVO DE TENSORBOARD (Criterio 8)
* **QUÉ MOSTRAR EN PANTALLA:**
  - Celda 12 con el comando `%tensorboard --logdir logs`.
  - Muestra la pestaña **SCALARS** (activas y desactivas una arquitectura para ver la curva interactiva).
  - Cambias a la pestaña **GRAPHS** para mostrar el diagrama de bloques de las capas densas de Keras.
* **QUÉ DECIR TEXTUALMENTE:**
  > *"En esta celda desplegamos TensorBoard directamente dentro del cuaderno mediante `%tensorboard --logdir logs`.  
  > - En la pestaña **Scalars** podemos comparar dinámicamente el comportamiento de cada experimento, suavizar las curvas y analizar la convergencia.  
  > - En la pestaña **Graphs** vemos el flujo computacional de tensores: las 15 entradas, la primera capa densa con sus operaciones de producto punto y sesgo, la activación ReLU, la segunda capa oculta y la salida lineal.  
  > Con esto concluyo la demostración de los Bloques I y II. Ahora le cedo la palabra a mi compañero Nicolás Mejía, quien explicará las métricas de validación final y la implementación en Arduino."*

---

# PARTE 2: GUION DE NICOLÁS MEJÍA OCHOA
*(Desde las métricas de validación hasta la simulación en Arduino Wokwi)*

---

### PASO 13: MÉTRICAS DE VALIDACIÓN EN UNIDADES REALES (Criterio 9)
* **QUÉ MOSTRAR EN PANTALLA:**
  - La tabla de métricas finales o el texto resumen del Bloque III.
* **QUÉ DECIR TEXTUALMENTE:**
  > *"Buenos días, profesor. Continuando con la sustentación, evaluamos el modelo ganador (Arquitectura 2 con Adam) sobre las 1,752 muestras de prueba no vistas, desnormalizadas a unidades tangibles de bicicletas:  
  > - **$R^2 = 0.8253$ (82.53%):** El modelo explica más del 82.5% de la varianza total de la demanda en Seúl.  
  > - **MAE = 174.93 bicicletas/hora:** En promedio, la red se desvía en 174 bicicletas frente a la demanda real. Considerando que la demanda oscila entre 0 y 3,556 bicicletas, este error medio representa apenas el **4.9% del rango operativo**, siendo altamente confiable para la gestión de flotas.  
  > - **RMSE = 269.82 bicicletas/hora:** Mide la dispersión cuadrática penalizando desviaciones atípicas."*

---

### PASO 14: JUSTIFICACIÓN TÉCNICA DE SELECCIÓN DEL MODELO (Criterio 10)
* **QUÉ MOSTRAR EN PANTALLA:**
  - La tabla comparativa de Arquitecturas (mostrando los parámetros: 1,057 vs 3,649).
* **QUÉ DECIR TEXTUALMENTE:**
  > *"Para seleccionar el mejor modelo aplicamos un criterio de ingeniería riguroso enfocado en sistemas embebidos (*Trade-off* Precisión vs Recursos):  
  > - La Arquitectura 3 (3 capas) alcanza un $R^2 = 0.8295$, pero requiere **3,649 parámetros**, consumiendo cerca de 14.6 KB de Flash.  
  > - La Arquitectura 2 (2 capas) alcanza un $R^2 = 0.8253$. La diferencia en precisión es de un insignificante **0.42%**, pero la Arquitectura 2 solo utiliza **1,057 parámetros**.  
  > - Esto representa una **reducción del 71.0% en memoria y complejidad computacional**. El modelo ocupa apenas **4.2 KB de memoria Flash**, permitiendo que quepa holgadamente en el microcontrolador ATmega328P de 32 KB del Arduino Uno y en el ATmega2560 de Arduino Mega, dejando más del 85% de la memoria libre para sensores y comunicaciones."*

---

### PASO 15: DESPLIEGUE EN ARDUINO / WOKWI Y VERIFICACIÓN (Criterio 11)
* **QUÉ MOSTRAR EN PANTALLA:**
  - Cambia a la pestaña del navegador con el proyecto de **Wokwi** (`seoul_bike_mlp.ino`).
  - Dale click al botón verde de **Play / Simular**.
  - Abre el **Serial Monitor** para ver los números imprimiéndose en tiempo real.
* **QUÉ DECIR TEXTUALMENTE:**
  > *"Siguiendo las directrices estrictas del curso, **no utilizamos TensorFlow Lite ni librerías externas**. Implementamos una red neuronal en C++ nativo puro sobre Arduino:  
  > 1. Exportamos los 1,057 pesos entrenados al archivo de cabecera `weights.h` en arreglos constantes `const float PROGMEM`.  
  > 2. En `seoul_bike_mlp.ino`, la función `forward()` realiza el escalado interno Min-Max de los 15 sensores, ejecuta la multiplicación matricial $z = \sum w_i x_i + b$ con bucles `for`, aplica la función ReLU $\max(0, z)$ y la salida lineal.  
  > 3. **Línea de Desnormalización Obligatoria en Arduino:** En la línea 63 del código de Arduino aplicamos:  
  >    `float y_real = y_norm * (Y_MAX - Y_MIN) + Y_MIN;`  
  >    Esto permite que el Arduino no imprima un número abstracto entre 0 y 1, sino las **bicicletas reales por hora**.  
  > 4. **Verificación numérica frente a Keras:**  
  >    En pantalla el Serial Monitor evalúa 5 muestras del conjunto de prueba. Fíjense en la Muestra 1: Keras predijo `1251.54` y Arduino calcula exactamente `1251.54`. En la Muestra 2: Keras predijo `977.18` y Arduino `977.18`. La diferencia es de apenas $10^{-5}$, demostrando coincidencia matemática absoluta entre el modelo de Keras y el microcontrolador."*

---

# ANEXO RÁPIDO: PREGUNTAS TÍPICAS DEL PROFESOR GILBER (BANCO ANTI-CORCHADAS)

Si el profesor interrumpe o hace preguntas al final:

1. **"En la Muestra 2, la demanda real fue 822 y el modelo predijo 977. ¿Ese error de 155 bicicletas no es muy grande?"**
   - **Respuesta:** *"No, profesor. En Seúl la demanda llega a 3,556 bicicletas. Un error de 155 es apenas el 4.3% del rango total, muy por debajo de nuestro MAE de 174 bicicletas. Además, modelamos comportamiento humano libre de 10 millones de personas, no circuitos ideales. Predecir 822 exacto en datos de prueba ciegos sería síntoma de sobreajuste catastrófico."*

2. **"¿Por qué la capa de salida tiene activación lineal y no Sigmoide o ReLU?"**
   - **Respuesta:** *"Porque es un problema de regresión continua. La Sigmoide acotaría la salida entre 0 y 1 destruyendo la escala real; y la ReLU truncaría a cero cualquier cálculo negativo intermedio antes de desnormalizar. La activación lineal $f(z) = z$ permite a la neurona proyectar en toda la recta real $\mathbb{R}$."*

3. **"¿Qué es Backpropagation?"**
   - **Respuesta:** *"Es el algoritmo que propaga el error desde la salida hacia atrás mediante la Regla de la Cadena del cálculo diferencial, calculando el gradiente $\frac{\partial L}{\partial W}$ para saber en qué dirección y magnitud ajustar cada uno de los 1,057 pesos."*

4. **"¿Por qué usaron MinMaxScaler y no StandardScaler (Z-score)?"**
   - **Respuesta:** *"MinMaxScaler acota estrictamente los valores entre 0 y 1, lo que en microcontroladores embebidos como Arduino evita el desbordamiento de registros de punto flotante de 32 bits y reduce la matemática a una resta y división elementales."*

5. **"¿Qué pasaría si quitan la desnormalización en el Arduino?"**
   - **Respuesta:** *"El Arduino entregaría un decimal adimensional como 0.3519, el cual es inútil para la operación logística de la estación. Al desnormalizar con $(y_{max} - y_{min}) + y_{min}$, el Arduino entrega las 1,251 bicicletas requeridas para la toma de decisiones."*
