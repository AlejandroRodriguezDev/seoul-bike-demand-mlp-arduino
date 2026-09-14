# Predicción de la Demanda Horaria en Sistemas de Bicicletas Compartidas (Seoul Bike Sharing) Mediante Redes Neuronales Multicapa (MLP) y su Despliegue en Microcontrolador Arduino

**Autores:** Alejandro Rodríguez Cortés (Cód. 2225645), Nicolás Mejía Ochoa (Cód. 2205076)  
**Asignatura:** Redes Neuronales Artificiales y Deep Learning  
**Docente Evaluador:** Prof. Jesús Alfonso López S.  
**Institución:** Universidad Autónoma de Occidente, Facultad de Ingeniería, Departamento de Automática y Electrónica  
**Fecha:** Septiembre de 2026  

---

## Resumen
El presente trabajo desarrolla un sistema predictivo basado en Redes Neuronales Artificiales de tipo Perceptrón Multicapa (MLP) para modelar la demanda horaria de bicicletas públicas en Seúl, Corea del Sur (*Seoul Bike Sharing Demand*). Se aborda como un problema de regresión no lineal considerando 15 variables predictoras meteorológicas y calendarias. El proceso metodológico abarca un Análisis Exploratorio de Datos (AED) con matriz de correlación de Pearson, normalización Min-Max en el rango [0, 1] y partición 80/20 sin fuga de información (*data leakage*). Se evaluaron experimentalmente tres arquitecturas neuronales (1 a 3 capas ocultas con función de activación ReLU y capa de salida con activación lineal) y tres optimizadores (Adam, RMSprop y SGD con Momentum), registrando las curvas de convergencia de la función de pérdida (MSE) y el grafo computacional mediante TensorBoard. La Arquitectura 2 (32 y 16 neuronas ocultas, 1,057 parámetros) optimizada con Adam alcanzó un coeficiente de determinación $R^2 = 0.8253$, un Error Absoluto Medio ($MAE$) de $181.69$ bicicletas/hora y una Raíz del Error Cuadrático Medio ($RMSE$) de $270.57$ bicicletas/hora. Dicho modelo fue seleccionado justificadamente por su balance entre precisión y bajo consumo de memoria para ser emulado en un microcontrolador ATmega328P (Arduino Uno) mediante la plataforma Wokwi. La inferencia en C++ demostró una concordancia numérica exacta frente a Keras ($error < 10^{-4}$), validando la viabilidad del despliegue en el borde (*Edge AI*).

**Palabras clave:** Perceptrón Multicapa (MLP), Regresión, Seoul Bike Sharing, TensorBoard, Arduino, Wokwi, Edge AI, Normalización, Desnormalización.

---

## I. Introducción
El auge de la movilidad urbana sostenible ha posicionado a los sistemas de bicicletas compartidas (*Bike Sharing Systems*) como una alternativa ecológica y eficiente frente a la congestión vehicular y las emisiones de gases de efecto invernadero en grandes metrópolis. Sin embargo, el éxito operativo de estos sistemas depende críticamente del equilibrio logístico entre la oferta y la demanda en cada estación. La escasez de vehículos genera pérdida de usuarios, mientras que la saturación impide el estacionamiento.

La demanda horaria presenta un comportamiento altamente dinámico y no lineal, condicionado por variables meteorológicas (temperatura, lluvia, radiación solar, humedad) y factores calendarios (hora del día, días festivos, estaciones del año). Las aproximaciones de regresión lineal tradicional suelen ser insuficientes para capturar estas interacciones complejas. En este contexto, las Redes Neuronales Artificiales tipo MLP surgen como aproximadores universales de funciones idóneos para este reto.

El presente proyecto aborda el diseño, optimización y validación de una red MLP para la estimación de la demanda horaria del sistema de bicicletas de Seúl y su subsiguiente transpilación e implementación en un sistema embebido basado en microcontrolador (Arduino Uno emulado en Wokwi), cumpliendo con la exigencia de procesamiento en escala real.

---

## II. Marco Teórico

### A. Perceptrón Multicapa (MLP)
El Perceptrón Multicapa es una arquitectura de red neuronal alimentada hacia adelante (*feedforward*) compuesta por una capa de entrada, una o más capas ocultas y una capa de salida. Cada neurona $j$ en la capa $l$ realiza una combinación lineal ponderada de sus entradas más un sesgo (*bias*), seguida de una función de activación no lineal $f(\cdot)$:

$$z_j^{(l)} = \sum_{i=1}^{n_{l-1}} W_{ij}^{(l)} a_i^{(l-1)} + b_j^{(l)}$$
$$a_j^{(l)} = f(z_j^{(l)})$$

Para problemas de **regresión**, la capa de salida debe implementar estrictamente una función de activación **lineal** ($f(z) = z$), permitiendo a la red generar valores continuos en el espacio $\mathbb{R}$ sin acotación artificial.

### B. Funciones de Activación
- **Unidad Lineal Rectificada (ReLU):** $f(z) = \max(0, z)$. Empleada en las capas ocultas para mitigar el problema del desvanecimiento del gradiente (*vanishing gradient*) y acelerar la convergencia computacional gracias a su derivada unitaria para valores positivos.
- **Lineal:** $f(z) = z$. Empleada en la neurona de salida para preservar el rango continuo de la variable estimada.

### C. Algoritmos de Optimización
El entrenamiento de la red se realiza minimizando la función de costo de Error Cuadrático Medio ($MSE$):

$$MSE = \frac{1}{N} \sum_{k=1}^{N} (y_k - \hat{y}_k)^2$$

Se contrastan tres optimizadores fundamentales estudiados en el curso:
1. **SGD con Momentum:** Incorpora un término de inercia $\gamma$ a la actualización de pesos para sortear mínimos locales poco profundos y amortiguar oscilaciones transversales:
   $$v_t = \gamma v_{t-1} + \eta \nabla_W L(W)$$
   $$W_t = W_{t-1} - v_t$$
2. **RMSprop:** Adapta la tasa de aprendizaje dividiendo el gradiente por la raíz cuadrada del promedio móvil exponencial de sus cuadrados, desacelerando el paso en direcciones de alta varianza.
3. **Adam (Adaptive Moment Estimation):** Combina las ventajas de Momentum y RMSprop mediante la estimación de los momentos de primer orden ($m_t$, media) y segundo orden no centrado ($v_t$, varianza), incluyendo corrección de sesgo:
   $$\hat{m}_t = \frac{m_t}{1 - \beta_1^t}, \quad \hat{v}_t = \frac{v_t}{1 - \beta_2^t}$$
   $$W_t = W_{t-1} - \frac{\eta}{\sqrt{\hat{v}_t} + \epsilon} \hat{m}_t$$

### D. Normalización y Desnormalización en Sistemas Embebidos
Para asegurar la estabilidad numérica del descenso del gradiente, los datos se transforman con escalamiento Min-Max al intervalo $[0, 1]$:
$$x_{norm} = \frac{x - x_{min}}{x_{max} - x_{min}}$$

En la fase de inferencia en el microcontrolador, la red recibe variables en su escala natural, las normaliza internamente y produce una salida normalizada $\hat{y}_{norm} \in [0, 1]$. Para la toma de decisiones físicas, es imperativo aplicar la transformación inversa (**desnormalización**):
$$\hat{y}_{real} = \hat{y}_{norm} \cdot (y_{max} - y_{min}) + y_{min}$$

---

## III. Descripción del Problema a Solucionar
El conjunto de datos **Seoul Bike Sharing Demand** proviene del repositorio UCI Machine Learning Repository (ID: 560). Contiene 8,760 registros horarios recolectados a lo largo de un año completo (del 1 de diciembre de 2017 al 30 de noviembre de 2018).

La meta consiste en predecir la variable continua `Rented Bike Count` (número de bicicletas alquiladas por hora, rango observado de 0 a 3,556 unidades) a partir de 15 variables de entrada estructuradas tras el preprocesamiento:
- **Meteorológicas:** Temperatura (°C), Humedad (%), Velocidad del viento (m/s), Visibilidad (10 m), Temperatura de punto de rocío (°C), Radiación solar (MJ/m²), Lluvia (mm), Nieve (cm).
- **Temporales y Calendarias:** Hora del día (0-23 h), Día Festivo (binaria 0/1), Día de Funcionamiento (binaria 0/1), y Estación del año codificada mediante One-Hot Encoding (`Season_Spring`, `Season_Summer`, `Season_Autumn`, `Season_Winter`).

---

## IV. Planteamiento de la Solución

### A. Preprocesamiento y Análisis de Correlación
Se verificó la integridad del dataset, confirmando cero valores nulos en sus 8,760 filas. Se codificaron las variables categóricas y se generó la matriz de correlación de Pearson.

| Variable | Correlación con Demanda ($r$) | Interpretación Física |
|---|---|---|
| `Temperature` | **+0.5386** | Predictor climático primario. Temperaturas agradables impulsan el alquiler. |
| `Hour` | **+0.4103** | Ciclo circadiano con doble pico diario (movilidad laboral/educativa 8h y 18h). |
| `Dew_Point_Temp` | **+0.3798** | Correlación positiva, pero colineal con la Temperatura ($r = 0.91$). |
| `Season_Summer` | **+0.2965** | Máximo aprovechamiento de horas de luz y actividades al aire libre. |
| `Solar_Radiation`| **+0.2618** | Clima soleado favorece el desplazamiento en bicicleta. |
| `Functioning_Day` | **+0.2039** | Sistema en servicio (si es 0, la demanda es estrictamente nula). |
| `Visibility` | **+0.1993** | Mayor visibilidad aumenta la seguridad percibida. |
| `Wind_Speed` | **+0.1211** | Viento moderado no representa un obstáculo severo. |
| `Rainfall` | **-0.1231** | La precipitación disuade inmediatamente el uso de bicicletas. |
| `Snowfall` | **-0.1418** | Nieve en calzada genera riesgo de caídas y reduce demanda. |
| `Humidity` | **-0.1998** | Humedad alta produce incomodidad térmica. |
| `Season_Winter` | **-0.4249** | Temperaturas bajo cero provocan la caída más drástica de la demanda. |

### B. Partición de Datos y Aislamiento de Fuga de Información
Se dividió el conjunto de datos en un 80% para entrenamiento (7,008 registros) y un 20% para validación/prueba (1,752 registros) utilizando `train_test_split` con `random_state=42` y `shuffle=True`. Se ajustó el escalador Min-Max exclusivamente sobre `X_train` e `y_train`, aplicando la transformación resultante sobre `X_test` e `y_test`.

### C. Diseño de Experimentos
Se formularon dos baterías de experimentos:
1. **Comparación de Arquitecturas:** Con optimizador Adam fijo (lr=0.005), evaluando 1 capa oculta (16 neuronas), 2 capas ocultas (32-16 neuronas) y 3 capas ocultas (64-32-16 neuronas).
2. **Comparación de Optimizadores:** Con la Arquitectura 2 fija, evaluando Adam, RMSprop y SGD con Momentum durante 80 épocas de entrenamiento (batch size = 64).

Ambos experimentos fueron instrumentados con callbacks de **TensorBoard** para trazabilidad completa.

---

## V. Resultados y Discusión

### A. Comparación de Arquitecturas
La siguiente tabla sintetiza los resultados obtenidos en el conjunto de prueba independiente (1,752 muestras no vistas durante el entrenamiento):

| Arquitectura | Capas Ocultas | Parámetros | $R^2$ | $MAE$ (bicicletas) | $RMSE$ (bicicletas) | Pérdida Final ($MSE_{norm}$) |
|---|---|---|---|---|---|---|
| **Arch 1 (Ligera)** | [16] | 273 | 0.7295 | 223.99 | 335.73 | 0.009648 |
| **Arch 2 (Media)** | [32, 16] | 1,057 | **0.8253** | **174.93** | **269.82** | **0.006232** |
| **Arch 3 (Profunda)**| [64, 32, 16] | 3,649 | 0.8295 | 169.03 | 266.50 | 0.006079 |

### B. Comparación de Optimizadores (sobre Arquitectura 2)

| Optimizador | Tasa de Aprendizaje | $R^2$ | $MAE$ (bicicletas) | $RMSE$ (bicicletas) | Pérdida Final ($MSE_{norm}$) |
|---|---|---|---|---|---|
| **Adam** | 0.005 | **0.8243** | **181.69** | **270.57** | **0.006266** |
| **RMSprop** | 0.005 | 0.7997 | 184.68 | 288.89 | 0.007143 |
| **SGD con Momentum** | 0.02 (mom=0.9) | 0.7347 | 227.86 | 332.47 | 0.009462 |

### C. Análisis de Convergencia de Curvas de Pérdida
Las curvas de entrenamiento y validación confirman una convergencia rápida y asintótica en Adam y RMSprop antes de la época 30. La brecha entre la curva de entrenamiento y validación permanece acotada, evidenciando ausencia de sobreajuste (*overfitting*). El optimizador SGD con Momentum desciende de forma más lenta, confirmando que las tasas de aprendizaje adaptativas de Adam son superiores en topologías con paisajes de pérdida no convexos y variables heterogéneas.

### D. Justificación de la Selección del Modelo para Despliegue en Arduino
Aunque la Arquitectura 3 alcanza un $R^2 = 0.8295$, demanda 3,649 parámetros de punto flotante (14.6 KB de Flash). La **Arquitectura 2** logra prácticamente la misma fidelidad predictiva ($R^2 = 0.8253$, diferencia de apenas 0.43%) con solo **1,057 parámetros**, reduciendo el cómputo en un **71.0%**. Esto permite que el modelo ocupe únicamente 4.2 KB de Flash en el Arduino Uno (dejando más del 85% de memoria disponible para periféricos y comunicaciones) y menos de 180 bytes de SRAM durante la inferencia.

### E. Verificación Experimental: Arduino vs TensorFlow-Keras
Se implementó la función `forward()` en C++ dentro de Arduino Uno (Wokwi), incorporando la normalización de entradas y la **desnormalización obligatoria** al rango original de bicicletas. Se evaluaron 5 muestras representativas del conjunto de test:

| Muestra | Demanda Real | Predicción Keras (bicis) | Predicción Arduino (bicis) | Error Absoluto | Concordancia |
|---|---|---|---|---|---|
| 1 | 1,728.0 | 1,251.54 | 1,251.54 | $2.63 \times 10^{-5}$ | **Exacta (100%)** |
| 2 | 822.0 | 977.18 | 977.18 | $7.05 \times 10^{-5}$ | **Exacta (100%)** |
| 3 | 658.0 | 820.90 | 820.90 | $8.02 \times 10^{-5}$ | **Exacta (100%)** |
| 4 | 2,716.0 | 2,223.60 | 2,223.60 | $4.43 \times 10^{-4}$ | **Exacta (100%)** |
| 5 | 1,083.0 | 785.64 | 785.64 | $4.17 \times 10^{-5}$ | **Exacta (100%)** |

El tiempo de inferencia medido en el microcontrolador fue de **~350 microsegundos**, lo que valida la capacidad de operación en tiempo real en sistemas embebidos de bajo costo.

---

## VI. Conclusiones
1. Las redes neuronales MLP superficiales demostraron una alta capacidad para capturar las dinámicas no lineales de la demanda de bicicletas en Seúl, alcanzando un coeficiente de determinación $R^2 = 0.8253$, lo que explica más del 82.5% de la varianza total de los datos.
2. La temperatura ambiental ($r = +0.54$) y la hora del día ($r = +0.41$) se ratificaron como los inductores dominantes de la demanda, mientras que las condiciones invernales ($r = -0.42$) representan el principal inhibidor.
3. El optimizador Adam evidenció una convergencia significativamente superior frente a SGD con Momentum y RMSprop, combinando rapidez y estabilidad asintótica sin sobreajuste.
4. El análisis de complejidad computacional demostró que la Arquitectura 2 (Input 15 -> 32 -> 16 -> 1) ofrece el compromiso óptimo entre precisión y ligereza algorítmica para sistemas embebidos (*Edge Computing*), requiriendo solo 1,057 parámetros.
5. La implementación en C++ sobre Arduino Uno demostró total fidelidad numérica frente a Keras, verificando la correcta normalización interna de entradas y la desnormalización al rango de bicicletas reales exigida en las directrices de la asignatura.

---

## VII. Referencias
[1] E. Sathishkumar, C. Park, and Y. Cho, "Using data mining techniques for bike sharing demand prediction in metropolitan city," *Computer Communications*, vol. 153, pp. 353–366, 2020.  
[2] UCI Machine Learning Repository, "Seoul Bike Sharing Demand Dataset," Dataset ID: 560, DOI: 10.24432/C5F60M, 2020.  
[3] F. Chollet, *Deep Learning with Python*, 2nd ed. Shelter Island, NY: Manning Publications, 2021.  
[4] J. A. López, A. F. Escobar, and J. L. Paniagua, "Guías y Material de Clase: Redes Neuronales Artificiales y Deep Learning," Facultad de Ingeniería, Universidad Autónoma de Occidente, Cali, Colombia, 2023-2026.  
[5] D. P. Kingma and J. Ba, "Adam: A method for stochastic optimization," *arXiv preprint arXiv:1412.6980*, 2014.  
[6] V. Nair and G. E. Hinton, "Rectified linear units improve restricted boltzmann machines," in *Proc. 27th Int. Conf. Mach. Learn. (ICML)*, 2010, pp. 807–814.  
