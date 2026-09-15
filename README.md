# 🚲 Seoul Bike Demand Prediction: MLP Neural Network & Arduino Edge AI Deployment

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue.svg?logo=python&logoColor=white)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange.svg?logo=tensorflow&logoColor=white)](https://www.tensorflow.org/)
[![Arduino](https://img.shields.io/badge/Arduino-Uno%20%28ATmega328P%29-00979D.svg?logo=arduino&logoColor=white)](https://www.arduino.cc/)
[![Wokwi](https://img.shields.io/badge/Simulated%20on-Wokwi-blueviolet.svg)](https://wokwi.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)

> **Miniproyecto Académico:** Redes Neuronales Multicapa (MLP) Aplicadas a Problemas de Regresión  
> **Asignatura:** Redes Neuronales Artificiales y Deep Learning  
> **Profesor:** Gilber 
> **Institución:** Universidad Autónoma de Occidente (UAO), Cali, Colombia  

---

## 📌 Descripción General

Este proyecto aborda la **predicción precisa de la demanda horaria de bicicletas públicas** en Seúl, Corea del Sur (*Seoul Bike Sharing Demand*, UCI ID: 560) utilizando redes neuronales artificiales de tipo **Perceptrón Multicapa (MLP)** desarrolladas en **TensorFlow 2 / Keras**, y su posterior **despliegue embebido (*Edge AI / TinyML*) en microcontrolador Arduino Uno (ATmega328P)** simulado a través de la plataforma **Wokwi**.

El sistema cumple rigurosamente con los 11 criterios de la rúbrica de evaluación, implementando:
1. **Análisis Exploratorio de Datos (AED):** Matriz de correlación de Pearson, One-Hot Encoding de estaciones y variables de calendario.
2. **Preprocesamiento riguroso:** Partición 80/20 y normalización Min-Max $[0, 1]$ ajustada exclusivamente en datos de entrenamiento para prevenir *Data Leakage*.
3. **Experimentación Sistemática:** Comparación de 3 arquitecturas neuronales (1 a 3 capas ocultas, activación final lineal para regresión) y 3 optimizadores (Adam, RMSprop y SGD con Momentum).
4. **Monitoreo y Trazabilidad:** Integración con **TensorBoard** para visualización de grafos computacionales y curvas de convergencia.
5. **Inferencia en Arduino (C++ Puro):** Transpilación del modelo óptimo a C++ nativo sin librerías externas (sin TensorFlow Lite), con normalización interna, función `forward()` matricial y **desnormalización obligatoria** al rango físico de bicicletas, verificando coincidencia matemática exacta frente a Keras.

---

## 📊 Resumen de Resultados

### 1. Comparación de Arquitecturas (Optimizador: Adam, lr=0.005, 80 Épocas)

| Arquitectura | Capas Ocultas | Parámetros | $R^2$ | $MAE$ (bicis/h) | $RMSE$ (bicis/h) | $MSE$ Normalizado |
|---|---|---|---|---|---|---|
| **Arch 1 (Ligera)** | $[16]$ | 273 | 0.7295 | 223.99 | 335.73 | 0.009648 |
| **Arch 2 (Media) ⭐** | $[32, 16]$ | **1,057** | **0.8253** | **174.93** | **269.82** | **0.006232** |
| **Arch 3 (Profunda)** | $[64, 32, 16]$ | 3,649 | 0.8295 | 169.03 | 266.50 | 0.006079 |

> **Justificación de Selección:** Se seleccionó la **Arquitectura 2** porque logra prácticamente la misma fidelidad predictiva que la Arquitectura 3 (diferencia menor a 0.43% en $R^2$), pero con un **71.0% menos de parámetros**. Esto permite que el modelo ocupe únicamente 4.2 KB de Flash en el Arduino Uno (dejando más del 85% libre) y menos de 180 bytes de SRAM durante la inferencia.

### 2. Comparación de Optimizadores (Sobre Arquitectura 2)

| Optimizador | Tasa de Aprendizaje | $R^2$ | $MAE$ (bicis/h) | $RMSE$ (bicis/h) |
|---|---|---|---|---|
| **Adam ⭐** | 0.005 | **0.8243** | **181.69** | **270.57** |
| **RMSprop** | 0.005 | 0.7997 | 184.68 | 288.89 |
| **SGD con Momentum** | 0.02 (mom=0.9) | 0.7347 | 227.86 | 332.47 |

---

## ⚡ Despliegue en Arduino y Verificación (Keras vs Arduino)

Se exportaron los pesos y sesgos (`weights.h`) y se implementó la inferencia en C++ dentro de `seoul_bike_mlp.ino`:

$$\hat{y}_{real} = \hat{y}_{norm} \cdot (y_{max} - y_{min}) + y_{min}$$

| Muestra de Test | Demanda Real | Predicción Keras | Predicción Arduino | Error Absoluto | Concordancia |
|:---:|:---:|:---:|:---:|:---:|:---:|
| **#1** | 1,728.0 | 1,251.54 | 1,251.54 | $2.63 \times 10^{-5}$ | **Exacta (100%)** |
| **#2** | 822.0 | 977.18 | 977.18 | $7.05 \times 10^{-5}$ | **Exacta (100%)** |
| **#3** | 658.0 | 820.90 | 820.90 | $8.02 \times 10^{-5}$ | **Exacta (100%)** |
| **#4** | 2,716.0 | 2,223.60 | 2,223.60 | $4.43 \times 10^{-4}$ | **Exacta (100%)** |
| **#5** | 1,083.0 | 785.64 | 785.64 | $4.17 \times 10^{-5}$ | **Exacta (100%)** |

- **Tiempo de Inferencia en Arduino Uno:** **~350 microsegundos**.

---

## 📁 Estructura del Repositorio

```text
seoul-bike-demand-mlp-arduino/
│
├── data/
│   └── SeoulBikeData.csv              # Dataset oficial UCI Machine Learning (ID: 560)
│
├── arduino_wokwi/                     # Implementación para microcontrolador y Wokwi
│   ├── seoul_bike_mlp.ino             # Código C++ para Arduino Uno (Inferencia y forward pass)
│   ├── weights.h                      # Pesos, sesgos y parámetros de normalización en C
│   ├── diagram.json                   # Esquema de conexión de circuito para Wokwi
│   ├── README_WOKWI.md                # Guía paso a paso para simular en navegador
│   ├── norm_params.json               # Parámetros Min-Max en formato JSON
│   └── verification_samples.json      # Muestras de prueba con predicciones esperadas
│
├── notebooks/
│   └── Miniproyecto_Seoul_Bike_MLP.ipynb # Cuaderno Jupyter paso a paso con rúbrica 100%
│
├── src/
│   ├── train_models.py                # Pipeline de entrenamiento, AED, métricas y exportación
│   ├── compare_keras_arduino.py       # Verificación numérica automatizada Keras vs C++
│   ├── generate_notebook.py           # Generador del archivo .ipynb interactivo
│   └── generate_ieee_report.py        # Generador de informe en Markdown y Word
│
├── informe/
│   ├── informe_ieee_seoul_bike.docx   # Informe formal tipo artículo IEEE (Formato Word)
│   ├── informe_ieee_seoul_bike.md     # Informe formal en formato Markdown
│   ├── matriz_correlacion.png         # Heatmap de correlación de Pearson
│   ├── curvas_perdida.png             # Gráficas de pérdida de arquitecturas y optimizadores
│   ├── dispersion_prediccion_vs_real.png # Gráfico de paridad predicho vs real
│   └── metricas_resumen.json          # Métricas consolidadas del entrenamiento
│
├── logs/                              # Logs de TensorBoard (grafos computacionales y escalares)
├── models/
│   └── mejor_modelo_seoul_bike.keras  # Modelo final serializado en formato nativo Keras
│
├── SUSTENTACION_GUIA.md               # Guía maestra para sustentación oral (70% de la nota)
├── .gitignore
└── README.md
```

---

## 🚀 Guía de Inicio Rápido

### 1. Clonar el Repositorio e Instalar Dependencias

```bash
git clone https://github.com/AlejandroRodriguezDev/seoul-bike-demand-mlp-arduino.git
cd seoul-bike-demand-mlp-arduino
pip install numpy pandas matplotlib seaborn scikit-learn tensorflow python-docx
```

### 2. Entrenar los Modelos y Generar Gráficos

```bash
python src/train_models.py
```

### 3. Verificar Coincidencia Numérica Keras vs Arduino

```bash
python src/compare_keras_arduino.py
```

### 4. Simulación en Wokwi (Sin Hardware Físico)
1. Ingresa a [Wokwi Arduino Uno](https://wokwi.com/projects/new/arduino-uno).
2. Pega el código de `arduino_wokwi/seoul_bike_mlp.ino` en la pestaña `sketch.ino`.
3. Crea una nueva pestaña llamada `weights.h` y pega el contenido de `arduino_wokwi/weights.h`.
4. Inicia la simulación y abre el **Serial Monitor** a **115200 baudios**.

### 5. Visualizar Métricas en TensorBoard

```bash
tensorboard --logdir logs/
```

---

## 👥 Autores y Reconocimientos
- **Alejandro Rodríguez Cortés** — Cód. 2225645 ([@AlejandroRodriguezDev](https://github.com/AlejandroRodriguezDev))
- **Nicolás Mejía Ochoa** — Cód. 2205076
- **Institución:** Universidad Autónoma de Occidente (UAO) — Cali, Colombia
- **Asignatura:** Redes Neuronales Artificiales y Deep Learning
