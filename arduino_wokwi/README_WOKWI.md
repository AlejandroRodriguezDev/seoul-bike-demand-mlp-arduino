# Guía de Simulación en Wokwi (Arduino Uno)

Este directorio contiene la implementación en C++ de la Red Neuronal Multicapa (MLP) entrenada en TensorFlow-Keras para predecir la demanda de bicicletas en Seúl, lista para ser emulada sin necesidad de hardware físico.

---

## Archivos del Proyecto Arduino

1. **`seoul_bike_mlp.ino`**: Código principal en C/C++ para Arduino Uno (ATmega328P).
   - Realiza la **propagación hacia adelante (forward pass)** en tiempo real.
   - Aplica activación **ReLU** en capas ocultas y activación **Lineal** en la capa de salida.
   - Realiza la **normalización Min-Max** de las 15 variables de entrada recibidas en escala original.
   - Realiza la **DESNORMALIZACIÓN OBLIGATORIA** de la predicción para entregar el número de bicicletas reales por hora.
2. **`weights.h`**: Cabecera en C generada automáticamente por `train_models.py` conteniendo:
   - Constantes `x_min` y `x_max` de cada variable.
   - Parámetros `y_min` e `y_max` para desnormalización.
   - Matrices de pesos $W_1, W_2, W_3$ y vectores de sesgo $b_1, b_2, b_3$.
3. **`diagram.json`**: Esquema de simulación para Wokwi (Arduino Uno).

---

## ¿Cómo Ejecutar en Wokwi Web (Paso a Paso)?

1. Entra a [https://wokwi.com/projects/new/arduino-uno](https://wokwi.com/projects/new/arduino-uno).
2. En la pestaña **`sketch.ino`**, borra el contenido por defecto y pega todo el código de [`seoul_bike_mlp.ino`](file:///c:/Users/aleja/OneDrive/Escritorio/Redes%20Neuronales/proyecto_seoul_bike/arduino_wokwi/seoul_bike_mlp.ino).
3. Haz clic en la pestaña con el símbolo **`+`** (New File) en la esquina superior izquierda del editor de código de Wokwi y nombra el nuevo archivo: `weights.h`.
4. Abre [`weights.h`](file:///c:/Users/aleja/OneDrive/Escritorio/Redes%20Neuronales/proyecto_seoul_bike/arduino_wokwi/weights.h), copia todo su contenido y pégalo en la pestaña `weights.h` de Wokwi.
5. Haz clic en el botón verde **"Play" / "Start Simulation"**.
6. Abre el **Serial Monitor** en la parte inferior a una velocidad de **115200 baudios**.
7. Verás la salida inmediata de las 5 muestras de prueba con:
   - Características de entrada en su escala original.
   - Predicción normalizada.
   - **Predicción final desnormalizada (bicicletas/hora)**.
   - Tiempo de inferencia en microsegundos (~350 microsegundos en Arduino Uno).

---

## ¿Cómo Sustentar ante el Profesor?

Durante la sustentación (Criterio 11 de la rúbrica):
- Muestra el Serial Monitor corriendo en Wokwi.
- Explica la función `forward()`: muestra dónde se normalizan las entradas (`(raw_inputs[i] - x_min[i]) / (x_max[i] - x_min[i])`), cómo se multiplican por los pesos de Keras, cómo se aplica ReLU `max(0, z)` y cómo se **desnormaliza** la salida final multiplicando por `(y_max - y_min) + y_min`.
- Muestra que el resultado numérico es idéntico al de Keras con un error menor a $10^{-4}$ (diferencia atribuible solo a la precisión de punto flotante de 32 bits).
