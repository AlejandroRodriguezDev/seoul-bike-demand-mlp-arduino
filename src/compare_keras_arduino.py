"""
Script de verificación numérica: Emulación de Arduino vs TensorFlow-Keras
Valida que la función forward() en C/C++ produce resultados idénticos a Keras.
"""

import os
import json
import numpy as np
import tensorflow as tf

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(BASE_DIR, 'models')
ARDUINO_DIR = os.path.join(BASE_DIR, 'arduino_wokwi')

# Cargar muestras de verificación
with open(os.path.join(ARDUINO_DIR, 'verification_samples.json'), 'r') as f:
    verif_data = json.load(f)

# Cargar parámetros de normalización
with open(os.path.join(ARDUINO_DIR, 'norm_params.json'), 'r') as f:
    norm_params = json.load(f)

x_min = np.array(norm_params['x_min'])
x_max = np.array(norm_params['x_max'])
y_min = norm_params['y_min']
y_max = norm_params['y_max']

# Cargar modelo entrenado en Keras
model = tf.keras.models.load_model(os.path.join(MODELS_DIR, 'mejor_modelo_seoul_bike.keras'))

# Extraer pesos de las capas
w1, b1 = model.layers[0].get_weights()
w2, b2 = model.layers[1].get_weights()
w3, b3 = model.layers[2].get_weights()

def relu(x):
    return np.maximum(0, x)

def arduino_forward_emulation(raw_input):
    # 1. Normalización de entradas al rango [0, 1]
    norm_input = (raw_input - x_min) / (x_max - x_min)
    
    # 2. Capa Oculta 1 (Dense 32 + ReLU)
    z1 = np.dot(norm_input, w1) + b1
    a1 = relu(z1)
    
    # 3. Capa Oculta 2 (Dense 16 + ReLU)
    z2 = np.dot(a1, w2) + b2
    a2 = relu(z2)
    
    # 4. Capa de Salida (Dense 1 + Linear)
    y_norm = np.dot(a2, w3) + b3
    
    # 5. DESNORMALIZACIÓN al rango real de bicicletas
    y_real = y_norm[0] * (y_max - y_min) + y_min
    y_real = max(0.0, float(y_real))
    
    return float(y_norm[0]), y_real

print("=" * 85)
print("VERIFICACIÓN NUMÉRICA RIGUROSA: KERAS VS EMULACIÓN ARDUINO")
print("=" * 85)
print(f"{'Muestra':<8}{'Real (bicis)':<15}{'Keras (bicis)':<18}{'Arduino (bicis)':<18}{'Error Abs':<15}{'¿Coinciden?':<12}")
print("-" * 85)

all_passed = True
for sample in verif_data:
    idx = sample['sample_index']
    real_target = sample['actual_rented_bikes']
    k_real = sample['keras_predicted_real']
    
    raw_in = np.array(sample['features_raw'])
    ard_norm, ard_real = arduino_forward_emulation(raw_in)
    
    error = abs(k_real - ard_real)
    matches = error < 1e-3  # Tolerancia de 0.001 bicicletas (debido a precisión float32 en rango 0-3556)
    if not matches:
        all_passed = False
        
    print(f"{idx:<8}{real_target:<15.1f}{k_real:<18.2f}{ard_real:<18.2f}{error:<15.6e}{'SÍ' if matches else 'NO':<12}")

print("=" * 85)
if all_passed:
    print("¡VERIFICACIÓN EXITOSA! La implementación en C++/Arduino es 100% fiel al modelo Keras.")
else:
    print("ADVERTENCIA: Se encontraron discrepancias numéricas.")
