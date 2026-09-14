"""
Script principal de entrenamiento para el miniproyecto:
Redes MLP aplicadas a problemas de regresión (Seoul Bike Sharing Demand)
Autores: Grupo de Redes Neuronales y Deep Learning
Universidad Autónoma de Occidente
"""

import os
import sys
import json
import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import TensorBoard
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

# Fijar semillas para reproducibilidad
SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)

# Rutas del proyecto
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, 'data', 'SeoulBikeData.csv')
MODELS_DIR = os.path.join(BASE_DIR, 'models')
LOGS_DIR = os.path.join(BASE_DIR, 'logs')
INFORME_DIR = os.path.join(BASE_DIR, 'informe')
ARDUINO_DIR = os.path.join(BASE_DIR, 'arduino_wokwi')

os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(LOGS_DIR, exist_ok=True)
os.makedirs(INFORME_DIR, exist_ok=True)
os.makedirs(ARDUINO_DIR, exist_ok=True)

print("=" * 70)
print("1. CARGA Y ENTENDIMIENTO DEL DATASET")
print("=" * 70)

# Carga de datos
try:
    df = pd.read_csv(DATA_PATH, encoding='utf-8')
except UnicodeDecodeError:
    df = pd.read_csv(DATA_PATH, encoding='latin-1')

print(f"Dimensiones iniciales del dataset: {df.shape[0]} filas, {df.shape[1]} columnas.")

# Renombrar columnas para manejo estándar y limpio
column_names = [
    'Date', 'Rented_Bike_Count', 'Hour', 'Temperature', 'Humidity',
    'Wind_Speed', 'Visibility', 'Dew_Point_Temp', 'Solar_Radiation',
    'Rainfall', 'Snowfall', 'Seasons', 'Holiday', 'Functioning_Day'
]
df.columns = column_names

print("\nPrimeras 3 filas:")
print(df.head(3))

print("\nVerificación de valores nulos:")
print(df.isnull().sum())

# Codificación de variables categóricas
df['Holiday_Binary'] = (df['Holiday'] == 'Holiday').astype(int)
df['Functioning_Binary'] = (df['Functioning_Day'] == 'Yes').astype(int)

# One-hot encoding para Seasons (Spring, Summer, Autumn, Winter)
seasons_dummies = pd.get_dummies(df['Seasons'], prefix='Season', dtype=int)
df_processed = pd.concat([df, seasons_dummies], axis=1)

# Variables predictoras (X) y variable objetivo (y)
feature_cols = [
    'Hour', 'Temperature', 'Humidity', 'Wind_Speed', 'Visibility',
    'Dew_Point_Temp', 'Solar_Radiation', 'Rainfall', 'Snowfall',
    'Holiday_Binary', 'Functioning_Binary',
    'Season_Spring', 'Season_Summer', 'Season_Autumn', 'Season_Winter'
]
target_col = 'Rented_Bike_Count'

X = df_processed[feature_cols]
y = df_processed[[target_col]]

print(f"\nVariables predictoras seleccionadas ({len(feature_cols)}): {feature_cols}")
print(f"Variable objetivo: {target_col}")

print("\n" + "=" * 70)
print("2. ANÁLISIS EXPLORATORIO Y MATRIZ DE CORRELACIÓN")
print("=" * 70)

# Matriz de correlación
df_corr = pd.concat([y, X], axis=1).corr()
print("\nCorrelación lineal de Pearson con Rented_Bike_Count:")
print(df_corr[target_col].sort_values(ascending=False))

# Guardar figura de la matriz de correlación
plt.figure(figsize=(14, 10))
sns.heatmap(df_corr, annot=True, fmt=".2f", cmap='coolwarm', cbar=True, square=True)
plt.title('Matriz de Correlación de Pearson - Seoul Bike Sharing Demand', fontsize=14, pad=15)
plt.tight_layout()
corr_path = os.path.join(INFORME_DIR, 'matriz_correlacion.png')
plt.savefig(corr_path, dpi=300)
plt.close()
print(f"Gráfico de matriz de correlación guardado en: {corr_path}")

print("\n" + "=" * 70)
print("3. PARTICIÓN DE DATOS Y NORMALIZACIÓN (MinMaxScaler)")
print("=" * 70)

# Partición train_test_split (80% entrenamiento, 20% prueba/validación)
X_train_raw, X_test_raw, y_train_raw, y_test_raw = train_test_split(
    X.values, y.values, test_size=0.2, random_state=SEED, shuffle=True
)

print(f"Conjunto de entrenamiento: {X_train_raw.shape[0]} muestras")
print(f"Conjunto de prueba: {X_test_raw.shape[0]} muestras")

# Normalización Min-Max al rango [0, 1]
# IMPORTANTE: fit_transform exclusivamente en train para evitar data leakage
scaler_X = MinMaxScaler(feature_range=(0, 1))
X_train = scaler_X.fit_transform(X_train_raw)
X_test = scaler_X.transform(X_test_raw)

scaler_y = MinMaxScaler(feature_range=(0, 1))
y_train = scaler_y.fit_transform(y_train_raw)
y_test = scaler_y.transform(y_test_raw)

print("Normalización MinMax completada.")
print(f"X_train rango: [{X_train.min():.2f}, {X_train.max():.2f}]")
print(f"y_train rango: [{y_train.min():.2f}, {y_train.max():.2f}]")

input_dim = X_train.shape[1]

# Guardar parámetros de normalización para Arduino
norm_params = {
    'features': feature_cols,
    'x_min': scaler_X.data_min_.tolist(),
    'x_max': scaler_X.data_max_.tolist(),
    'y_min': float(scaler_y.data_min_[0]),
    'y_max': float(scaler_y.data_max_[0])
}
norm_params_path = os.path.join(ARDUINO_DIR, 'norm_params.json')
with open(norm_params_path, 'w') as f:
    json.dump(norm_params, f, indent=2)
print(f"Parámetros de normalización guardados en: {norm_params_path}")

print("\n" + "=" * 70)
print("4. DEFINICIÓN DE ARQUITECTURAS Y OPTIMIZADORES")
print("=" * 70)

from tensorflow.keras.layers import Dense, Input

# Funciones generadoras de arquitecturas (máximo 3 capas ocultas, salida LINEAL para regresión)
def build_arch_1(input_shape):
    """Arquitectura 1: Ligera (1 capa oculta) - Ideal para microcontroladores simples"""
    model = Sequential([
        Input(shape=(input_shape,)),
        Dense(16, activation='relu', name='oculta_1'),
        Dense(1, activation='linear', name='salida_lineal')
    ], name='MLP_Arch1_Ligera')
    return model

def build_arch_2(input_shape):
    """Arquitectura 2: Media (2 capas ocultas) - Balance ideal precisión / memoria"""
    model = Sequential([
        Input(shape=(input_shape,)),
        Dense(32, activation='relu', name='oculta_1'),
        Dense(16, activation='relu', name='oculta_2'),
        Dense(1, activation='linear', name='salida_lineal')
    ], name='MLP_Arch2_Media')
    return model

def build_arch_3(input_shape):
    """Arquitectura 3: Profunda (3 capas ocultas) - Mayor capacidad de representación"""
    model = Sequential([
        Input(shape=(input_shape,)),
        Dense(64, activation='relu', name='oculta_1'),
        Dense(32, activation='relu', name='oculta_2'),
        Dense(16, activation='relu', name='oculta_3'),
        Dense(1, activation='linear', name='salida_lineal')
    ], name='MLP_Arch3_Profunda')
    return model

EPOCHS = 80
BATCH_SIZE = 64

# ==============================================================================
# EXPERIMENTO 1: COMPARACIÓN DE 3 ARQUITECTURAS (usando optimizador Adam)
# ==============================================================================
print("\n---> EXPERIMENTO 1: Comparación de 3 Arquitecturas con Adam")

architectures = {
    'Arch1_Ligera (1 capa: 16)': build_arch_1(input_dim),
    'Arch2_Media (2 capas: 32-16)': build_arch_2(input_dim),
    'Arch3_Profunda (3 capas: 64-32-16)': build_arch_3(input_dim)
}

histories_arch = {}
results_arch = {}

for name, model in architectures.items():
    print(f"\nEntrenando {name}...")
    model.summary()
    opt = tf.keras.optimizers.Adam(learning_rate=0.005)
    model.compile(optimizer=opt, loss='mean_squared_error', metrics=['mean_absolute_error'])
    
    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    log_dir = os.path.join(LOGS_DIR, f"arch_{name.split()[0]}_{timestamp}")
    tb_cb = TensorBoard(log_dir=log_dir, histogram_freq=0, write_graph=True)
    
    h = model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        verbose=1,
        callbacks=[tb_cb]
    )
    histories_arch[name] = h.history
    
    # Evaluación en test
    y_pred_norm = model.predict(X_test, verbose=0)
    y_pred_real = scaler_y.inverse_transform(y_pred_norm)
    
    r2 = r2_score(y_test_raw, y_pred_real)
    mae = mean_absolute_error(y_test_raw, y_pred_real)
    mse = mean_squared_error(y_test_raw, y_pred_real)
    rmse = np.sqrt(mse)
    
    results_arch[name] = {
        'R2': float(r2),
        'MAE': float(mae),
        'MSE': float(mse),
        'RMSE': float(rmse),
        'Val_Loss_Final': float(h.history['val_loss'][-1]),
        'Params': int(model.count_params())
    }
    print(f"Resultado {name}: R2 = {r2:.4f}, MAE = {mae:.2f} bicis, RMSE = {rmse:.2f} bicis")

# ==============================================================================
# EXPERIMENTO 2: COMPARACIÓN DE 3 OPTIMIZADORES (usando Arquitectura 2 Media)
# ==============================================================================
print("\n---> EXPERIMENTO 2: Comparación de 3 Optimizadores en Arquitectura 2")

optimizers_dict = {
    'Adam': tf.keras.optimizers.Adam(learning_rate=0.005),
    'RMSprop': tf.keras.optimizers.RMSprop(learning_rate=0.005),
    'SGD_Momentum': tf.keras.optimizers.SGD(learning_rate=0.02, momentum=0.9)
}

histories_opt = {}
results_opt = {}
trained_models = {}

for opt_name, opt_instance in optimizers_dict.items():
    print(f"\nEntrenando Arquitectura 2 con optimizador {opt_name}...")
    model = build_arch_2(input_dim)
    model.compile(optimizer=opt_instance, loss='mean_squared_error', metrics=['mean_absolute_error'])
    
    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    log_dir = os.path.join(LOGS_DIR, f"opt_{opt_name}_{timestamp}")
    tb_cb = TensorBoard(log_dir=log_dir, histogram_freq=0, write_graph=True)
    
    h = model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        verbose=1,
        callbacks=[tb_cb]
    )
    histories_opt[opt_name] = h.history
    trained_models[opt_name] = model
    
    y_pred_norm = model.predict(X_test, verbose=0)
    y_pred_real = scaler_y.inverse_transform(y_pred_norm)
    
    r2 = r2_score(y_test_raw, y_pred_real)
    mae = mean_absolute_error(y_test_raw, y_pred_real)
    mse = mean_squared_error(y_test_raw, y_pred_real)
    rmse = np.sqrt(mse)
    
    results_opt[opt_name] = {
        'R2': float(r2),
        'MAE': float(mae),
        'MSE': float(mse),
        'RMSE': float(rmse),
        'Val_Loss_Final': float(h.history['val_loss'][-1])
    }
    print(f"Resultado Optimizador {opt_name}: R2 = {r2:.4f}, MAE = {mae:.2f} bicis, RMSE = {rmse:.2f} bicis")

print("\n" + "=" * 70)
print("5. GRÁFICAS DE CURVAS DE PÉRDIDA (LOSS) Y VALIDACIÓN")
print("=" * 70)

fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Subplot 1: Pérdida por Arquitectura
for name, h in histories_arch.items():
    axes[0].plot(h['loss'], label=f"Train - {name.split()[0]}")
    axes[0].plot(h['val_loss'], linestyle='--', label=f"Val - {name.split()[0]}")
axes[0].set_title('Convergencia de Pérdida (MSE) por Arquitectura', fontsize=12)
axes[0].set_xlabel('Época')
axes[0].set_ylabel('Mean Squared Error (Normalizado)')
axes[0].legend(fontsize=9)
axes[0].grid(True, alpha=0.3)

# Subplot 2: Pérdida por Optimizador
for opt_name, h in histories_opt.items():
    axes[1].plot(h['loss'], label=f"Train - {opt_name}")
    axes[1].plot(h['val_loss'], linestyle='--', label=f"Val - {opt_name}")
axes[1].set_title('Convergencia de Pérdida (MSE) por Optimizador (Arch 2)', fontsize=12)
axes[1].set_xlabel('Época')
axes[1].set_ylabel('Mean Squared Error (Normalizado)')
axes[1].legend(fontsize=9)
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
loss_path = os.path.join(INFORME_DIR, 'curvas_perdida.png')
plt.savefig(loss_path, dpi=300)
plt.close()
print(f"Gráfico de curvas de pérdida guardado en: {loss_path}")

# ==============================================================================
# 6. SELECCIÓN DEL MEJOR MODELO Y EXPORTACIÓN
# ==============================================================================
print("\n" + "=" * 70)
print("6. SELECCIÓN DEL MEJOR MODELO PARA ARDUINO")
print("=" * 70)

best_model = trained_models['Adam']
best_model_path = os.path.join(MODELS_DIR, 'mejor_modelo_seoul_bike.keras')
best_model.save(best_model_path)
print(f"Mejor modelo guardado en: {best_model_path}")

# Gráfico de Dispersión: Predicción vs Real (escala original)
y_pred_best = best_model.predict(X_test, verbose=0)
y_pred_best_real = scaler_y.inverse_transform(y_pred_best)

plt.figure(figsize=(8, 8))
plt.scatter(y_test_raw, y_pred_best_real, alpha=0.3, color='royalblue', edgecolors='none', label='Datos de prueba')
max_val = max(y_test_raw.max(), y_pred_best_real.max())
plt.plot([0, max_val], [0, max_val], 'r--', lw=2, label='Ajuste Ideal (y = x)')
plt.title('Predicción vs Valor Real (Demanda de Bicicletas en Seúl)', fontsize=13)
plt.xlabel('Demanda Real (Bicicletas / hora)')
plt.ylabel('Demanda Predicha por Keras (Bicicletas / hora)')
plt.legend()
plt.grid(True, alpha=0.3)
parity_path = os.path.join(INFORME_DIR, 'dispersion_prediccion_vs_real.png')
plt.savefig(parity_path, dpi=300)
plt.close()
print(f"Gráfico de paridad guardado en: {parity_path}")

# ==============================================================================
# 7. EXPORTACIÓN DE PESOS Y SESGOS A C/C++ (weights.h)
# ==============================================================================
print("\n" + "=" * 70)
print("7. EXPORTACIÓN DE PESOS Y SESGOS PARA ARDUINO (weights.h)")
print("=" * 70)

header_content = """// ===================================================================
// weights.h - Pesos, sesgos y normalizacion exportados desde Keras
// Modelo: Arquitectura 2 (Input 15 -> Dense 32 -> Dense 16 -> Dense 1)
// Generado automaticamente para Arduino / Wokwi
// ===================================================================

#ifndef WEIGHTS_H
#define WEIGHTS_H

#define NUM_INPUTS 15
#define NUM_HIDDEN1 32
#define NUM_HIDDEN2 16
#define NUM_OUTPUTS 1

// --- PARAMETROS DE NORMALIZACION MIN-MAX ---
"""

x_min_str = ", ".join([f"{float(v):.6f}f" for v in scaler_X.data_min_])
x_max_str = ", ".join([f"{float(v):.6f}f" for v in scaler_X.data_max_])
header_content += f"const float x_min[NUM_INPUTS] = {{{x_min_str}}};\n"
header_content += f"const float x_max[NUM_INPUTS] = {{{x_max_str}}};\n\n"

header_content += f"const float y_min = {float(scaler_y.data_min_[0]):.6f}f;\n"
header_content += f"const float y_max = {float(scaler_y.data_max_[0]):.6f}f;\n\n"

dense_layers = [l for l in best_model.layers if isinstance(l, Dense)]

for idx, layer in enumerate(dense_layers):
    w, b = layer.get_weights()
    rows, cols = w.shape
    print(f"Capa {idx+1} ({layer.name}): W = {w.shape}, b = {b.shape}")
    
    header_content += f"// Pesos Capa {idx+1}: {layer.name} ({rows}x{cols})\n"
    header_content += f"const float W{idx+1}[{rows}][{cols}] = {{\n"
    for r in range(rows):
        row_str = ", ".join([f"{float(val):.6f}f" for val in w[r]])
        header_content += f"  {{{row_str}}},\n"
    header_content += "};\n\n"
    
    b_str = ", ".join([f"{float(val):.6f}f" for val in b])
    header_content += f"// Sesgos Capa {idx+1}: {layer.name} ({cols})\n"
    header_content += f"const float b{idx+1}[{cols}] = {{{b_str}}};\n\n"

header_content += "#endif // WEIGHTS_H\n"

weights_header_path = os.path.join(ARDUINO_DIR, 'weights.h')
with open(weights_header_path, 'w', encoding='utf-8') as f:
    f.write(header_content)
print(f"Archivo weights.h generado con éxito en: {weights_header_path}")

# ==============================================================================
# 8. GUARDAR MUESTRAS DE PRUEBA PARA VALIDACIÓN KERAS VS ARDUINO
# ==============================================================================
test_samples_raw = X_test_raw[:5]
test_samples_norm = X_test[:5]
keras_pred_norm = best_model.predict(test_samples_norm, verbose=0)
keras_pred_real = scaler_y.inverse_transform(keras_pred_norm)
actual_y_real = y_test_raw[:5]

verification_data = []
for i in range(5):
    verification_data.append({
        'sample_index': i + 1,
        'features_raw': [float(v) for v in test_samples_raw[i]],
        'actual_rented_bikes': float(actual_y_real[i][0]),
        'keras_predicted_norm': float(keras_pred_norm[i][0]),
        'keras_predicted_real': float(keras_pred_real[i][0])
    })

verif_path = os.path.join(ARDUINO_DIR, 'verification_samples.json')
with open(verif_path, 'w', encoding='utf-8') as f:
    json.dump(verification_data, f, indent=2)
print(f"Datos de verificación guardados en: {verif_path}")

print("\n" + "=" * 70)
print("RESUMEN DE MÉTRICAS OBTENIDAS:")
print("=" * 70)
print("\n--- Comparación de Arquitecturas (con Adam) ---")
df_res_arch = pd.DataFrame(results_arch).T
print(df_res_arch[['R2', 'MAE', 'RMSE', 'Params', 'Val_Loss_Final']])

print("\n--- Comparación de Optimizadores (Arquitectura 2) ---")
df_res_opt = pd.DataFrame(results_opt).T
print(df_res_opt[['R2', 'MAE', 'RMSE', 'Val_Loss_Final']])

metrics_summary = {
    'architectures': results_arch,
    'optimizers': results_opt
}
summary_path = os.path.join(INFORME_DIR, 'metricas_resumen.json')
with open(summary_path, 'w', encoding='utf-8') as f:
    json.dump(metrics_summary, f, indent=2)
print(f"\nResumen de métricas guardado en: {summary_path}")
print("\n¡Entrenamiento y exportación finalizados exitosamente!")
