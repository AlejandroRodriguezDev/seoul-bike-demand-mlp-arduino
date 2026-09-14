/*
  =============================================================================
  PROYECTO: REDES MULTICAPA APLICADAS A PROBLEMAS DE REGRESIÓN
  Dataset: Seoul Bike Sharing Demand (Predicción de Demanda Horaria)
  Asignatura: Redes Neuronales y Deep Learning
  Autores: Alejandro Rodríguez Cortés (2225645), Nicolás Mejía Ochoa (2205076)
  Profesor: Jesús Alfonso López - Universidad Autónoma de Occidente
  =============================================================================
  
  DESCRIPCIÓN DE LA IMPLEMENTACIÓN EN ARDUINO / WOKWI:
  - Red Neuronal MLP feedforward con 2 capas ocultas (Input 15 -> 32 -> 16 -> 1).
  - Activaciones: ReLU en capas ocultas f(z) = max(0, z), Lineal en capa de salida.
  - Normalización interna de entradas en escala original mediante x_min y x_max.
  - DESNORMALIZACIÓN OBLIGATORIA de la salida al rango original de bicicletas:
      Demanda_Real = y_norm * (y_max - y_min) + y_min
  - Verificación directa y comparación frente a las predicciones de TensorFlow-Keras.
*/

#include "weights.h"

// Arreglos intermedios de activación
float a_in[NUM_INPUTS];
float a_h1[NUM_HIDDEN1];
float a_h2[NUM_HIDDEN2];
float y_out_norm = 0.0f;
float y_out_real = 0.0f;

// Función de activación ReLU
inline float relu(float z) {
  return (z > 0.0f) ? z : 0.0f;
}

// Función de propagación hacia adelante (Forward Pass)
float forward(const float raw_inputs[NUM_INPUTS]) {
  // 1. Normalización de entradas al rango [0, 1]
  for (int i = 0; i < NUM_INPUTS; i++) {
    if (x_max[i] - x_min[i] != 0.0f) {
      a_in[i] = (raw_inputs[i] - x_min[i]) / (x_max[i] - x_min[i]);
    } else {
      a_in[i] = 0.0f;
    }
  }

  // 2. Capa Oculta 1: Dense(32, activation='relu')
  // z1 = W1 * a_in + b1
  for (int j = 0; j < NUM_HIDDEN1; j++) {
    float sum = b1[j];
    for (int i = 0; i < NUM_INPUTS; i++) {
      sum += a_in[i] * W1[i][j];
    }
    a_h1[j] = relu(sum);
  }

  // 3. Capa Oculta 2: Dense(16, activation='relu')
  // z2 = W2 * a_h1 + b2
  for (int k = 0; k < NUM_HIDDEN2; k++) {
    float sum = b2[k];
    for (int j = 0; j < NUM_HIDDEN1; j++) {
      sum += a_h1[j] * W2[j][k];
    }
    a_h2[k] = relu(sum);
  }

  // 4. Capa de Salida: Dense(1, activation='linear')
  // y_norm = W3 * a_h2 + b3
  float sum_out = b3[0];
  for (int k = 0; k < NUM_HIDDEN2; k++) {
    sum_out += a_h2[k] * W3[k][0];
  }
  y_out_norm = sum_out; // Activación lineal

  // 5. DESNORMALIZACIÓN al rango real de bicicletas (Regla de oro del docente)
  y_out_real = (y_out_norm * (y_max - y_min)) + y_min;
  if (y_out_real < 0.0f) {
    y_out_real = 0.0f; // La demanda no puede ser negativa físicamente
  }

  return y_out_real;
}

// 5 Muestras de prueba extraidas del conjunto de test (X_test_raw) para verificacion exacta
// Variables: [Hour, Temp, Humidity, WindSpeed, Visibility, DewPoint, SolarRad, Rainfall, Snowfall, Holiday, FuncDay, Spring, Summer, Autumn, Winter]
const float test_samples[5][NUM_INPUTS] = {
  // Muestra 1: Demanda Real = 1728 bicis | Prediccion esperada Keras ~ 1251.54 bicis
  {8.0f, 27.2f, 69.0f, 1.8f, 1999.0f, 21.0f, 0.7f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f},
  // Muestra 2: Demanda Real = 822 bicis | Prediccion esperada Keras ~ 977.18 bicis
  {12.0f, 32.6f, 51.0f, 2.1f, 800.0f, 21.1f, 3.2f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f},
  // Muestra 3: Demanda Real = 658 bicis | Prediccion esperada Keras ~ 820.90 bicis
  {14.0f, 34.0f, 50.0f, 1.2f, 1744.0f, 22.1f, 1.7f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f},
  // Muestra 4: Demanda Real = 2716 bicis | Prediccion esperada Keras ~ 2223.60 bicis
  {18.0f, 16.9f, 47.0f, 1.4f, 1637.0f, 5.5f, 0.1f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 1.0f, 0.0f},
  // Muestra 5: Demanda Real = 1083 bicis | Prediccion esperada Keras ~ 785.64 bicis
  {7.0f, 6.4f, 51.0f, 1.0f, 1398.0f, -3.0f, 0.2f, 0.0f, 0.0f, 0.0f, 1.0f, 1.0f, 0.0f, 0.0f, 0.0f}
};

void setup() {
  Serial.begin(115200);
  while (!Serial) { ; } // Esperar conexión serial
  
  delay(1000);
  Serial.println(F("=============================================================="));
  Serial.println(F("  MINIPROYECTO REDES NEURONALES - MLP REGRESION EN ARDUINO"));
  Serial.println(F("  Dataset: Seoul Bike Sharing Demand"));
  Serial.println(F("  Universidad Autonoma de Occidente"));
  Serial.println(F("=============================================================="));
  Serial.println();
  
  Serial.print(F("Entradas: ")); Serial.println(NUM_INPUTS);
  Serial.print(F("Capa Oculta 1: ")); Serial.print(NUM_HIDDEN1); Serial.println(F(" neuronas (ReLU)"));
  Serial.print(F("Capa Oculta 2: ")); Serial.print(NUM_HIDDEN2); Serial.println(F(" neuronas (ReLU)"));
  Serial.print(F("Capa de Salida: ")); Serial.print(NUM_OUTPUTS); Serial.println(F(" neurona (Lineal)"));
  Serial.println();
  Serial.println(F("--- INICIANDO VALIDACION DE MUESTRAS DE PRUEBA ---"));
  Serial.println();

  for (int s = 0; s < 5; s++) {
    unsigned long t_start = micros();
    float pred_real = forward(test_samples[s]);
    unsigned long t_end = micros();

    Serial.println(F("--------------------------------------------------------------"));
    Serial.print(F("MUESTRA #")); Serial.println(s + 1);
    Serial.print(F("  Hora: ")); Serial.print((int)test_samples[s][0]);
    Serial.print(F("h | Temp: ")); Serial.print(test_samples[s][1], 1);
    Serial.print(F(" C | Hum: ")); Serial.print((int)test_samples[s][2]);
    Serial.print(F("% | Lluvia: ")); Serial.print(test_samples[s][7], 1); Serial.println(F(" mm"));
    
    Serial.print(F("  Prediccion Normalizada (y_norm): "));
    Serial.println(y_out_norm, 5);
    
    Serial.print(F("  PREDICCION FINAL DESNORMALIZADA: "));
    Serial.print(pred_real, 1);
    Serial.println(F(" bicicletas / hora"));
    
    Serial.print(F("  Tiempo de inferencia en Arduino: "));
    Serial.print(t_end - t_start);
    Serial.println(F(" microsegundos"));
  }
  
  Serial.println(F("--------------------------------------------------------------"));
  Serial.println(F("Validacion completada con exito. Sistema listo."));
  Serial.println(F("Ingrese cualquier caracter por Serial para repetir las pruebas."));
}

void loop() {
  if (Serial.available() > 0) {
    while (Serial.available() > 0) { Serial.read(); }
    setup();
  }
  delay(100);
}
