import tensorflow as tf
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import sys
import io

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

print("="*60)
print("NEURONA ARTIFICIAL - USO CONSECUTIVO DE NODOS")
print("Nodo 1 -> Nodo 2 (Pipeline consecutivo)")
print("="*60 + "\n")

print("📊 PARTE 1: ENTRENAMIENTO CON DATOS DE PRUEBA\n")

cigarros_prueba = np.array([0, 5, 10, 15, 20, 25, 30, 40], dtype=float)
anios_prueba = np.array([0, 2, 5, 8, 12, 15, 20, 25], dtype=float)
riesgo_prueba = np.array([0, 20, 35, 50, 65, 75, 85, 95], dtype=float)

print("🔹 NODO 1: Entrenando con 1 entrada (cigarros/día)...\n")

nodo1_prueba = tf.keras.Sequential([
    tf.keras.layers.Dense(units=1, input_shape=[1], name='nodo1')
])

nodo1_prueba.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.1),
    loss='mean_squared_error'
)

historial_nodo1_prueba = nodo1_prueba.fit(
    cigarros_prueba,
    riesgo_prueba,
    epochs=100,
    verbose=0
)

print(f"✅ Nodo 1 entrenado - Error final: {historial_nodo1_prueba.history['loss'][-1]:.4f}\n")

predicciones_nodo1_prueba = nodo1_prueba.predict(cigarros_prueba, verbose=0).flatten()

print("🔹 NODO 2: Entrenando con 2 entradas (cigarros + años) de forma CONSECUTIVA...\n")

entradas_nodo2_prueba = np.column_stack([cigarros_prueba, anios_prueba])

nodo2_prueba = tf.keras.Sequential([
    tf.keras.layers.Dense(units=1, input_shape=[2], name='nodo2')
])

nodo2_prueba.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.1),
    loss='mean_squared_error'
)

historial_nodo2_prueba = nodo2_prueba.fit(
    entradas_nodo2_prueba,
    riesgo_prueba,
    epochs=100,
    verbose=0
)

print(f"✅ Nodo 2 entrenado - Error final: {historial_nodo2_prueba.history['loss'][-1]:.4f}\n")

predicciones_nodo2_prueba = nodo2_prueba.predict(entradas_nodo2_prueba, verbose=0).flatten()

print("\n" + "="*60)
print("📊 PARTE 2: ENTRENAMIENTO CON DATOS REALES\n")

cigarros_real = np.array([0, 3, 7, 12, 15, 18, 22, 28, 35, 42], dtype=float)
anios_real = np.array([0, 1, 3, 6, 9, 11, 14, 18, 22, 28], dtype=float)
riesgo_real = np.array([5, 15, 28, 42, 53, 61, 72, 81, 89, 94], dtype=float)

print("🔹 NODO 1: Entrenando con datos REALES (1 entrada)...\n")

nodo1_real = tf.keras.Sequential([
    tf.keras.layers.Dense(units=1, input_shape=[1], name='nodo1_real')
])

nodo1_real.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.1),
    loss='mean_squared_error'
)

historial_nodo1_real = nodo1_real.fit(
    cigarros_real,
    riesgo_real,
    epochs=100,
    verbose=0
)

print(f"✅ Nodo 1 (real) entrenado - Error final: {historial_nodo1_real.history['loss'][-1]:.4f}\n")

predicciones_nodo1_real = nodo1_real.predict(cigarros_real, verbose=0).flatten()

print("🔹 NODO 2: Entrenando con datos REALES (2 entradas)...\n")

entradas_nodo2_real = np.column_stack([cigarros_real, anios_real])

nodo2_real = tf.keras.Sequential([
    tf.keras.layers.Dense(units=1, input_shape=[2], name='nodo2_real')
])

nodo2_real.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.1),
    loss='mean_squared_error'
)

historial_nodo2_real = nodo2_real.fit(
    entradas_nodo2_real,
    riesgo_real,
    epochs=100,
    verbose=0
)

print(f"✅ Nodo 2 (real) entrenado - Error final: {historial_nodo2_real.history['loss'][-1]:.4f}\n")

predicciones_nodo2_real = nodo2_real.predict(entradas_nodo2_real, verbose=0).flatten()

print("\n" + "="*60)
print("📈 GENERANDO GRÁFICAS COMPARATIVAS\n")

fig = plt.figure(figsize=(16, 10))

plt.subplot(3, 2, 1)
plt.title("Nodo 1 - Aprendizaje (Datos Prueba)", fontsize=12, fontweight='bold')
plt.xlabel("Época")
plt.ylabel("Error (MSE)")
plt.plot(historial_nodo1_prueba.history['loss'], color='blue', linewidth=2)
plt.grid(True, alpha=0.3)

plt.subplot(3, 2, 2)
plt.title("Nodo 1 - Aprendizaje (Datos Reales)", fontsize=12, fontweight='bold')
plt.xlabel("Época")
plt.ylabel("Error (MSE)")
plt.plot(historial_nodo1_real.history['loss'], color='green', linewidth=2)
plt.grid(True, alpha=0.3)

plt.subplot(3, 2, 3)
plt.title("Nodo 1 - Predicciones (Datos Prueba)", fontsize=12, fontweight='bold')
plt.xlabel("Cigarros/día")
plt.ylabel("Riesgo de salud")
plt.scatter(cigarros_prueba, riesgo_prueba, color='blue', s=100, label='Datos reales', zorder=3)
plt.scatter(cigarros_prueba, predicciones_nodo1_prueba, color='red', s=60, marker='x', label='Predicciones', zorder=4)
plt.legend()
plt.grid(True, alpha=0.3)

plt.subplot(3, 2, 4)
plt.title("Nodo 1 - Predicciones (Datos Reales)", fontsize=12, fontweight='bold')
plt.xlabel("Cigarros/día")
plt.ylabel("Riesgo de salud")
plt.scatter(cigarros_real, riesgo_real, color='green', s=100, label='Datos reales', zorder=3)
plt.scatter(cigarros_real, predicciones_nodo1_real, color='orange', s=60, marker='x', label='Predicciones', zorder=4)
plt.legend()
plt.grid(True, alpha=0.3)

plt.subplot(3, 2, 5)
plt.title("Nodo 2 - Real vs Predicho (Datos Prueba)", fontsize=12, fontweight='bold')
plt.xlabel("Riesgo Real")
plt.ylabel("Riesgo Predicho")
plt.scatter(riesgo_prueba, predicciones_nodo2_prueba, color='blue', s=100, label='Predicciones Nodo 2')
ideal = np.linspace(0, 100, 100)
plt.plot(ideal, ideal, 'r--', linewidth=2, label='Ajuste perfecto (y=x)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.axis('equal')

plt.subplot(3, 2, 6)
plt.title("Nodo 2 - Real vs Predicho (Datos Reales)", fontsize=12, fontweight='bold')
plt.xlabel("Riesgo Real")
plt.ylabel("Riesgo Predicho")
plt.scatter(riesgo_real, predicciones_nodo2_real, color='green', s=100, label='Predicciones Nodo 2')
plt.plot(ideal, ideal, 'r--', linewidth=2, label='Ajuste perfecto (y=x)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.axis('equal')

plt.tight_layout()
plt.savefig('comparacion_datos_prueba_vs_reales.png', dpi=300, bbox_inches='tight')
print("✅ Gráfica guardada: comparacion_datos_prueba_vs_reales.png")

print("\n" + "="*60)
print("📊 RESUMEN DE RESULTADOS")
print("="*60)

print("\n🔹 DATOS DE PRUEBA:")
print(f"   Nodo 1 - Error final: {historial_nodo1_prueba.history['loss'][-1]:.4f}")
print(f"   Nodo 2 - Error final: {historial_nodo2_prueba.history['loss'][-1]:.4f}")
print(f"   Mejora: {((historial_nodo1_prueba.history['loss'][-1] - historial_nodo2_prueba.history['loss'][-1]) / historial_nodo1_prueba.history['loss'][-1] * 100):.2f}%")

print("\n🔹 DATOS REALES:")
print(f"   Nodo 1 - Error final: {historial_nodo1_real.history['loss'][-1]:.4f}")
print(f"   Nodo 2 - Error final: {historial_nodo2_real.history['loss'][-1]:.4f}")
print(f"   Mejora: {((historial_nodo1_real.history['loss'][-1] - historial_nodo2_real.history['loss'][-1]) / historial_nodo1_real.history['loss'][-1] * 100):.2f}%")

print("\n🔹 COMPARACIÓN DATOS PRUEBA vs REALES:")
print(f"   Nodo 2 Prueba - Error: {historial_nodo2_prueba.history['loss'][-1]:.4f}")
print(f"   Nodo 2 Real - Error:   {historial_nodo2_real.history['loss'][-1]:.4f}")

if historial_nodo2_real.history['loss'][-1] < historial_nodo2_prueba.history['loss'][-1]:
    print("\n   ✅ Los datos REALES son MÁS EFICIENTES (menor error)")
else:
    print("\n   ⚠️ Los datos de PRUEBA tienen menor error (datos sintéticos idealizados)")

print("\n" + "="*60)
