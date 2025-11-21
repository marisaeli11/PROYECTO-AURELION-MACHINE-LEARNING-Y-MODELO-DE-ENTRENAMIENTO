import pandas as pd
import numpy as np
import joblib

# ==============================================================================
# 1. GENERAR DATOS DE NUEVOS CLIENTES (SIMULACIÓN)
# ==============================================================================
print("📝 Generando lista de clientes nuevos...")

# Fíjate: Aquí NO ponemos la frecuencia, porque son clientes nuevos (1 sola compra)
# y tu modelo ya es lo suficientemente inteligente para no necesitarla.
datos_nuevos = {
    'id_cliente': [1001, 1002, 1003, 1004, 1005],
    'nombre': ['Juan Perez', 'Maria Gomez', 'Carlos Ruiz', 'Ana Diaz', 'Luis Torres'],
    'recency_days': [5, 150, 10, 300, 2],       # Días desde la compra
    'monetary_log': [11.5, 8.5, 10.8, 9.0, 11.2], # Cuánto gastaron (Log)
    'ciudad': ['Cordoba', 'Villa Maria', 'Carlos Paz', 'Cordoba', 'Rio Cuarto'],
    'categoria_preferida': ['Alimentos', 'Limpieza', 'Alimentos', 'Limpieza', 'Alimentos']
}

df_nuevos = pd.DataFrame(datos_nuevos)
print("✅ Datos listos para analizar.")

# ==============================================================================
# 2. CARGAR TU MODELO (.PKL)
# ==============================================================================
print("🔌 Cargando modelo de Inteligencia Artificial...")
try:
    modelo = joblib.load('modelo_fidelidad_aurelion.pkl')
    print("✅ Modelo cargado.")
except FileNotFoundError:
    print("❌ Error: No se encuentra el archivo .pkl")
    exit()

# ==============================================================================
# 3. PREDECIR EL FUTURO (INFERENCIA)
# ==============================================================================
print("🔮 Analizando perfiles de clientes...")

# El modelo predice si será fiel (1) o no (0)
predicciones = modelo.predict(df_nuevos)
# El modelo calcula la seguridad de su decisión (%)
probabilidades = modelo.predict_proba(df_nuevos)[:, 1]

# ==============================================================================
# 4. REPORTE PARA MARKETING
# ==============================================================================
df_nuevos['Es_VIP_Potencial'] = predicciones
df_nuevos['Probabilidad'] = probabilidades

# Convertimos el 1 y 0 a texto bonito
df_nuevos['Estado'] = df_nuevos['Es_VIP_Potencial'].map({1: '⭐ FIDELIZAR', 0: 'Normal'})

print("\n📊 RESULTADOS DEL ANÁLISIS:")
print(df_nuevos[['nombre', 'Estado', 'Probabilidad']])

# Guardar en Excel/CSV
df_nuevos.to_csv('REPORTE_FINAL_MARKETING.csv', index=False)
print("\n💾 Archivo guardado: 'REPORTE_FINAL_MARKETING.csv'")
print("🏁 ¡Ciclo de Data Science completado!")