import joblib

print("🔌 Intentando cargar el modelo desde el archivo .pkl...")

try:
    # Aquí es donde ocurre la magia:
    # Python abre el "frasco" y saca el modelo listo para usar
    modelo_cargado = joblib.load('modelo_fidelidad_aurelion.pkl')
    
    print("\n✅ ¡Éxito! El modelo se cargó correctamente.")
    print("---------------------------------------------")
    print("¿Qué hay dentro del archivo?")
    print(modelo_cargado)
    print("---------------------------------------------")
    print("Este objeto ya está listo para recibir nuevos datos y predecir.")

except FileNotFoundError:
    print("❌ Error: No encuentro el archivo .pkl. Asegúrate de estar en la misma carpeta.")