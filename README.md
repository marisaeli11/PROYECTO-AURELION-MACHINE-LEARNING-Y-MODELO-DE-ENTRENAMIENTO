# Proyecto Aurelion - Sprint 3: Clasificación de Fidelidad (Machine Learning)

## 📝 Inventario de Entrega
Este proyecto se compone de los siguientes archivos que deben estar en la misma carpeta para su correcta visualización en VS Code:

1.  **sprint3_aurelion_notebook.ipynb** (Notebook Principal)
2.  **master_rfm_aurelion_limpio.csv** (Dataset)
3.  **entrenamiento_modelo_aurelion.py** (Script de Entrenamiento)
4.  **grafico_distribucion_target.png** (Imagen)
5.  **grafico_frecuencia_vs_gasto.png** (Imagen)
6.  **grafico_frontera_decision.png** (Imagen)

---

## 1. Objetivo del Modelo
**Problema:** La tienda Aurelion tiene ventas constantes pero no identifica a sus clientes valiosos. Gastamos marketing en gente que no vuelve.
**Solución:** Un modelo de Machine Learning (Clasificación) que etiqueta a los clientes como **Fieles** o **Ocasionales** basándose en su comportamiento histórico.

---

## 2. Descripción del Dataset (X e y)

Para entrenar el modelo, dividimos la información en dos grupos. 

| Rol en ML | Variable | Definición (Qué representa) |
|-----------|----------|-----------------------------|
| **y (Target)** | `is_fidelizado` | **La Respuesta a predecir.** <br> 1 = Cliente Fiel (2+ compras). <br> 0 = Cliente Ocasional (1 compra). |
| **X (Excluido)** | `frequency` | **Variable de Negocio.** Define la fidelidad. **Se elimina del entrenamiento (X)** para evitar que el modelo memorice la regla. |
| **X (Feature)** | `recency_days` | **Variable Predictora.** Cantidad de días desde la última compra hasta hoy. |
| **X (Feature)** | `monetary_log` | **Variable Predictora.** Logaritmo del total gastado (usamos logaritmo para suavizar montos muy altos). |

### 🚨 Decisión Técnica: Prevención de Data Leakage
Durante el desarrollo, detectamos que incluir la variable `frequency` generaba un modelo con 100% de precisión artificial, lo cual indicaba una fuga de información (el modelo "leía" la regla de negocio en lugar de predecir).

**Acción Tomada:**
Decidimos eliminar `frequency` de las variables predictoras (X).

**¿Por qué?**
Queremos un modelo que pueda predecir si un cliente nuevo (con 1 sola compra) tiene potencial de ser fiel en el futuro, basándose únicamente en su perfil de gasto y recencia, sin esperar a que realice la segunda compra.

---

## 3. Ficha Técnica del Modelo

*   **Algoritmo:** Regresión Logística (`LogisticRegression`)
*   **Librería:** Scikit-Learn (Python)
*   **Tipo:** Clasificación Binaria Supervizada
*   **Optimizador (Solver):** `liblinear` (Ideal para datasets pequeños)
*   **Hiperparámetros:**
    *   Tasa de Aprendizaje: 0.01
    *   Iteraciones (Epochs): 100

### ¿Por qué Regresión Logística y no Lineal?
*   **Lineal:** Dibuja una recta. Predice números infinitos (ej: precio, temperatura).
*   **Logística:** Dibuja una "S". Predice **Probabilidad** (de 0 a 1). Como queremos clasificar "Sí/No", necesitamos la Logística.

---

## 4. Guía para la Demo (Los 10 Puntos)

| Punto Requerido | Dónde mostrarlo en VS Code |
|-----------------|----------------------------|
| 1. Objetivo | Ver Sección 1 de este README. |
| 2. Dataset (X e y) | Ver Sección 2 de este README (Tabla de Variables). |
| 3. Preprocesamiento | Notebook (Celda 3): `StandardScaler` y `OneHotEncoder`. |
| 4. División Train/Test | Notebook (Celda 4): `train_test_split`. |
| 5. Selección Algoritmo | Notebook: `LogisticRegression`. |
| 6. Entrenamiento | Notebook: `.fit(X_train, y_train)`. |
| 7. Predicciones | Notebook: `.predict(X_test)`. |
| 8. Métricas | Notebook: `confusion_matrix`, Accuracy 100%. |
| 9. Modelo Final | Script `entrenamiento_modelo_aurelion.py`. |
| 10. Gráficos | Ver Notebook o las imágenes adjuntas abajo. |

---

## 5. Visualización de Datos (Evidencia)

### Distribución del Target (Balance de clases)
![Distribución](./grafico_distribucion_target.png)

### Patrón de Comportamiento (Nuestras variables X)
![Patrón](./grafico_frecuencia_vs_gasto.png)

### Frontera de Decisión del Modelo
![Frontera](./grafico_frontera_decision.png)

---

## 6. Matriz de Confusión (Ayuda Memoria)

*   **TP (Verde):** La IA dijo "Fiel" y ACERTÓ.
*   **TN (Verde):** La IA dijo "Ocasional" y ACERTÓ.
*   **FP (Rojo - Error Tipo 1):** Dijo "Fiel" pero era Ocasional. (Gastamos dinero en vano).
*   **FN (Rojo - Error Tipo 2):** Dijo "Ocasional" pero era Fiel. (Perdimos un cliente VIP).
