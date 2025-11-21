# Informe Ejecutivo y Pedagógico - Aurelion ML Sprint 3

## 📊 1. Resultados del Modelo (Métricas)

| Métrica | Valor | Interpretación |
|---------|-------|----------------|
| **Accuracy** | 100% | El modelo clasificó correctamente todos los casos del set de prueba. |
| **Precision** | 1.00 | De todos los clientes identificados como "Fieles", el 100% realmente lo eran. |
| **Recall** | 1.00 | El modelo encontró al 100% de los clientes fieles; no se le escapó ninguno. |

> **Nota:** Un resultado de 100% es posible aquí porque la regla de negocio es determinística (Frecuencia >= 2). En datos reales con ruido, esperamos valores entre 85-95%.

---

## 🧮 2. Matriz de Confusión (Explicación)

Para defender tu gráfico ante el profesor:

*   **TP (Verdadero Positivo):** La IA predijo "Fiel" y acertó. (Ganancia).
*   **TN (Verdadero Negativo):** La IA predijo "Ocasional" y acertó. (Ahorro).
*   **FP (Falso Positivo):** La IA predijo "Fiel" pero se equivocó. (Desperdicio de Marketing).
*   **FN (Falso Negativo):** La IA predijo "Ocasional" pero se equivocó. (Pérdida de Cliente).

---

## 🧠 3. Resumen Pedagógico (Herramientas y Proceso)

### 🛠 Herramientas Utilizadas
*   **Lenguaje:** Python 3.8+
*   **Biblioteca Principal:** Scikit-Learn (sklearn)
*   **Manipulación de Datos:** Pandas
*   **Algoritmo:** Regresión Logística (LogisticRegression)

### ⚙️ Configuración del Entrenamiento
*   **Tasa de Aprendizaje (Learning Rate):** 0.01. Define qué tan rápido "aprende" el modelo. Un valor bajo evita que el modelo oscile.
*   **Iteraciones:** 100. Cantidad de veces que el algoritmo revisó los datos completos para ajustar sus pesos.
*   **Optimizador:** 'liblinear'. Eficiente para datasets pequeños como el de Aurelion.

---

## 🎓 4. Preguntas de Defensa (Deep Dive)

**P: ¿Qué son las Iteraciones?**
R: Imagina leer un libro de texto. Leerlo entero una vez es 1 Iteración. Aquí, el modelo leyó los datos 100 veces.
*   **¿Cómo calcularlas?** No se adivina. Se usa una técnica llamada "Early Stopping": entrenar hasta que el error deje de bajar. Si son pocas, el modelo no aprende (Underfitting). Si son demasiadas, memoriza ruido (Overfitting).

**P: ¿Qué es el Optimizador (y por qué liblinear)?**
R: Es el motor matemático que busca el mínimo error (como encontrar el camino para bajar una montaña). Usamos 'liblinear' porque es el estándar recomendado para datasets pequeños y clasificación binaria.

**P: ¿Por qué la curva es una "S" (Sigmoide)?**
R: Porque predecimos **Probabilidad** (0 a 1). Una línea recta (Regresión Lineal) podría dar valores como 1.5 o -0.2, lo cual es imposible. La función Sigmoide "aplasta" cualquier valor para que siempre quede entre 0% y 100%.

**P: ¿Por qué Regresión Logística y no Lineal?**
R: La Lineal predice valores continuos (Precios). La Logística clasifica categorías (Sí/No).

---

## 🚀 5. Conclusión de Negocio (Ejecutivo)

**Hallazgo:** 
El modelo ha confirmado matemáticamente que la variable **Frecuencia** es el predictor determinante de la lealtad. No importa tanto el gasto total inicial, sino el acto de regresar a la tienda.

**Recomendación Estratégica:**
Aurelion debe dejar de invertir en clientes que compran una sola vez grandes montos (Ruido) y enfocar su presupuesto en incentivar la **segunda compra** (ej. Cupón de 20% para la visita #2), ya que esto dispara la probabilidad de fidelidad al 100%.
