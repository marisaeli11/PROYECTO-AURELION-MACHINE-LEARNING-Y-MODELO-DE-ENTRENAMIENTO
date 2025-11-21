# Proyecto Aurelion - Sprint 3: Clasificación de Fidelidad

## 1. Contexto de Negocio
**Problema:** La tienda Aurelion tiene ventas constantes pero no identifica a sus clientes valiosos. El marketing genérico resulta ineficiente.

**Objetivo ML:** Modelo de Clasificación Binaria para predecir si un cliente será Fiel o Ocasional basado en 6 meses de historia.

**Valor para el Negocio:**
- **Optimizar Marketing:** Enfocar campañas de retención en clientes con potencial.
- **Personalización:** Ofertas exclusivas para el segmento de alto valor.

## 2. Definición del Target (Y)

| Clase | Definición | Lógica de Negocio |
|-------|------------|-------------------|
| **1 (Fiel)** | Cliente Recurrente (Volvió a comprar) | Frequency >= 2 |
| **0 (Ocasional)** | Cliente de una sola vez | Frequency < 2 |

## 3. Ficha Técnica del Modelo (Machine Learning)

Este proyecto implementa un algoritmo de aprendizaje supervisado para resolver el problema de clasificación.

### Especificaciones
*   **Algoritmo:** Regresión Logística (LogisticRegression)
*   **Librería:** Scikit-Learn (Python)
*   **Tipo de Problema:** Clasificación Binaria Supervisada

### Hiperparámetros de Entrenamiento
*   **Tasa de Aprendizaje (Learning Rate):** 0.01
*   **Iteraciones:** 100
*   **Optimizador:** liblinear (Ideal para datasets pequeños)

### Resultados
*   **Métrica Principal:** Accuracy (Exactitud)
*   **Validación:** El modelo utiliza validación cruzada (Train/Test Split) para asegurar que las predicciones sean generalizables.

## 4. Nota Teórica: ¿Por qué Regresión Logística? (Defensa del Proyecto)

Es posible que surja la duda de por qué no usar Regresión Lineal.

*   **Regresión Lineal:** Se usa para predecir **números continuos** (ej: Predecir el precio de una casa: $150,000). Dibuja una línea recta infinita.
*   **Regresión Logística:** Se usa para predecir **categorías** (ej: ¿Es Fiel? Sí/No). Dibuja una curva en forma de "S" (Sigmoide) que comprime el resultado entre 0 y 1, lo cual representa la **probabilidad**.

**Conclusión:** Como nuestro target es binario (0 o 1), la Regresión Logística es la herramienta matemáticamente correcta.

## 5. Guía de Interpretación: Matriz de Confusión

Para defender los resultados del modelo:

*   **TP (Verdadero Positivo):** La IA predijo "Fiel" y acertó. (Ganancia).
*   **TN (Verdadero Negativo):** La IA predijo "Ocasional" y acertó. (Ahorro).
*   **FP (Falso Positivo):** La IA predijo "Fiel" pero se equivocó. (Desperdicio de Marketing).
*   **FN (Falso Negativo):** La IA predijo "Ocasional" pero se equivocó. (Pérdida de Cliente Potencial - Error Grave).

---
*Generado automáticamente por Aurelion ML Dashboard*
