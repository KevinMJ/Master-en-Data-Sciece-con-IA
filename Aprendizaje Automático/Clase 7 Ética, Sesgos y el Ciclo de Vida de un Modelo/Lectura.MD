# Lectura: El Científico de Datos Completo — Ética, Sesgos y el Ciclo de Vida de un Modelo
El objetivo de esta lectura complementaria es introducir los principios éticos, los riesgos de sesgo y el ciclo de vida completo de un modelo de Machine Learning, diferenciadores profesionales que distinguen a un Data Scientist responsable de uno que solo sabe escribir código.

## 7.1 El Ciclo de Vida de un Modelo de Machine Learning.
Un modelo de Machine Learning no es un artefacto estático: tiene un ciclo de vida que va desde la definición del problema hasta su despliegue y monitoreo en producción. Comprender este ciclo es esencial para quienes aspiran a trabajar en equipos de Data Science profesionales.

Las etapas del ciclo son:

1. **Definición del problema:** ¿Qué pregunta de negocio quiero responder? ¿Es un problema de clasificación, regresión o clustering? ¿Qué métrica importa más?

2. **Recopilación y preparación de datos:** El pipeline de limpieza, exploración y transformación visto en el Módulo 3.

3. **Selección y entrenamiento del modelo:** Elegir el algoritmo adecuado, ajustar hiperparámetros y entrenar.

4. **Evaluación:** Medir el rendimiento con métricas apropiadas al problema (accuracy, F1, Silhouette, etc.).

5. **Despliegue:** Integrar el modelo en un sistema productivo donde pueda recibir datos nuevos y hacer predicciones.

6. **Monitoreo:** Vigilar que el modelo mantenga su rendimiento en el tiempo. Los datos del mundo real cambian (data drift), y un modelo que funcionaba bien puede degradarse.

## 7.2 El Problema del Sobreajuste y el Subajuste.
Uno de los conceptos más importantes en Machine Learning es el balance entre sobreajuste (overfitting) y subajuste (underfitting):

**Sobreajuste:** El modelo memoriza los datos de entrenamiento en lugar de generalizar patrones. Tiene muy buen rendimiento en entrenamiento pero pésimo en datos nuevos. Un KNN con k=1 es un ejemplo extremo: acierta perfectamente en entrenamiento pero puede ser terrible en prueba.

**Subajuste:** El modelo es demasiado simple para capturar la complejidad de los datos. Tiene mal rendimiento tanto en entrenamiento como en prueba.

El objetivo es encontrar el punto de bias-variance tradeoff donde el modelo generaliza bien: aprende los patrones reales sin memorizar el ruido de los datos de entrenamiento.

```python
# El análisis de error por K ilustra visualmente el overfitting/underfitting en KNN

error = []

for i in range(1, 50):

knn = KNeighborsClassifier(n_neighbors=i)

knn.fit(X_train, y_train)

pred_i = knn.predict(X_test)

error.append(np.mean(pred_i != y_test))

# K muy bajo → sobreajuste (memoriza el ruido)

# K muy alto → subajuste (demasiado generalista)

# El K óptimo está en el valle de la curva

```
## 7.3 Ética y Sesgo en Machine Learning.
Los modelos de Machine Learning no son neutrales: heredan los sesgos de los datos con los que son entrenados. Un modelo entrenado con datos históricos que reflejan decisiones injustas reproducirá y amplificará esas injusticias.

Algunos principios éticos que todo Data Scientist debe internalizar:

* **Representatividad de los datos:** Si el conjunto de entrenamiento no representa adecuadamente todos los grupos de la población, el modelo tendrá peor rendimiento para los grupos subrepresentados. En diagnóstico médico, esto puede tener consecuencias graves.

* **Métricas alineadas con el impacto:** La accuracy (porcentaje de aciertos) puede ser engañosa en datasets desbalanceados. Si el 99% de las transacciones son legítimas, un modelo que siempre predice "legítima" tiene 99% de accuracy pero es inútil para detectar fraude. Usar recall, precision y F1 por clase evita esta trampa.

* **Transparencia e interpretabilidad:** En contextos de alto riesgo (crédito, diagnóstico médico, justicia), los modelos de "caja negra" (que no explican sus decisiones) son inapropiados. Prefiere modelos interpretables o técnicas de explicabilidad (SHAP, LIME).

## 7.4 Buenas Prácticas para un Proyecto de ML Robusto.
Siempre separar datos de entrenamiento y prueba antes de cualquier transformación. El scaler debe ajustarse (fit) solo con datos de entrenamiento. Si incluyes datos de prueba en el fit, introduces data leakage (el modelo "ve" el futuro durante el entrenamiento).

* Guardar los modelos entrenados con un nombre descriptivo que incluya el algoritmo, la fecha y la versión.

* Documentar los experimentos: Qué hiperparámetros se probaron, qué métricas se obtuvieron, qué versión de los datos se usó.

* Fijar la semilla aleatoria (random_state) en todas las operaciones estocásticas para garantizar reproducibilidad.