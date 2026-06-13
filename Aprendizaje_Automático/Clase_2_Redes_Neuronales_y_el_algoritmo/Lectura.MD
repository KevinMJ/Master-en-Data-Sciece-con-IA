# Lectura: La Distancia como Criterio — Redes Neuronales y el Algoritmo KNN
> El objetivo de esta lectura es continuar con la lectura anterior y profundizar en dos algoritmos fundamentales del Aprendizaje Supervisado: las Redes Neuronales Artificiales y el algoritmo K-Nearest Neighbors (KNN), incluyendo sus métricas de evaluación.

## 2.1 Redes Neuronales Artificiales.
Una red neuronal artificial es un modelo de cómputo inspirado en la estructura del cerebro humano. Está compuesta por capas de nodos interconectados (neuronas) que transforman las entradas en predicciones a través de operaciones matemáticas.

En TensorFlow, una red neuronal sencilla para regresión lineal se define así:

```python
import tensorflow as tf

modelo = tf.keras.Sequential([

tf.keras.layers.Dense(units=1, input_shape=[1])

])

modelo.compile(optimizer='sgd', loss='mean_squared_error')
```

El parámetro optimizer='sgd' indica que se usará el Descenso de Gradiente Estocástico para ajustar los pesos de la red. El parámetro loss='mean_squared_error' define cómo se mide el error entre la predicción y el valor real.

El entrenamiento se ejecuta con el método .fit():

```python
historial = modelo.fit(xs, ys, epochs=500, verbose=0)
```

Cada época es una pasada completa del modelo sobre todos los datos de entrenamiento. A mayor número de épocas, mayor refinamiento de los pesos, aunque existe el riesgo de sobreajuste (overfitting) si el modelo memoriza los datos en lugar de generalizar.

Una vez entrenado, el modelo puede guardar sus pesos para reutilizarse sin necesidad de reentrenar:

```python
modelo.save('mymodel.h5')
```
## 2.2 K-Nearest Neighbors (KNN).
K-Nearest Neighbors es uno de los algoritmos de clasificación más intuitivos del Machine Learning. Su lógica es simple: para clasificar un nuevo punto de datos, busca los k ejemplos más cercanos en el conjunto de entrenamiento y asigna la clase más frecuente entre ellos.

```python 
from sklearn.neighbors import KNeighborsClassifier

classifier = KNeighborsClassifier(n_neighbors=11)

classifier.fit(X_train, y_train)

```

La elección del valor de k es crítica. Una regla práctica es comenzar con la raíz cuadrada del número de observaciones de entrenamiento:

```python
import math

k_sugerido = math.sqrt(len(X_train)) # Si hay 120 datos de entrenamiento → k ≈ 11
```

Dado que KNN mide distancias entre puntos, es imprescindible normalizar los datos antes de entrenar. Si una variable varía entre 0 y 1 y otra entre 0 y 10,000, la segunda dominará artificialmente el cálculo de distancias.

## 2.3 Evaluación del Modelo: Métricas de Clasificación.
Una vez entrenado el modelo, la evaluación va más allá de un simple porcentaje de aciertos. El reporte de clasificación de Scikit-learn proporciona métricas por clase:

```python
from sklearn.metrics import classification_report, ConfusionMatrixDisplay

import matplotlib.pyplot as plt

y_pred = classifier.predict(X_test)

# Reporte detallado por clase

print(classification_report(y_test, y_pred))

# Matriz de confusión visual

class_names = ['setosa', 'versicolor', 'virginica']

ConfusionMatrixDisplay.from_estimator(

classifier, X_test, y_test,

display_labels=class_names,

cmap=plt.cm.Blues

)

plt.show()
```

La matriz de confusión muestra cuántas predicciones fueron correctas e incorrectas por clase, revelando si el modelo confunde sistemáticamente unas clases con otras.

## 2.4 Selección Óptima de K: El Análisis de Error.
Para encontrar el mejor valor de k, se puede evaluar el error promedio del modelo con distintos valores y elegir el que minimice los errores en el conjunto de prueba:

```python
import numpy as np

error = []

for i in range(1, 50):

knn = KNeighborsClassifier(n_neighbors=i)

knn.fit(X_train, y_train)

pred_i = knn.predict(X_test)

error.append(np.mean(pred_i != y_test))

plt.figure(figsize=(12, 6))

plt.plot(range(1, 50), error, color='#1d91c0', linestyle='dashed', marker='o')

plt.ylim(0, 0.2)

plt.title('Error promedio vs. número de vecinos K')

plt.xlabel('Número de vecinos K')

plt.ylabel('Error promedio')

plt.grid(alpha=0.3)

plt.show()

```

Este análisis permite tomar una decisión informada sobre el hiperparámetro k, basándose en evidencia visual y no en suposiciones arbitrarias.