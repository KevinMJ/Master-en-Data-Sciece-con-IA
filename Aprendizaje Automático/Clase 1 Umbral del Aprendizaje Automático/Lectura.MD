# Lectura: Aprender con Etiquetas — Fundamentos del Aprendizaje Supervisado
> El objetivo de esta lectura es introducir los conceptos clave del Aprendizaje Supervisado, comenzando por la distinción entre los tipos de problemas que aborda y profundizando en el algoritmo KNN como herramienta de clasificación, incluyendo sus aplicaciones en datos reales.

## 3.1 Definición del Aprendizaje Supervisado.
El Aprendizaje Supervisado es el paradigma de Machine Learning en el que el modelo aprende a partir de un conjunto de datos etiquetados: cada ejemplo de entrenamiento incluye tanto las características de entrada como la respuesta correcta (la etiqueta). El objetivo del modelo es encontrar la función que mejor mapea las entradas con las salidas, de forma que pueda generalizar a datos nuevos.

Los dos grandes tipos de problemas supervisados son:

1. Clasificación: La variable objetivo es una categoría discreta. El modelo aprende a asignar cada nueva observación a una de las clases predefinidas. Ejemplo: detectar si un tumor es maligno o benigno.

2. Regresión: La variable objetivo es un valor numérico continuo. El modelo aprende a predecir un número a partir de las características de entrada. Ejemplo: predecir el precio de una vivienda a partir de sus atributos.

## 3.2 KNN para Clasificación: Aplicación en Datos Reales.
El algoritmo KNN funciona midiendo la distancia euclidiana entre el punto a clasificar y todos los puntos del conjunto de entrenamiento, seleccionando los k más cercanos y asignando la clase mayoritaria entre ellos.

En datasets con múltiples variables numéricas, la inspección visual mediante un pair plot permite identificar qué pares de variables separan mejor las clases:

```python
import seaborn as sns

import matplotlib.pyplot as plt

sns.pairplot(x_vars='sepal_length', y_vars='sepal_width',

data=iris, plot_kws={'s': 50}, height=7, hue='species')

plt.show()

```

Una vez identificadas las características relevantes, el pipeline completo de KNN es:

```python
import pandas as pd

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import MinMaxScaler

from sklearn.neighbors import KNeighborsClassifier

from sklearn.metrics import classification_report

# 1. Separar características y etiqueta

X = iris.iloc[:, :4].values # Todas las columnas de características

y = iris.iloc[:, 4].values # Columna objetivo (especie)

# 2. Dividir en entrenamiento (80%) y prueba (20%)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# 3. Normalizar (obligatorio para KNN)

scaler = MinMaxScaler(feature_range=(0, 1))

scaler.fit(X_train)

X_train = scaler.transform(X_train)

X_test = scaler.transform(X_test)

# 4. Entrenar el modelo

classifier = KNeighborsClassifier(n_neighbors=11)

classifier.fit(X_train, y_train)

# 5. Predecir y evaluar

y_pred = classifier.predict(X_test)

print(classification_report(y_test, y_pred))

```

## 3.3 Aplicación a Datos de Cáncer: Clasificación Binaria.
Una de las aplicaciones más relevantes del aprendizaje supervisado es el diagnóstico médico. En el dataset de cáncer de mama (data.csv), el objetivo es clasificar tumores como malignos (M) o benignos (B) a partir de características numéricas extraídas de imágenes médicas.

Este es un problema de clasificación binaria, donde el error de clasificación tiene consecuencias distintas según su tipo:

Falso Negativo (FN): El modelo clasifica un tumor maligno como benigno. En contexto médico, es el error más costoso.

Falso Positivo (FP): El modelo clasifica un tumor benigno como maligno, generando un falso alarma.

La matriz de confusión hace visibles estos errores:

``` python
from sklearn.metrics import ConfusionMatrixDisplay

ConfusionMatrixDisplay.from_estimator(

classifier, X_test, y_test,

display_labels=['Benigno', 'Maligno'],

cmap=plt.cm.Blues

)

plt.title('Matriz de Confusión — Diagnóstico de Tumores')

plt.show()

```
## 3.4 Calidad del Vino: Clasificación Multiclase.
El dataset de calidad del vino (Wine Quality Data) presenta una complejidad adicional: no es binario sino multiclase. La variable objetivo es una puntuación de calidad que toma múltiples valores discretos.

En estos casos, el reporte de clasificación desglosa las métricas por cada clase, permitiendo identificar qué niveles de calidad el modelo predice mejor y cuáles le resultan más difíciles de distinguir.

La clave para abordar este tipo de problemas con KNN sigue siendo la misma: normalización rigurosa, selección cuidadosa de k y evaluación con métricas apropiadas.