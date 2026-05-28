# Lectura: Grupos sin Respuesta — Clustering con K-Means y Detección de Anomalías
> El objetivo de esta lectura es introducir los conceptos clave del Aprendizaje No Supervisado, abordando el algoritmo K-Means desde sus fundamentos hasta su aplicación en la detección de fraude con datos reales.

## 5.1 Clustering: Particionando el Espacio de Datos.
El análisis de agrupamiento o clustering es el proceso de particionar un conjunto de datos en subconjuntos llamados clusters o grupos, donde:

* Los datos de un mismo cluster deben ser lo más similares posible entre sí.

* Los datos de clusters distintos deben ser lo más diferentes posible.

Esta dualidad es el criterio de calidad de todo algoritmo de clustering: maximizar la cohesión interna y la separación externa.

## 5.2 K-Means en Datos Financieros: Detección de Fraude.
La detección de fraude en transacciones con tarjeta de crédito es uno de los casos de uso más importantes del clustering. El dataset creditcard.csv contiene transacciones reales (obtenidas de Kaggle) donde la columna Class indica si una transacción es fraudulenta (1) o legítima (0).

La pregunta central del análisis es: ¿los datos fraudulentos se concentran en un único cluster o se distribuyen entre varios?

```python
import pandas as pd

from sklearn.cluster import KMeans

from sklearn.model_selection import train_test_split

# El clustering usa todas las columnas EXCEPTO la etiqueta objetivo

X = df.drop('Class', axis=1)

y = df['Class']

# Separar un 30% de los datos para etiquetado posterior

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=42)

# Entrenar K-Means

model = KMeans(n_clusters=2)

model.fit(X_train)

```

Nótese un detalle crítico: aunque tenemos la columna Class, no la usamos durante el entrenamiento. K-Means no conoce las etiquetas; simplemente agrupa los datos por similitud matemática. Las etiquetas solo se usan al final para interpretar qué significa cada cluster.

```python
# Etiquetar el conjunto de prueba con el cluster asignado

X_test['cluster'] = model.predict(X_test)

# Analizar: ¿en qué cluster(s) se concentran los fraudes?

fraudes = df[df['Class'] == 1]

print(f"Distribución de fraudes por cluster:\n{X_test[X_test.index.isin(fraudes.index)]['cluster'].value_counts()}")

```
## 5.3 El Método del Codo y Silhouette en Datos Reales.
Con datasets del mundo real, la elección de k raramente es obvia. La aplicación conjunta de ambos métodos es el estándar profesional:

```python
from sklearn.metrics import silhouette_score

import matplotlib.pyplot as plt

# Método del Codo

inertias = {k: KMeans(n_clusters=k).fit(X_train).inertia_ for k in range(2, 10)}

# Método Silhouette

silhouettes = {

k: silhouette_score(X_train, KMeans(n_clusters=k).fit_predict(X_train))

for k in range(2, 9)

}
```

La interpretación conjunta de ambas curvas permite justificar la elección de k con argumentos tanto visuales como estadísticos.

## 5.4 Recomendaciones Prácticas para K-Means.
K-Means tiene sensibilidades conocidas que deben gestionarse en cualquier aplicación profesional:

**Normalización obligatoria:** K-Means mide distancias euclidianas. Si las variables tienen escalas muy distintas (ej: una en miles y otra en decimales), las de mayor escala dominarán los clusters artificialmente. La normalización previa garantiza que todas las variables contribuyan equitativamente.

```python
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)

X_test_scaled = scaler.transform(X_test)
```

Interpretación de los centroides: Los centroides del modelo resumen las características típicas de cada cluster. Analizar sus valores es clave para darle significado de negocio a los grupos encontrados.

```python
centroides_df = pd.DataFrame(

model.cluster_centers_,

columns=X.columns

)

print(centroides_df)
```

Sensibilidad a la inicialización: K-Means puede converger a soluciones locales según la posición inicial de los centroides. Ejecutar el algoritmo múltiples veces con distintas inicializaciones y quedarse con la mejor solución es una práctica recomendada (n_init en Scikit-learn).