# Lectura: Silhouette y Centroides — Evaluación Avanzada del Clustering
El objetivo de esta lectura es continuar con la lectura anterior y abordar la evaluación avanzada de modelos de clustering, incluyendo el diagrama Silhouette como herramienta de diagnóstico profundo y las mejores prácticas para la interpretación de los resultados en contextos de negocio reales.

## 6.1 El Diagrama Silhouette: Diagnóstico Visual Profundo.
Mientras que el Silhouette score promedio ofrece un número único que resume la calidad del clustering, el diagrama Silhouette va más allá: muestra el coeficiente individual de cada observación, agrupada por cluster y ordenada por valor, revelando la composición interna de cada grupo.

```python
from sklearn.metrics import silhouette_samples, silhouette_score

import matplotlib.cm as cm

import numpy as np

n_clusters_range = range(2, 9)

n_plots = len(n_clusters_range)

cols = 3

rows = (n_plots + cols - 1) // cols

fig, axes = plt.subplots(rows, cols, figsize=(cols * 6, rows * 4))

axes = axes.flatten()

for idx, k in enumerate(n_clusters_range):

ax = axes[idx]

ax.set_xlim([-0.1, 1])

ax.set_ylim([0, len(X_train) + (k + 1) * 10])

clusterer = KMeans(n_clusters=k, random_state=10, n_init='auto')

cluster_labels = clusterer.fit_predict(X_train)

silhouette_avg = silhouette_score(X_train, cluster_labels)

sample_silhouette_values = silhouette_samples(X_train, cluster_labels)

y_lower = 10

for i in range(k):

ith_cluster_values = sample_silhouette_values[cluster_labels == i]

ith_cluster_values.sort()

size_cluster_i = ith_cluster_values.shape[0]

y_upper = y_lower + size_cluster_i

color = cm.nipy_spectral(float(i) / k)

ax.fill_betweenx(np.arange(y_lower, y_upper), 0, ith_cluster_values,

facecolor=color, edgecolor=color, alpha=0.7)

ax.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))

y_lower = y_upper + 10

ax.axvline(x=silhouette_avg, color="red", linestyle="--")

ax.set_title(f"k = {k} (avg = {silhouette_avg:.3f})", fontsize=10)

plt.tight_layout()

plt.show()

```

Cada "hoja" del diagrama representa un cluster. Una buena configuración de clusters produce **hojas anchas** (muchas observaciones bien clasificadas) **y uniformes**. Hojas delgadas o irregulares indican clusters problemáticos.

## 6.2 Interpretación del Diagrama: Qué Buscar.
Al analizar el diagrama Silhouette, se buscan estas características en la configuración ideal:

* **Hojas que superan la línea punteada (promedio):** Indica que la mayoría de las observaciones están bien asignadas.

* **Hojas de tamaño similar:** Señal de clusters balanceados, sin uno dominantemente grande y otros muy pequeños.

* **Ausencia de coeficientes negativos:** Los valores negativos indican observaciones que probablemente fueron asignadas al cluster incorrecto.

* **Ancho consistente:** Todas las observaciones de un cluster tienen coeficientes cercanos, sin una larga "cola" de valores bajos.

## 6.3 De los Clusters al Conocimiento de Negocio.
La evaluación técnica del clustering (inercia, Silhouette) es solo la mitad del trabajo. La otra mitad es interpretar qué significan los clusters en términos del problema de negocio.

Los centroides son la clave de esta interpretación. Cada centroide resume las características típicas de su cluster:

```python
# Obtener los centroides como DataFrame interpretable

centroides_df = pd.DataFrame(

model.cluster_centers_,

columns=X.columns

)

# Visualizar los centroides

centroides_df.T.plot(kind='bar', figsize=(12, 6))

plt.title('Características Promedio por Cluster')
plt.xlabel('Variable')
plt.ylabel('Valor Promedio')
plt.legend(title='Cluster')
plt.tight_layout()

plt.show()
```

En el contexto de detección de fraude con creditcard.csv, por ejemplo, el análisis de los centroides permite responder preguntas como: ¿Los fraudes tienden a ocurrir en transacciones de cierto monto? ¿En ciertos horarios? ¿Con ciertos patrones de comportamiento en las variables transformadas?

## 6.4 Etiquetado del Conjunto de Prueba.
Una vez definido el número óptimo de clusters y entrenado el modelo final, el paso práctico es etiquetar el conjunto de prueba para análisis posteriores:

```python
# Predecir el cluster de cada transacción en el set de prueba (30% de los datos)

X_test['cluster'] = model.predict(X_test)

# Analizar la distribución de la variable objetivo por cluster

resultado = pd.concat([X_test['cluster'], y_test], axis=1)

print(resultado.groupby('cluster')['Class'].value_counts())

```

Este análisis revela si K-Means logró separar naturalmente las transacciones fraudulentas de las legítimas, o si los fraudes se distribuyen de forma dispersa entre los clusters, lo que indicaría que las características disponibles no son suficientemente discriminantes para este enfoque.