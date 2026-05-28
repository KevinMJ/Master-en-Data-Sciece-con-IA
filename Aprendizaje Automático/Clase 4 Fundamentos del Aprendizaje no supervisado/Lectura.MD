# Lectura: Clasificar sin Etiquetas — Fundamentos del Aprendizaje No Supervisado
> El objetivo de esta lectura es continuar con la lectura anterior y abordar el Aprendizaje No Supervisado, específicamente el algoritmo K-Means para clustering, incluyendo los métodos estadísticos para determinar el número óptimo de grupos.

## 4.1 Definición del Aprendizaje No Supervisado.
El Aprendizaje No Supervisado aborda un problema fundamentalmente distinto al supervisado: los datos no tienen etiquetas. No existe una "respuesta correcta" que el modelo pueda aprender. En cambio, el objetivo es descubrir la estructura inherente de los datos: grupos, patrones o representaciones que emergen de las propias relaciones entre las observaciones.

Como lo describió el investigador Yann LeCun: "Si la inteligencia fuera un pastel, el aprendizaje no supervisado sería el pastel, el aprendizaje supervisado sería el glaseado, y el aprendizaje por refuerzo sería la cereza encima."

El clustering o agrupamiento es la tarea más común del aprendizaje no supervisado: agrupar observaciones similares sin conocer de antemano cuántos grupos existen ni a qué clase pertenece cada uno.

## 4.2 K-Means: Fundamentos.
K-Means es el algoritmo de clustering más utilizado en la industria. Crea k grupos a partir de datos numéricos, donde el input son las mediciones de interés y el output es:

Los centroides de los clusters resultantes (el punto medio de cada grupo).

La asignación de cada dato a un cluster determinado.

La calidad del modelo se mide con la inercia (también llamada Within Sum of Squares):

$$WSS = \sum_{i=1}^{K} \sum_{j=1}^{n_i} |x_{ij} - c_i|^2$$

La inercia mide la distancia cuadrática media entre cada observación y su centroide más cercano. Cuanto menor es la inercia, más compactos son los clusters.

```python
from sklearn.cluster import KMeans

# Separar características de la etiqueta (no se usa para entrenar)

X = iris.drop('species', axis=1)

y = iris['species']

# Dividir en entrenamiento y prueba

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=42, stratify=y)

# Entrenar el modelo con k=3 clusters

model = KMeans(n_clusters=3)

model.fit(X_train)

print(f'Inercia del modelo: {model.inertia_}')

print(f'Centroides: {model.cluster_centers_}')

```

Una vez entrenado, el modelo puede asignar nuevas observaciones al cluster más cercano:

```python
# Predecir el cluster de los datos de prueba

labels = model.predict(X_test)

X_test['cluster'] = labels
```
## 4.3 El Método del Codo: Encontrar el K Óptimo.
La inercia siempre disminuye al aumentar k (más clusters = grupos más pequeños y compactos), por lo que no es posible minimizarla sin más. La regla del codo busca el punto de inflexión donde añadir más clusters ya no produce una reducción significativa de la inercia:

```python 
inertias = {k: KMeans(n_clusters=k).fit(X_train).inertia_ for k in range(1, 13)}

plt.plot(list(inertias.keys()), list(inertias.values()), '-o')

plt.xlabel('Número de clusters, k')

plt.ylabel('Inercia')

plt.xticks(list(inertias.keys()))

plt.title('Método del Codo — Selección de K')

plt.show()
```

El "codo" de la curva (el punto donde la pendiente cambia bruscamente) sugiere el número óptimo de clusters.

## 4.4 El Coeficiente Silhouette: Una Métrica Más Precisa.
El Silhouette score ofrece una evaluación más robusta que el método del codo. Para cada observación, calcula:

$$\text{Silhouette}(i) = \frac{b(i) - a(i)}{\max(a(i), b(i))}$$

Donde:

**a:** es la distancia media a otras instancias del mismo cluster (cohesión interna).

**b:** es la distancia media al cluster más cercano diferente (separación externa).

El coeficiente varía entre -1 y 1:

Cerca de 1: La instancia está bien ubicada en su cluster y lejos de los demás.

Cerca de 0: La instancia está en la frontera entre dos clusters.

Cerca de -1: La instancia fue asignada al cluster incorrecto.

```python
from sklearn.metrics import silhouette_score

silhouettes = {

k: silhouette_score(X_train, KMeans(n_clusters=k).fit_predict(X_train))

for k in range(2, 9)

}

plt.plot(list(silhouettes.keys()), list(silhouettes.values()), '-o')

plt.xlabel('Número de clusters, k')

plt.ylabel('Silhouette Score')

plt.title('Coeficiente Silhouette — Selección de K')

plt.show()

```

Usando ambos métodos conjuntamente —el codo para una primera aproximación y el Silhouette para confirmar— se puede justificar estadísticamente la elección del número de clusters.