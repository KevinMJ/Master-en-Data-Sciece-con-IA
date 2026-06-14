# Apuntes
## K-Means en Profundidad: 
> ¿Cómo funciona matemáticamente el algoritmo K-Means paso a paso? Explica el rol de los centroides, la inercia y el proceso iterativo de asignación y actualización de clusters. ¿Por qué K-Means siempre converge pero no siempre encuentra el óptimo global?

K-Means es básicamente un juego de tira y afloja entre puntos y promedios. Quieres partir tus datos en K grupos de forma que cada punto esté lo más cerca posible del centro de su grupo, y repites eso hasta que los centros dejan de moverse.

### La función que K-Means intenta minimizar
Todo gira alrededor de la inercia, también llamada SSE (sum of squared errors):

$$ J(C, \mu) = \sum_{k=1}^{K} \sum_{x_i \in C_k} \|x_i - \mu_k\|^2 $$
- $C_k$ es el conjunto de puntos asignados al cluster k
- $\mu_k$ es el centroide del cluster k
- $\|x_i - \mu_k\|^2$ es la distancia euclidiana al cuadrado

Minimizar $J$ es lo mismo que decir: haz los clusters lo más compactos posible.

### El rol de los centroides
Un centroide no es un punto real de tus datos, es el promedio aritmético de todos los puntos en su cluster:

$$\mu_k = \frac{1}{|C_k|} \sum_{x_i \in C_k} x_i$$

Por eso K-Means solo funciona bien con distancias euclidianas: el promedio es el punto que minimiza la suma de distancias cuadradas. Si cambias la métrica, el promedio deja de ser óptimo.

El centroide cumple dos papeles:

- **Representante:** resume dónde está el cluster
- **Atractor:** en la siguiente iteración, atrae hacia sí a los puntos más cercanos

### Paso a paso del algoritmo de Lloyd
Empiezas eligiendo K (a mano o con elbow/silhouette) y colocando K centroides iniciales $\mu_1^{(0)},...,\mu_K^{(0)}$.

Luego iteras:

1. **Asignación**
Cada punto elige el centroide más cercano:

$$C_k^{(t)} = \{ x_i : \|x_i - \mu_k^{(t)}\|^2 \le \|x_i - \mu_j^{(t)}\|^2 \ \forall j \neq k \}$$

Es una partición de Voronoi: trazas fronteras a mitad de camino entre centroides.

2. **Actualización**
Recalculas cada centroide como la media de sus puntos asignados:

$$\mu_k^{(t+1)} = \frac{1}{|C_k^{(t)}|} \sum_{x_i \in C_k^{(t)}} x_i$$
Si un cluster queda vacío, lo reinicializas (típico en implementaciones).

Repite 1 y 2 hasta que $\mu_k^{(t+1)} \approx \mu_k^{(t)}$ o las asignaciones no cambian.

### Por qué la inercia siempre baja
Cada paso está diseñado para no empeorar $J$:

- En asignación, mantienes $\mu$ fijo y mueves cada $x_i$ al $\mu_k$ más cercano. Por definición, eso no puede aumentar $\|x_i - \mu_k\|^2$.

- En actualización, mantienes $C_k$ fijo y mueves $\mu_k$ al promedio. El promedio es el minimizador exacto de la suma de cuadrados, así que J baja o se queda igual.
$J$ está acotada inferiormente por 0 y solo hay un número finito de particiones posibles ( $K^n$ como máximo, aunque en práctica muchas menos). Entonces la secuencia $J^{(0)} \ge J^{(1)} \ge J^{(2)}...$ tiene que estabilizarse. Ahí es cuando K-Means "converge".

Converge en pasos finitos, normalmente en 10-20 iteraciones para datos normales.

### Por qué converge no significa óptimo global
Aquí está la trampa: K-Means hace descenso por coordenadas en una superficie que no es convexa.

1. **Depende de dónde empiezas.** Si pones los centroides iniciales mal, el algoritmo queda atrapado en un mínimo local. Imagina dos lunas crecientes: con mala inicialización, K-Means parte una luna por la mitad en vez de separar las dos formas.
2. **El problema es NP-hard.** Encontrar la partición global que minimiza J para K arbitrario es computacionalmente intratable cuando n crece. Lloyd solo explora una trayectoria codiciosa.
3. **Suposiciones fuertes.** K-Means asume clusters esféricos, de tamaño similar y con varianza isotrópica. Si tus datos son alargados, con densidades distintas o con mucho ruido, el mínimo global de J ni siquiera corresponde a lo que tú llamarías "buen clustering".

Por eso en la práctica:

- Usas **k-means++** para inicializar: eliges el primer centroide al azar, luego cada siguiente con probabilidad proporcional a D(x)^2, la distancia al centroide más cercano ya elegido. Reduce drásticamente malas inicializaciones.
- Corres el algoritmo 10-100 veces con semillas distintas y te quedas con la de menor inercia final.
- Miras el codo de la inercia vs K para no forzar un K equivocado.

En resumen: K-Means siempre converge porque cada iteración baja monótonamente una función acotada, pero solo garantiza un óptimo local. El óptimo global depende de la inicialización y de que tus datos realmente se parezcan a K bolas en el espacio euclidiano.

## Detección de Anomalías: 
> ¿Cómo puede utilizarse el clustering para detectar anomalías o fraudes en datos financieros? ¿Qué limitaciones tiene K-Means para este tipo de problemas y qué algoritmos alternativos (como DBSCAN o Isolation Forest) se utilizan en la industria para superar esas limitaciones?

Detección de anomalías con clustering es, en esencia, enseñarle al modelo qué es "normal" sin decirle qué es fraude.

En datos financieros funciona así:

- Agrupas transacciones por comportamiento parecido: monto, hora, comercio, dispositivo, velocidad, país. Lo que cae en clusters grandes y densos se asume como comportamiento legítimo.
- Todo lo que queda fuera, lejos del centroide o marcado como "ruido", se etiqueta como atípico. La detección de anomalías "identifica valores atípicos en datos mediante análisis de clústeres, permitiendo detectar fraudes o patrones nuevos" y no requiere datos de entrenamiento conocido. 
- En la práctica bancaria se usa para compras con tarjeta, transferencias AML, y compras empresariales, combinando clustering con técnicas no supervisadas porque el fraude es raro (en el dataset clásico de Kaggle solo 0,172% son fraudes). 

### ¿Por qué K-Means se queda corto?
K-Means fue el primer paso en muchos equipos, pero tiene tres frenos que duelen en fraude:

1. **Tienes que decirle cuántos clusters hay.** "Require the number of clusters to be defined in advance". En fraude no sabes cuántos tipos de comportamiento normal existen, y mucho menos cuántos esquemas nuevos aparecerán. 

2. **Ignora lo pequeño.** "Often overlook small clusters where fraudulent behavior may exist". El fraude vive justo en esos micro-clusters. 

3. **Asume formas bonitas y datos limpios.** Es sensible al ruido y "struggle with highly imbalanced datasets". Las transacciones reales no son esféricas, tienen colas largas, outliers legítimos (viajes, compras grandes) y mucho ruido. 

Por eso la literatura latinoamericana ya advierte: "El empleo de técnicas de agrupamiento como el algoritmo K-means no debe ser vista como la única solución o la más factible". Y en comparativas recientes se concluye que "Kmeans es ideal para datasets simples y bien separables, pero en casos más complicados, DBSCAN, One Class SVM e Isolation Forest ofrecen un rendimiento superior". 

### Lo que usa la industria hoy
#### DBSCAN: densidad en vez de centroides
- No pide k. Agrupa por densidad y todo lo que no alcanza densidad mínima lo marca como ruido, que es justo donde viven las anomalías.
- "DBSCAN es un algoritmo de aprendizaje no supervisado que agrupa datos mediante densidad, detectando anomalías y ruido. Se utiliza en análisis geoespacial y detección de defectos, ofreciendo ventajas sobre K-Means en escenarios complejos". 
- Detecta "anomalías grupales", no solo puntos sueltos, útil para redes de mulas o carruseles de tarjetas. 

En producción se usa como pre-proceso: un paper de Scientific Reports 2025 crea un "hybrid ensemble" donde DBSCAN refina features y luego votan RF, KNN y SVM, logrando 99,5% recall y 99,8% F1 en tarjetas. La clave: "DBSCAN detects anomalies and patterns within the dataset. Its outputs are incorporated as an additional feature". 

Limitaciones: necesitas elegir bien eps y minPts, y con índices su complejidad es O(n log n) frente a O(n²) sin ellos. No escala tan bien como árboles cuando pasas de millones de transacciones.

#### Isolation Forest: aísla en lugar de perfilar
En vez de aprender lo normal, construye árboles aleatorios y mide qué tan rápido queda aislado un punto. El fraude se aísla en pocos cortes.

Ventajas que lo hicieron estándar en banca y fintech:

- "Eficiente y escalable para conjuntos de datos grandes" 
- "No requiere conocimiento previo de la distribución de datos" 
- "Adecuado para datos multidimensionales" 
- "Efectivo en la detección de outliers incluso en conjuntos de datos desequilibrados" 

KNIME lo resume bien: "Isolation Forest directly isolates anomalies, reducing the need for extensive data processing", y "runs very fast with a linear time complexity. In comparison, algorithms like DBSCAN have log-linear time complexity". En su prueba con el dataset de tarjetas europeas obtuvo 97,49% de accuracy, "performing essentially the same as some of our well-performing previous techniques, such as Random Forest and DBSCAN" pero con mejor runtime. 

Limitaciones reales: necesita tuning y es sensible al tamaño de muestra, con riesgo de "swamping y masking effects on the model, where the anomalies get buried within normal data". 

### ¿Cómo se combinan en la práctica?
La industria rara vez usa un solo algoritmo. Los patrones más comunes en 2024-2025:

1. DBSCAN + supervisado: DBSCAN etiqueta densidad/ruido, esa etiqueta entra como feature a XGBoost o RF. Es el enfoque del paper de Nature citado arriba.
2. Clustering + Isolation Forest: "hybrid approach combining clustering and Isolation Forest for anomaly detection in enterprise purchase processes, addressing challenges of unlabelled data". Primero segmentas por tipo de cliente, luego aplicas Isolation Forest dentro de cada segmento para no mezclar comportamientos. 
3. AD-KMeans y variantes adaptativas: para no fijar k, ajustan número de clusters por varianza y densidad, justo para capturar esos pequeños grupos de fraude que K-Means clásico pierde. 

Tabla rápida
|Algoritmo | Idea central | Mejor para |Punto débil en fraude|
|---|---|---|---|
|K-Means| centroides, distancia euclidiana |baseline rápido, datos limpios| necesita k, ignora clusters pequeños, sensible a ruido|
DBSCAN| densidad, marca ruido| fraude grupal, formas irregulares| tuning eps/minPts, escala menos que árboles|
Isolation Forest| aislamiento aleatorio| grandes volúmenes, desbalance extremo, tiempo real| swamping con muchas anomalías, necesita calibrar contamination|


Si estás empezando un proyecto con datos de pagos o transferencias, te conviene:

- No arrancar con K-Means solo. Úsalo solo para explorar segmentos grandes de clientes.
- Prueba DBSCAN para detectar "islas" de comportamiento raro, y alimenta esas etiquetas a tu modelo supervisado.
- Pon Isolation Forest como detector en tiempo real por su velocidad lineal, y recalibra semanalmente porque el fraude cambia.

## Aprendizaje No Supervisado en la Industria: 
> ¿Cuáles son los 3 casos de uso más impactantes del aprendizaje no supervisado en la industria actual? Para cada uno, explica qué tipo de algoritmo se utiliza (clustering, reducción de dimensionalidad, detección de anomalías) y qué valor de negocio genera para las empresas.

El aprendizaje no supervisado ya no es un experimento de laboratorio, es la capa invisible que mantiene funcionando fábricas, bancos y plataformas digitales. En 2025-2026, tres usos concentran casi todo el retorno:

### 1. Detección de anomalías — mantenimiento predictivo, fraude y ciberseguridad
**Qué algoritmo se usa:** la base son modelos sin etiquetas que aprenden qué es "normal" y marcan lo raro. En producción industrial se combinan **K-means** y **TimeSeriesKMeans** para agrupar series de sensores, y luego **Isolation Forest**, **One-Class SVM** y **autoencoders** para puntuar desviaciones. Las redes neuronales (autoencoders, GANs) son las que mejor capturan relaciones no lineales entre temperatura, vibración y ciclos de uso.

Cómo crea valor:

- En fabricación, el modelo vigila miles de puntos IoT y detecta fluctuaciones anómalas de temperatura o vibración antes de que una prensa falle, evitando paradas no planificadas y defectos enviados al cliente.
- En finanzas, agrupa transacciones por geografía, importe y hora; cuando una operación cae en dos o tres clústeres anómalos a la vez, se dispara alerta de fraude en tiempo real.
- En ciberseguridad y operaciones TI multinube, analiza logs y tráfico para encontrar patrones que las reglas estáticas nunca verían, reduciendo la "fatiga de alerta". 

La clave de negocio es pasar de reparar a prevenir: menos downtime, menos pérdidas por fraude, y modelos que se adaptan solos a nuevos ataques sin reescribir reglas. 

### 2. Clustering — segmentación hiper-personalizada de clientes y operaciones
**Qué algoritmo se usa:** **K-means** (y su versión optimizada k-means++) es el workhorse por ser simple, rápido y escalable a millones de registros. Para formas más complejas se usan **DBSCAN**, **HDBSCAN** y **Gaussian Mixture Models**. Es un método "duro": cada cliente cae en un solo clúster según distancia a centroides.

Cómo crea valor:

- Retail y telecom dividen la base en grupos por ingreso, frecuencia y ticket medio. El modelo 2 clásico "Ingreso Anual + Spending Score" sigue siendo el más útil para decidir a quién enviar qué oferta. 
- La segmentación permite campañas dirigidas, pricing dinámico y ubicación óptima de almacenes (ejemplo Qlik: 118 distribuidores reducidos a 5 centroides de almacén).
- Más allá de marketing, se usa para agrupar documentos, imágenes médicas y rutas logísticas, porque el algoritmo es eficiente y se generaliza bien a datos numéricos de baja dimensión. 

El valor es directo: mayor conversión, menor costo de adquisición y mejor uso de inventario, porque dejas de tratar a todos como promedio.

### 3. Reducción de dimensionalidad — comprimir datos para que la IA escale
**Qué algoritmo se usa:** dos familias. **Lineales:** **PCA**, que crea componentes ortogonales que capturan la máxima varianza. **No lineales:** **t-SNE**, **UMAP** y **autoencoders**, que preservan vecindarios locales en embeddings de imágenes, texto o genómica.

Cómo crea valor:

- Eficiencia computacional: menos características significa entrenamiento más rápido para modelos como YOLO y despliegue viable en edge AI con recursos limitados. 
- Visualización y descubrimiento: comprime datos de 512 o 10,000 dimensiones a 2D/3D para que analistas vean clústeres y outliers sin ser expertos en álgebra. 
- Reducción de ruido y costo: al quedarse con la varianza relevante, filtra redundancia, mejora generalización y reduce almacenamiento en la nube. 

En la práctica:

- Visión industrial: las CNN reducen progresivamente feature maps para pasar de píxeles a conceptos ("defecto", "grieta").
- Salud y genómica: PCA identifica biomarcadores en decenas de miles de genes.
- Recomendación: Netflix o Spotify usan factorización de matrices (una forma de reducción) para convertir la matriz dispersa usuario-ítem en características latentes y recomendar en milisegundos. 

PCA sigue siendo el estándar por su simplicidad, mientras UMAP y autoencoders dominan cuando los datos son no lineales como embeddings. 

Resumen rápido

|Caso| Algoritmo principal| Valor de negocio 2026|
|---|---|---|
**Detección de anomalías**| Isolation Forest, One-Class SVM, autoencoders, K-means temporal| -30% a -50% downtime, prevención de fraude en tiempo real|
**Clustering**| K-means / k-means++, DBSCAN, GMM| +15-25% en tasa de respuesta, optimización de stock y rutas|
**Reducción dimensional**| PCA, UMAP, t-SNE, autoencoders| -70% costo cómputo, modelos desplegables en edge, insights visuales

Los tres comparten una lógica: no necesitas etiquetas caras. Aprenden la estructura oculta y la convierten en dinero, ya sea evitando que algo se rompa, vendiendo mejor o haciendo que la IA quepa en un chip.

## Reflexión
Me parece interesante y me soprendió que ahora se pueden generar lo que llama artefactos, que son aplicaciones web que sirven para visualizar de mejor manera la teoría que explica.