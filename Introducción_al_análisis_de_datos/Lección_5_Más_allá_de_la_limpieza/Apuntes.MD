# Apuntes
## Prompt
Valores Atípicos: ¿Cuáles son las principales estrategias para detectar y tratar valores atípicos (outliers) en un dataset numérico? Explica en qué situaciones es correcto eliminarlos con el método del Rango Intercuartílico (IQR), cuándo es preferible aplicar una transformación como logaritmo o raíz cuadrada, y cuándo lo más adecuado es conservarlos porque representan fenómenos reales del negocio.
## Respuesta

Los valores atípicos no son un error que siempre hay que borrar, son una señal. Para un aprendiz de ciencia de datos en Python, la clave está en detectar bien, entender por qué aparecen y elegir el tratamiento según el objetivo del análisis, no según una receta.

### Principales estrategias para detectar
1. Visuales, rápidas para explorar

    * Boxplot: muestra directamente los puntos fuera de los bigotes de Tukey.
    * Histograma y KDE: revelan colas largas.
    * Scatter plot: útil cuando el outlier solo aparece en la relación entre dos variables.

    ```python
    import seaborn as sns
    sns.boxplot(x=df['ingreso'])
    ```

2. Estadísticas univariadas, clásicas

    * **Z-score:** (x - media) / desviación. Funciona si la distribución es aproximadamente normal. Marca como outlier si |z| > 3.
    * **IQR:** no asume normalidad. Calcula Q1, Q3, IQR = Q3 - Q1. Límites = [Q1 - 1.5IQR, Q3 + 1.5IQR].
    * **MAD (Median Absolute Deviation):** más robusto que z-score con datos sesgados. (x - mediana) / MAD.

3. Basadas en modelos, para casos complejos

    * **Isolation Forest, Local Outlier Factor, DBSCAN:** detectan outliers multivariantes o en alta dimensión, donde un valor parece normal solo pero es raro en combinación.

    ```python
    from sklearn.ensemble import IsolationForest
    clf = IsolationForest(contamination=0.01)
    df['outlier'] = clf.fit_predict(df[['x','y']])
    ```

### ¿Cuándo eliminar con IQR?
El IQR es correcto cuando el outlier es ruido, no señal.

Úsalo para eliminar si:

* El valor es imposible en el dominio. Un sensor que marca -5 grados en una variable que solo puede ser positiva, o una edad de 250 años.
* Es un error de captura claro, duplicado, o de migración de datos, y tienes pocos casos.
* Estás entrenando un modelo muy sensible a la escala como regresión lineal o k-means, y el dataset es grande, por lo que perder 1% no sesga el negocio.
* La distribución no es normal y quieres un método no paramétrico.

No lo uses como eliminación automática en series temporales o datos financieros, porque el IQR trata cada punto de forma aislada y borra picos legítimos.

### ¿Cuándo transformar con log o raíz cuadrada?
Transformar no borra, comprime la escala. Es preferible cuando el outlier es real pero distorsiona el modelo.

Elige **logaritmo (np.log1p)** si:

* La variable es estrictamente positiva y tiene varios órdenes de magnitud. Ingresos, montos de transacción, tiempo de sesión.
* Quieres interpretar cambios porcentuales y estabilizar la varianza para modelos lineales.
* Hay cola derecha muy larga y necesitas acercarte a normalidad.

Elige **raíz cuadrada (np.sqrt)** si:

* La variable incluye ceros, log(0) falla y log1p sigue siendo muy agresivo.
* Trabajas con conteos, número de clics, defectos por lote, donde la varianza crece con la media.

```python
import numpy as np
df['ingreso_log'] = np.log1p(df['ingreso'])  # log(1+x) evita problemas con 0
df['conteo_sqrt'] = np.sqrt(df['conteo'])
```

Transformar es reversible, por eso es más seguro que eliminar cuando no estás seguro del origen.

### ¿Cuándo conservarlos porque son el negocio?
Aquí está la diferencia entre limpiar datos y censurar la realidad. Conserva, e incluso aíslalos en una variable indicadora, cuando:

* El objetivo es detección de anomalías. Fraude, intrusión, fallo de máquina, churn extremo. El outlier es la etiqueta.
* Representan eventos raros pero de alto impacto. Un cliente que compra 100 veces más que el promedio, un día de pico de demanda.
* Eliminarlos hace que tu modelo subestime riesgo e ingresos.
* El dataset es pequeño. Quitar 5 puntos de 200 cambia la media más que dejarlos.
* El modelo que usarás es robusto por diseño. Árboles, Random Forest, XGBoost, o regresiones con Huber loss toleran outliers mejor que eliminarlos.

En estos casos, en vez de borrar, haz:

* Winsorizar: recortar al percentil 1 y 99 en lugar de eliminar.
* Crear una bandera es_outlier para que el modelo aprenda del patrón.
* Usar métricas robustas como mediana e IQR en el reporte, no media y desviación.

```python
from scipy.stats.mstats import winsorize
df['ingreso_w'] = winsorize(df['ingreso'], limits=[0.01, 0.01])
```
Una regla práctica para decidir: primero pregunta por qué existe. Si es error, IQR y elimina. Si es real pero molesta a la escala, transforma. Si es real y valioso, consérvalo y modela con herramientas robustas.

## Prompt
Correlación e Interpretación: ¿Qué diferencia existe entre el coeficiente de correlación de Pearson y el de Spearman, y cuándo se recomienda usar cada uno? Proporciona un ejemplo de una correlación alta que sea espuria (falsa) y explica por qué la regla "correlación no implica causalidad" es fundamental en el análisis de datos.

## Respuesta

Pearson y Spearman miden cosas parecidas pero con lentes distintos. Si los confundes, puedes declarar una relación lineal donde solo hay orden, o perder una señal fuerte porque no es una línea recta.

### Pearson vs Spearman, en qué se diferencian

**Pearson (r)**

* Mide relación lineal. ¿Si X sube, Y sube de forma proporcional y constante?
* Trabaja con los valores reales. Es sensible a outliers y asume que ambas variables son continuas y con distribución aproximadamente normal.
* Rango de -1 a 1. r = 0.9 significa que los puntos se alinean cerca de una recta.
**Spearman (ρ, rho)**

* Mide relación monotónica. ¿Si X sube, Y tiende a subir o a bajar, aunque sea en curva?
* Trabaja con los rangos, no con los valores. Primero ordena los datos y luego calcula Pearson sobre esos rangos.
* No asume normalidad, tolera outliers y funciona con datos ordinales o con relaciones no lineales.

```Python
from scipy.stats import pearsonr, spearmanr
r_pearson, _ = pearsonr(df['x'], df['y'])
rho_spearman, _ = spearmanr(df['x'], df['y'])
```

#### Cuándo usar cada uno
Usa **Pearson** cuando:

* Esperas una relación lineal y ambas variables son continuas.
* No hay outliers extremos y quieres cuantificar fuerza y dirección para modelos lineales.

Usa **Spearman** cuando:

* La relación es consistente pero curva. Por ejemplo, esfuerzo vs rendimiento, que mejora rápido al inicio y luego se estanca.
* Tienes datos ordinales, encuestas tipo Likert, o rankings.
* Hay outliers que no quieres eliminar pero que distorsionarían Pearson.
* La distribución es muy sesgada y no quieres transformar.

> Una regla práctica: grafica primero. Si el scatter parece una nube en diagonal recta, Pearson. Si parece una escalera o una curva que siempre sube, Spearman.

### Correlación alta que es espuria
Un clásico verificable: la correlación entre el consumo per cápita de queso y el número de muertes por enredarse en las sábanas en EE. UU. entre 2000 y 2009 es de r ≈ 0.95. Es altísima, casi perfecta.

Es espuria porque:

* No hay mecanismo causal plausible.
* Ambas series tienen una tendencia temporal común. Crecen ligeramente con la población y con el paso de los años.
* Hay una tercera variable oculta, el tiempo, que empuja a ambas.

Otro ejemplo más de negocio: ventas de helado y ahogamientos en piscinas tienen r > 0.8 en verano. No es que el helado cause ahogamientos, la temperatura alta aumenta ambas cosas.

### Por qué "correlación no implica causalidad" es fundamental
En análisis de datos tomamos decisiones con esto. Confundir correlación con causa lleva a tres errores típicos:

* **Variable de confusión.** X e Y se mueven juntas porque Z las mueve a ambas. Si intervienes sobre X, Y no cambia.
* **Causalidad inversa.** Crees que X causa Y, pero Y causa X. Más reseñas positivas correlacionan con más ventas, pero son las ventas las que generan más reseñas.
* **Coincidencia por azar.** Con suficientes variables, siempre encontrarás pares con r alto. Sin teoría ni experimento, es ruido.

Para un científico de datos, la correlación es solo una alarma. Te dice dónde mirar, no qué hacer. Antes de recomendar una acción, necesitas: temporalidad clara, un mecanismo razonable, control de confusores, y idealmente un experimento A/B o un diseño causal. Sin eso, un modelo basado en una correlación espuria funcionará en el histórico y fallará en producción.

## Prompt
Narrativa Visual: ¿Qué es el storytelling con datos y cómo se aplica en la presentación de un Análisis Exploratorio de Datos (EDA) para una audiencia no técnica? Menciona al menos 3 principios clave para transformar un conjunto de gráficos estadísticos en una narrativa visual coherente y convincente.
## Respuesta

Storytelling con datos no es decorar gráficos, es traducir números en decisiones. Es tomar el EDA, que para ti es una exploración abierta, y convertirlo en un camino cerrado para alguien que no sabe qué es un p-valor ni quiere saberlo.

Para una audiencia no técnica, el EDA no se presenta como "miren todos los plots que hice". Se presenta como una respuesta a una pregunta de negocio, contada con imágenes.

### Cómo se aplica en un EDA
Piensa el EDA en dos fases. La primera es para ti: generas 50 gráficos, pruebas distribuciones, outliers, correlaciones. La segunda es para ellos: seleccionas 4 o 5 visuales que sostengan una historia.

El flujo típico que funciona:

1. **Empieza con contexto, no con código.** Una diapositiva con la pregunta: "¿Por qué las cancelaciones subieron el último trimestre
2. **Muestra el panorama.** Un solo gráfico grande que responda "qué está pasando". Por ejemplo, tendencia de cancelaciones en el tiempo.
3. **Profundiza en el porqué.** Dos gráficos que rompan ese total por segmentos relevantes. No muestres todas las variables, solo las que explican la variación.
4. **Cierra con implicación.** Qué significa y qué opciones hay, sin jerga estadística.

No lleves histogramas de todas las columnas ni matrices de correlación completas. Si un gráfico no mueve la historia hacia adelante, queda fuera.

### 3 principios clave para transformar gráficos en narrativa
1. **Un mensaje por gráfico, escrito en el título**
Los no técnicos no leen ejes, leen títulos. Cambia "Distribución de tiempo de entrega" por "El 70% de los retrasos ocurren en solo dos rutas". El título es tu conclusión, el gráfico es la prueba.

En Python, fuerza esa idea:

``` Python
ax.set_title("Los clientes nuevos cancelan 3 veces más que los recurrentes", 
             loc='left', fontsize=14, weight='bold')
```

Elimina leyendas innecesarias, pon las etiquetas directamente sobre las barras o líneas.

2. **Guía la mirada con jerarquía visual**
El cerebro sigue contraste, color y posición. Usa un color intenso solo para lo importante, gris para el contexto. Quita rejillas, bordes y fondos. Añade anotaciones que expliquen el pico, no dejes que lo adivinen.

Principio: menos ruido, más señal. Si tienes que explicar un gráfico por más de 20 segundos, rediseñalo. Una flecha con "cambio de política en marzo" vale más que tres párrafos.

3. **Estructura de historia, no de carpeta**
Ordena los visuales como una mini trama:

* **Situación:** dónde estamos hoy.
* **Complicación:** qué cambió y dónde duele.
* **Resolución:** qué segmentos lo explican y qué podríamos probar.

Esto evita el "tour de gráficos". Cada visual responde a la pregunta que dejó el anterior. Termina siempre con una implicación accionable, no con "se requiere más análisis". Por ejemplo: "Si reducimos el tiempo en esas dos rutas, atacamos el 70% del problema".

Un cuarto principio que ayuda mucho: conoce el vocabulario de tu audiencia. Habla de clientes, ingresos, esperas, no de varianza, kurtosis o R². Si necesitas mostrar incertidumbre, dilo como "este patrón se repite en 8 de cada 10 meses", no como intervalo de confianza.

Con esto, tu EDA deja de ser una carpeta de outputs y se convierte en una conversación visual donde cada gráfico es una frase, y juntos forman un argumento que una persona no técnica puede recordar y defender.

## Reflexión

Lo que más me soprende de la herramienta es la habilidad para exponer de forma clara y concisa conceptos que me parecían hasta hace poco confusos.