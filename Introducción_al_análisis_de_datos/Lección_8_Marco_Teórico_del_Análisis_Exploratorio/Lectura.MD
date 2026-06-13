# Lectura: El Analista Profesional — Buenas Prácticas para un EDA de Calidad

El objetivo de esta lectura complementaria es introducir los principios y estándares profesionales que elevan un Análisis Exploratorio de Datos de funcional a excepcional, diferenciando el trabajo de un analista junior del de un profesional de alto impacto en la industria.

## 7.1 El Pipeline del EDA como Proceso Repetible.
En la industria, un Análisis Exploratorio de Datos no es un ejercicio de una sola vez: es un proceso estandarizado y repetible que se ejecuta cada vez que llega un dataset nuevo. Los equipos de Data Science más eficientes no improvisan el EDA; lo tienen documentado como un pipeline con etapas definidas que garantizan consistencia y trazabilidad.

La diferencia entre un EDA amateur y uno profesional no está en la cantidad de gráficos generados, sino en la lógica que los conecta. Cada visualización debe responder una pregunta concreta, y cada pregunta debe llevar a una decisión documentada sobre el dataset.

Un pipeline de EDA profesional sigue siempre el mismo orden:

* **Carga y diagnóstico inicial:** Verificar estructura, tipos de dato y valores nulos antes de cualquier operación.
  
* **Saneamiento:** Imputar, normalizar y convertir tipos. El dataset debe quedar íntegro antes de explorar.

* **Análisis univariable:** Estudiar cada variable de forma independiente con histogramas y boxplots.

* **Tratamiento de outliers:** Aplicar IQR o el criterio estadístico adecuado con documentación de la decisión.

* **Análisis bivariable y multivariable:** Explorar relaciones con matrices de correlación y gráficos por pares.

* **Comparación por categoría:** Segmentar las variables numéricas por grupos para encontrar patrones ocultos.

* **Narrativa visual:** Seleccionar los 3 a 5 hallazgos más relevantes y construir gráficos diseñados para comunicar.

## 7.2 Documentación del EDA: El Comentario como Herramienta Profesional.
Uno de los hábitos más infravalorados en el análisis de datos es la documentación inline. Un notebook sin comentarios es una caja negra: solo su autor sabe por qué se tomó cada decisión. En un entorno colaborativo o en un proyecto que se retoma semanas después, esa caja negra se convierte en un obstáculo.

El estándar profesional en Google Colab y Jupyter Notebook combina tres niveles de documentación:

* Celdas Markdown para contexto narrativo: Explican el por qué de cada sección, no solo el qué.
* Comentarios de código para decisiones técnicas: Documentan directamente en la línea donde ocurre la decisión.

``` python
# Se utiliza la mediana en lugar de la media porque el histograma

# mostró una distribución sesgada a la derecha con outliers evidentes.

df['price'] = df['price'].fillna(df['price'].median())

# IQR aplicado únicamente a 'price' porque es la única columna

# con valores atípicos confirmados por el boxplot previo.

Q1 = df['price'].quantile(0.25)

Q3 = df['price'].quantile(0.75)

IQR = Q3 - Q1

df = df[(df['price'] >= Q1 - 1.5 * IQR) & (df['price'] <= Q3 + 1.5 * IQR)]

```

Conclusiones al final de cada sección: Un párrafo breve que resume lo observado y justifica el siguiente paso.

Este nivel de documentación no solo beneficia a otros; te beneficia a ti mismo cuando regresas al análisis días después.

## 7.3 Selección Estratégica del Gráfico Correcto.
Una de las habilidades más valoradas en Data Science no es saber programar todos los tipos de gráficos, sino saber cuándo usar cada uno. Elegir el gráfico equivocado puede ocultar el hallazgo más importante o, peor aún, generar una conclusión incorrecta.

La siguiente matriz de decisión resume los criterios aplicados durante el Módulo 3:

| Objetivo del análisis	| Variable(s) involucrada(s) | Gráfico recomendado|
| :--- | :---: | ---: |
| Ver la distribución de una variable	| 1 numérica | Histograma con KDE |
Detectar outliers y cuartiles	| 1 numérica	| Boxplot | 
Comparar distribuciones entre grupos	| 1 numérica + 1 categórica	| Boxplot por categoría |
Ver la relación entre dos variables	| 2 numéricas	| Scatter plot / Regplot |
Resumir correlaciones múltiples | N numéricas	| Heatmap de correlación |
Comparar magnitudes entre categorías |	1 numérica + 1 categórica	| Barras horizontales |
Mostrar composición proporcional	| 1 categórica	| Gráfico de dona|
Mostrar evolución en el tiempo |	1 numérica + tiempo	|Gráfico de líneas|

Memorizar esta tabla no es el objetivo; internalizarla sí. Con el tiempo, la selección del gráfico correcto se vuelve instintiva porque responde siempre la misma pregunta: ¿qué quiero que mi audiencia descubra?

## 7.4 Reproducibilidad: El Estándar Invisible que Importa.
En la industria, un análisis que no puede ser reproducido por otro analista no tiene valor. La reproducibilidad es la capacidad de que cualquier persona con acceso al notebook y al dataset pueda obtener exactamente los mismos resultados ejecutando el código de principio a fin.

Para garantizarla en Google Colab, se siguen estas prácticas:

Fijar la semilla aleatoria cuando se usen operaciones que involucren aleatoriedad:

``` python
import numpy as np

np.random.seed(42) # Garantiza resultados reproducibles en operaciones aleatorias

```

Organizar el notebook en orden de ejecución lógica: Cada celda debe poder ejecutarse en secuencia desde la primera hasta la última sin errores. Las celdas fuera de orden son el error más común en notebooks compartidos.

Especificar versiones de librerías en proyectos críticos:

``` python
# Verificar versiones instaladas

import pandas as pd

import seaborn as sns

print(f"Pandas: {pd.__version__}")

print(f"Seaborn: {sns.__version__}")

```

Nombrar variables de forma descriptiva: df_airbnb_clean comunica más que df2. precio_promedio_por_alcaldia es más claro que resultado. Esta práctica, tomada del PEP 8 de Python, aplica igual al análisis de datos que a la programación general.

La reproducibilidad no es un lujo de proyectos grandes: es el hábito que distingue a un analista confiable de uno cuyo trabajo no puede verificarse. En equipos de Data Science, un notebook no reproducible es equivalente a no entregar el análisis.