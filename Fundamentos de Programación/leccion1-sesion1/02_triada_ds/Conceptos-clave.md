# Apuntes

> Triada de la Ciencia de Datos: 
> * **Estadística** 
> * **Análisis de datos** 
> * **Machine Learning**

## Flujo de Trabajo en un Proyecto Real
### 1. Fase de Análisis Exploratorio de Datos (EDA)
    
* Estadística Descriptiva (Clave)

> Objetivo:
   Entender la estructura de los datos, identificar patrones iniciales, descubrir anomalías, visualizar relaciones entre variables y encontrar insights que guíen las siguientes fases. 

* Proceso
    - Carga Y Resumen inicial.
      - Revisar el estado actual de los datos en bruto para ver su estado.
    - Limpieza de datos.
      - Corregir inconsistencias, eliminación de registros poco confiables.
    - Análisis univariante.
      - Ver si una variable categorica o numérica sigue alguna distribución o si es relevante para el estudio.
    - Análisis Bivariante.
      - Mover valores sobre una sola variable y ver como se comportan las demás si hay relación o no.
    - Detección de Outliers.
      - Un valor atípico (outlier) es un punto de datos que se desvía significativamente de la distribución general o el patrón del resto de la observaciones en un conjunto de datos.
    - Visualización de correlaciones.
  * Obtener un Insight Clave en esta fase
  
### 2. Fase de Estadística Inferencial. 
* Estadística (Clave)
> Objetivo:
    Confirmar si las diferencias observadas en el EDA son estadísticamente significativas y no solo producto del azar.
* Proceso
  - Pruebas de Hipótesis
    - Variables categóricas vs Churn
      - Usar la pruba Chi-cuadrado para determinar si existe una asociación estadísticamente significativa entre variables categóricas.
    - Variables Numéricas vs Churn
      - Usar pruebas t de Student o pruebas no paramétricas como Mann-Whitnery U (si no son normales) para comparar las medias de una variable numérica etre los grupos de churn y no churn.
  - Intervalos de confianza
  - Regresión logística
  - Ciclo dónde se siguen proponiendo nuevas hipótesis o alguna de ellas si brinda información relevante.
### 3. Fase de Feature Engineering ( Ingeniería de Características)
* Análisis de Datos y preparación para Machine Learning (ML)
>Objetivo:
    Crear nuevas variables (features) a partir de las existentes o transfomar las actuales para mejorar el rendimiento del modelo de ML.
* Proceso
  - Codificación de variables categóricas.
    - *One-hot encoding*
      - Convertir las variables categóricas en un formato numérico que los algoritmos de ML puedan entender.
    - Creación de Nuevas Características (dominio del negocio)
      - Proveer al responsable del negocio de nuevos términos a partir de la información recolectada.
    - Escalado de variables numéricas.
      - Estandarizar o normalizar las variables numéricas para que los modelos basados en distancia funcionen mejor.
    - Manejo de desequilirbio de clases
      - Si la proporción de clientes que abandona es mucho menor que los que no, usar técnicas como SMOTE para crear datos sintéticos de la clase minoritaria y balancear el dataset.
    - El resultado es un dataset limpio, transformado y con características más ricas que son más informativas para el modelo.    
### 4. Fase de Entrenamiento del Modelo
* Machine Learning (clave)
> Objetivo:
>   Entrenar uno o varios modelos de ML
* Proceso
  - División de datos
    - Entrenar con un 70% de los datos y ver si las predicciones concuerdan con el 30% restante.
  - Selección del algoritmos
    - Regresión Logística
    - Árboles de decisión / Random Forest / Gradient Boosting (XGBoost, LightGBM)
    - Redes Neuronales (Perceptrón multicapa)
  - Entrenamiento
  - Ajuste de hiperparámtros.
    - Usar técnicas como Grid Search o Random Search con validación cruzada (cross-validation) para encontrar la mejor combinación de hiperparámetros para cada modelo, optimizando una métrica de rendimiento (ej: F1-score).
  - Como resultado tenemos varios modelos entrenados y con hiperparámetros ajustados, listos para ser evaluados.
### 5. Fase de Evaluación del Modelo
* ML, Estadística y Análisis de Datos (Clave)
>Objetivo:
    Medir el rendimiento de los modelos en datos no vistos y seleccionar el mejor modelo para el problema del negocio.
* Proceso
  - Predicción en el conjunto de Prueba
    - Usar los modelos entrenados para hacer predicciones sobre el conkinto de prueba, que no se ha utilizado en el entrenamiento.
  - Métricas de evaluación.
    - Matriz de Confusión
    - Precisión y Exhaustividad.
    - F1-Score.
    - Curva ROC *Receiver Operating Characteristic* y área bajo la curva ROC (AUC-ROC)
    - Lift Chart
  - Importancia de las Características
  - Análisis de Errores
  - Comparación de modelos.
  - Si se logra un modelo que logra un AUC-ROC del 0.85, significa que el 85% de las veces predice correctamente.
### 6. Fase de Despliegue y Monitoreo
* ML, Análisis de Datos y Estadística. (Clave)
> Objetivo:
    Integrar el modelo predictivo en los sistemas de la empresa y asegurar que siga siendo efectivo a lo largo del tiempo.
* Proceso
  - Despliegue
    - Contenedorizar el modelo (ej: Docker) y desplegarlo como uun servicio web (API REST) en una plataforma en la nube (AWS, GCP, Azure) o en un servidor local.
    - Integrar este servicio con los sitemas CRM de la empresa o con herramientas de marketing.
  - Generación de predicciones.
  - Acciones de Negocio
  - Monitoreo del Modelo
    - monitoreo del rendimiento
    - Detección de Deriva de Datos.
    - Re-entrenamiento
#### Resultado Final
La empresa puede tomar decisiones basadas en las predicciones de ML, traduciéndose en beneficios para la misma.

## Resumen
* La estadística es teoría que ayuda a analizar datos con los cuales prepararemos motores que aprenderán y predecirán.

* Los datos son elementos brutos que una vez procesados y analizados con métodos científicos y algoritmos, se convierten en la base para obtener conocimientos y tomar decisiones informadas.

* Los outliers pueden tener un impacto considerable en el análisis de datos y en los modelos de machine learning distorsionando el análisis, a veces pueden deberse a una mala entrada de datos, no obstante también pueden ser de gran interés coomo al momento de detectar fraudes.
  
* Un outlier puede detectarse mediante visualización, métodos estadísticos (Z-Score, rango intercuartíl) y Algoritmos de Machine Learning ( Isolation Forest o Local Outlier Factor.)

* Un outlier puede eliminarse, transformarse, imputarse, mantenerse o hacerse menos sensible.

### Palabras / Conceptos Clave
> Dato: unidad más fundamental y cruda de información, una representación simbólica de un atributo, hecho, suceso o entidad

> Ciencia: Es un sistema de conocimiento que se basa en la observación,le experimentación y el razonamiento para obtener principios y leyes generales.

**Características de la ciencia**
* Sistemática y Metódica
* Empírica.
* Objetica
* Verificable y Falsable
* Acumulativa
* Predictiva

> Ciencia de datos: es un campo interdisciplinario que utiliza métodos científicos, procesos, algoritmos y sistemas para extraer conocimiento e información significativa de datos estructurados y no estructurados.
> 
> Su **propósito final** es aplicar estos conocimientos para obtener conclusiones, tomar decisiones y resolver problemas complejos en diversos dominios.

>Churn:  también conocido como tasa de abandono o tasa de rotación, se refiere a la proporción de clientes o usuarios que dejan de utilizar un producto, servicio o empresa durante un periodo de tiempo determinado.

> SMOTE *Synthetic Minority Over-sampling Technique* es una técnica de sobremuestreo utilizada para tratar el problema del desequilibrio de clases en un conjunto de datos, especialmente cuando la clase minoritaria tiene muy pocos ejemplos. SMOTE funciona creando nuevas instancias sintéticas (no duplicadas) de la clase minoritaria interpolando entre los ejemplos existentes de esa clase y sus vecinos más cercanos. 

> SVM: *Support Vector Machine* es un algoritmo de aprendizaje automático supervisado utilizado principalmente para clasificación (aunque también para regresión). Su objetivo es encontrar el hiperplano óptimo que mejor separa las clases de datos en un espacio n-dimensional, maximizando el "margen" (la distancia entre el hiperplano y los puntos de datos más cercanos de cada clase, llamados "vectores de soporte").

> K-NN *K-Nearest Neighbors* es un algoritmo de aprendizaje supervisado no paramétrico utilizado tanto para clasificación como para regresión. Para clasificar un nuevo punto de datos, K-NN busca los "K" puntos de datos más cercanos en el conjunto de entrenamiento (basado en una medida de distancia, como la euclidiana) y asigna al nuevo punto la clase más frecuente entre esos K vecinos (para clasificación) o el promedio de sus valores (para regresión).

> Random Forest es un algoritmo de aprendizaje automático que pertenece a la categoría de ensamblaje (ensemble learning). Construye múltiples árboles de decisión durante el entrenamiento y produce la clase que es la moda de las clases (clasificación) o la predicción media (regresión) de los árboles individuales. Introduce aleatoriedad en dos puntos: al seleccionar el subconjunto de datos para cada árbol y al seleccionar el subconjunto de características para dividir cada nodo, lo que reduce el sobreajuste (overfitting) y mejora la precisión.

> Gradient Boosting es un algoritmo de ensamblaje, pero a diferencia de Random Forest, construye los árboles de decisión de forma secuencial y aditiva. Cada nuevo árbol intenta corregir los errores o residuos de los árboles anteriores. Utiliza un enfoque de "descenso de gradiente" para optimizar el modelo, lo que lo hace muy potente para lograr alta precisión, aunque puede ser más propenso al sobreajuste si no se ajusta bien.

> Grid Search es una técnica para la optimización de hiperparámetros de un modelo de machine learning. Consiste en definir un conjunto predefinido de valores para cada hiperparámetro relevante y luego probar sistemáticamente todas las combinaciones posibles de estos valores. El modelo se entrena y evalúa con cada combinación para encontrar el conjunto de hiperparámetros que produce el mejor rendimiento.

> Random Search es una técnica para la optimización de hiperparámetros que selecciona aleatoriamente combinaciones de hiperparámetros de los rangos o distribuciones definidas para cada uno.

> F1-score Es una métrica de evaluación utilizada para medir la precisión de un modelo de clasificación, especialmente útil en problemas con desequilibrio de clases. El F1-score es la media armónica de la precisión (precision) y la exhaustividad (recall). Un valor alto de F1-score indica que el modelo tiene tanto una baja tasa de falsos positivos como una baja tasa de falsos negativos, ofreciendo un equilibrio entre ambas métricas.

> Accuracy *Exactitud* es el porcentaje de predicciones correctas sobre el total de predicciones.

> Precision *Precisión*  De todas las veces que el modelo predijo una clase positiva, ¿cuántas fueron realmente positivas? (TP/(TP+FP)

> Recall (Sensibilidad o Exhaustividad): De todas las veces que el modelo predijo una clase positiva, ¿cuántas fueron realmente positivas? (TP/(TP+FN)).

> F1-Score: la medida armónica de la precisión y el Recall.

> AUC-ROC: mide la capacidad de un modelo para distinguir entre clases. Un valor de 0.5 indica que el modelo no tiene capacidad de distinción (como lanzar una moneda), y 1.0 indica una distinción perfecta.

> MAE *Mean Absolute Error*: el promedio de las diferencias absolutas entre los valores predichos y los valores reales.

> MSE *Mean Squared Error*: el promedio de las diferencias al cuadrado entre los valores predichos y los valores reales.

> RMSE *Root Mean Squared Error* : la raíz cuadrada del MSE.

> R2 *Coeficiente de Determinación*: Indica la proporción de la varianza en la variable dependiente que es predecible a partir de la(s) variable(s) independiente(s). Un valor de 1.0 significa que el modelo explica toda la varianza.

* Autor: Kevin MJ