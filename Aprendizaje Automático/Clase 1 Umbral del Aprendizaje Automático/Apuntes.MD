# Apuntes
## Paradigmas del ML: 
> ¿Cuáles son las diferencias fundamentales entre el Aprendizaje Supervisado, No Supervisado y por Refuerzo? Para cada paradigma, proporciona un ejemplo de caso de uso real en la industria y explica qué tipo de datos y objetivos requiere cada uno.

Los tres paradigmas responden a preguntas distintas sobre los datos. La diferencia fundamental está en la señal que guía el aprendizaje: etiquetas explícitas, estructura implícita, o recompensa por interacción.

### Aprendizaje Supervisado
Aprende una función que mapea entradas X a salidas Y usando ejemplos etiquetados. El modelo minimiza el error entre su predicción y la etiqueta correcta.

**Datos que requiere:** Pares entrada-salida. Es el paradigma más caro en datos porque necesita etiquetado humano o de sistemas.

* Estructurados: tablas con features y target
* No estructurados: imágenes con clase, audio con transcripción, texto con sentimiento.

**Objetivo:** Generalizar para predecir Y en datos nuevos. Tareas típicas: clasificación y regresión.

**Caso de uso real en industria: Detección de fraude en pagos**
Visa, Mastercard y fintechs entrenan modelos supervisados con millones de transacciones históricas etiquetadas como fraude o legítima.

* Datos: monto, hora, comercio, geolocalización, historial del usuario, dispositivo, etiqueta binaria de fraude confirmada por investigación.
* Objetivo: maximizar recall de fraude con baja tasa de falsos positivos, porque bloquear a un cliente legítimo cuesta dinero.
* Por qué es supervisado: la etiqueta existe y es el criterio de verdad. El modelo aprende patrones que preceden a chargebacks.

Otros ejemplos: diagnóstico por imagen médica, scoring crediticio, predicción de churn.

### Aprendizaje No Supervisado
Aprende la estructura interna de los datos sin etiquetas. No hay respuesta correcta, solo patrones.

**Datos que requiere:** Solo entradas X, sin Y. Puede trabajar con datos crudos y masivos.

* Vectores de características, embeddings, secuencias

**Objetivo:** Descubrir organización oculta. Tareas típicas: clustering, reducción de dimensionalidad, detección de anomalías, modelado de temas.

**Caso de uso real en industria: Segmentación de clientes en retail y streaming**
Amazon, Mercado Libre, Netflix o Spotify agrupan usuarios por comportamiento sin decirles a priori qué segmentos existen.

* Datos: historial de compras/vistas, tiempo de sesión, categorías visitadas, frecuencia, valor de ticket. Sin etiquetas de segmento.
* Objetivo: descubrir arquetipos naturales, por ejemplo comprador de ofertas nocturno, explorador de novedades tech, usuario de fin de semana familiar.
* Uso: personalizar homepages, enviar promociones distintas, optimizar inventario.

El modelo no predice una clase dada, revela que la clase existe. Es exploratorio y suele alimentar después un sistema supervisado.

Otros ejemplos: detección de anomalías en logs de servidores, compresión con PCA/autoencoders, topic modeling en reseñas.

### Aprendizaje por Refuerzo
Aprende por prueba y error interactuando con un entorno. No hay pares X,Y ni distribución fija; hay estados, acciones y una señal de recompensa retardada.

**Datos que requiere:** Experiencia generada por interacción. No es un dataset estático, es una política que explora.

* Estado del entorno, acción tomada, recompensa recibida, siguiente estado

**Objetivo:** Aprender una política que maximice la recompensa acumulada a largo plazo, no la ganancia inmediata.

**Caso de uso real en industria: Optimización de recomendaciones y precios dinámicos**
YouTube, TikTok y plataformas de ads usan bandits y RL para decidir qué mostrar.

* Datos: estado del usuario = historial reciente, contexto, hora; acciones = qué video/anuncio mostrar; recompensa = watch time, click, retención, ingreso.
* Objetivo: maximizar engagement a largo plazo. Un clickbait puede dar recompensa inmediata pero dañar retención, el RL aprende a evitarlo.
* Por qué es refuerzo: no hay etiqueta correcta de mejor video. La calidad se revela solo después de mostrarlo y observar la reacción.

Otro caso clásico: trading algorítmico, control de robots en almacenes de Amazon, ajuste de temperatura en data centers de Google DeepMind que redujo consumo energético.

#### Comparación rápida
|Paradigma |Señal de aprendizaje| Datos necesarios| Objetivo principal| Riesgo típico|
|--------------|--------------|--------------|--------------|--------------|
|**Supervisado** | Etiquetas Y| Pares X,Y etiquetados |Predecir Y con precisión | Sobreajuste y sesgo de etiquetado|
**No Supervisado**| Estructura en X| Solo X sin etiquetas| Descubrir patrones/grupos| Interpretabilidad y evaluación subjetiva|
**Refuerzo** |Recompensa escalar| Interacción estado-acción-recompensa| Maximizar recompensa acumulada|Exploración insegura y recompensa mal diseñada|

En la práctica se combinan. Un sistema de recomendación puede usar no supervisado para crear embeddings de usuarios, supervisado para predecir probabilidad de click, y refuerzo para decidir la secuencia de recomendaciones que maximiza la sesión completa.

## Redes Neuronales

> ¿Qué es una red neuronal artificial y cómo se relaciona con el funcionamiento del cerebro humano? Explica qué son las épocas de entrenamiento, la función de pérdida y el descenso de gradiente, y cómo estos tres conceptos trabajan juntos para que un modelo "aprenda".

Una red neuronal artificial es una función compuesta por muchas funciones simples encadenadas. No es un cerebro, es una aproximación matemática inspirada en él.

### ¿Qué es y qué tiene que ver con el cerebro?
El cerebro biológico tiene neuronas que reciben señales por dendritas, las integran y si superan un umbral disparan un impulso por el axón hacia otras neuronas. La fuerza de las conexiones sinápticas cambia con la experiencia.

Una red artificial copia esa idea a nivel muy abstracto:

* **Neurona artificial:** recibe números de entrada, los multiplica por pesos, suma un sesgo y aplica una función de activación no lineal. Es un salida = activación( pesos·entrada + sesgo ).
* **Capas:** las neuronas se organizan en capas. La capa de entrada recibe los datos, las capas ocultas transforman representaciones y la capa de salida produce la predicción.
* **Pesos:** son los análogos de las sinapsis. Aprender es ajustar esos pesos.

La relación es inspiracional, no literal. El cerebro tiene ∼86 mil millones de neuronas con pulsos, química, plasticidad estructural y consumo bajísimo. Una red artificial es estática, síncrona, entrenada con álgebra matricial y necesita millones de ejemplos etiquetados para tareas que un niño aprende con pocos. La metáfora ayuda a intuir, pero el mecanismo es optimización numérica.

### Los tres engranajes del aprendizaje
Para que la red aprenda necesita medir qué tan mal lo hace, decidir cómo corregir y repetir.

#### 1. Épocas de entrenamiento
Una época es una pasada completa por todo el conjunto de entrenamiento.

En la práctica no se procesa todo de golpe. Se divide en lotes o batches. Un paso de actualización usa un batch. Cuando se han visto todos los batches, se completa 1 época.

* 1 época = ver cada ejemplo una vez
* 5 épocas = ver cada ejemplo cinco veces
  
Pocas épocas = subajuste, el modelo no captura patrones. Demasiadas épocas = sobreajuste, memoriza el entrenamiento y falla en datos nuevos. Por eso se usa validación temprana.

#### 2. Función de pérdida
Es la brújula. Convierte el error del modelo en un número escalar.

Si la red predice una probabilidad y la etiqueta real es otra, la pérdida cuantifica la distancia. Ejemplos comunes:

* **MSE:** para regresión, promedio de errores al cuadrado
* **Cross-Entropy:** para clasificación, penaliza confianza equivocada
Cuanto menor la pérdida, mejor el ajuste. La pérdida se calcula sobre un batch y luego se promedia en la época.

#### 3. Descenso de gradiente
Es el motor de corrección. La pérdida es una función de todos los pesos de la red. El gradiente es el vector de derivadas parciales que indica la dirección de mayor aumento de la pérdida.

Descenso de gradiente hace lo opuesto: mueve cada peso un pequeño paso en la dirección que reduce la pérdida.

peso_nuevo = peso_viejo - tasa_aprendizaje * gradiente

La tasa de aprendizaje controla el tamaño del paso. Grande acelera pero puede divergir. Pequeña es estable pero lenta.

El gradiente se obtiene con backpropagation, que aplica regla de la cadena desde la salida hacia la entrada para propagar el error capa por capa.

### Cómo trabajan juntos
El ciclo es iterativo:

1. **Forward pass:** entra un batch, la red calcula predicciones con los pesos actuales.
2. **Pérdida:** se compara predicción vs etiqueta real con la función de pérdida.
3. **Backward pass:** backpropagation calcula el gradiente de la pérdida respecto a cada peso.
4. **Actualización:** descenso de gradiente ajusta los pesos.
5. **Repetir** para el siguiente batch. Al terminar todos los batches se cierra una época.

Con cada época la pérdida de entrenamiento tiende a bajar. El modelo va esculpiendo sus pesos para mapear entradas a salidas.

Aprender aquí no es comprender, es optimizar. La red no sabe qué es un gato, ajusta millones de parámetros hasta que la función de pérdida sobre imágenes de gatos es mínima.

> Si quieres, puedo mostrarte este ciclo con un ejemplo numérico pequeño de una neurona con 2 entradas para que veas cómo cambian los pesos paso a paso.

Neurona mínima de 2 entradas Sin activación no lineal, para mostrar la aritmética.

### Configuración

Neurona lineal:

$$\hat{y} = w_1 x_1 + w_2 x_2 + b$$

Pérdida MSE para un ejemplo:

$$ L = \frac{1}{2}(\hat{y} - y)^2 $$

Hiperparámetros:
- Pesos iniciales: 
$$ w_1 = 0.1, w_2 = 0.2, b = 0.0 $$
- Tasa de aprendizaje: 
$$\eta = 0.1$$

Dataset de 2 ejemplos (1 época = ver los 2 ejemplos)

- Ejemplo A: 
$$x_1=2, x_2=3, y=1$$
- Ejemplo B: 
$$x_1=1, x_2=-1, y=0$$

#### Paso 1. Forward con Ejemplo A

$$ \hat{y}_A = 0.1\cdot2 + 0.2\cdot3 + 0 = 0.8 $$

Error:
$$ \hat{y}_A - y = 0.8 - 1 = -0.2 $$

Pérdida:
$$ L_A = 0.5 \cdot (-0.2)^2 = 0.02 $$

#### Paso 2. Gradientes de Ejemplo A
$$ \frac{\partial L}{\partial \hat{y}} = \hat{y} - y = -0.2 $$

$$\frac{\partial L}{\partial w_1} = (\hat{y}-y) \cdot x_1 = -0.2 \cdot 2 = -0.4 $$

$$ \frac{\partial L}{\partial w_2} = -0.2 \cdot 3 = -0.6 $$

$$ \frac{\partial L}{\partial b} = -0.2 $$

#### Paso 3. Actualización con descenso de gradiente

$$ w_1 \leftarrow w_1 - \eta \cdot \frac{\partial L}{\partial w_1} = 0.1 - 0.1 \cdot (-0.4) = 0.14 $$

$$ w_2 \leftarrow 0.2 - 0.1 \cdot (-0.6) = 0.26 $$

$$ b\leftarrow 0.0 - 0.1 \cdot (-0.2) = 0.02 $$

Recalculando A:
$$ \hat{y'}_A = 0.14\cdot2 + 0.26\cdot3 + 0.02 = 1.08 $$

La predicción se acercó a 1 y la pérdida bajó a 0.0032.

#### Paso 4. Siguiente batch, Ejemplo B

Forward:
$$ \hat{y}_B = 0.14\cdot1 + 0.26\cdot(-1) + 0.02 = -0.10 $$

Error:
$$ \hat{y}_B - y = -0.10 - 0 = -0.10 $$

Gradientes:
$$ \frac{\partial L}{\partial w_1} = -0.10 \cdot 1 = -0.1$$ 

$$ \frac{\partial L}{\partial w_2} = -0.10 \cdot (-1) =   1$$


$$ \frac{\partial L}{\partial b} = -0.10 $$

Actualización:
$$ w_1 = 0.14 - 0.1\cdot(-0.10) = 0.15 $$ 

$$ w_2 = 0.26 - 0.1\cdot(0.10) = 0.25 $$

$$ b = 0.02 - 0.1\cdot(-0.10) = 0.03  $$

Fin de la época 1 . Pesos finales:
$$ w_1 = 0.15, w_2 = 0.25, b = 0.03 $$

---

## Qué pasó

- **Época:** vimos A y luego B una vez cada uno.  
- **Función de pérdida:** A tuvo error -0.2 y B -0.1.  
- **Descenso de gradiente:** tradujo ese error en correcciones proporcionales a las entradas.  

Si repetimos épocas, los pesos siguen moviéndose en la dirección que reduce la pérdida promedio. Con suficientes épocas la neurona aprende la mejor combinación lineal de \(x_1\) y \(x_2\) para aproximar \(y\).

En una red real hay miles de neuronas, activaciones no lineales como ReLU o sigmoide, y *backpropagation* encadena estos mismos gradientes capa por capa. El principio es idéntico: **forward → pérdida → backward → update**.

> En código de python sería

```python
import numpy as np

# Configuración inicial
w1, w2, b = 0.1, 0.2, 0.0
lr = 0.1

# Dataset: (x1, x2, y)
data = [
    (2.0, 3.0, 1.0),   # Ejemplo A
    (1.0, -1.0, 0.0)   # Ejemplo B
]

def forward(w1, w2, b, x1, x2):
    return w1*x1 + w2*x2 + b

def loss(y_hat, y):
    return 0.5 * (y_hat - y)**2

print("Pesos iniciales: w1=%.2f, w2=%.2f, b=%.2f\n" % (w1, w2, b))

for epoch in range(1, 3):
    print(f"=== Época {epoch} ===")
    for i, (x1, x2, y) in enumerate(data, start=1):
        # Forward
        y_hat = forward(w1, w2, b, x1, x2)
        L = loss(y_hat, y)
        
        # Gradientes
        dL_dyhat = y_hat - y
        dL_dw1 = dL_dyhat * x1
        dL_dw2 = dL_dyhat * x2
        dL_db = dL_dyhat
        
        # Actualización
        w1_new = w1 - lr * dL_dw1
        w2_new = w2 - lr * dL_dw2
        b_new = b - lr * dL_db
        
        print(f" Ejemplo {i}: x=({x1},{x2}), y={y}")
        print(f"  Predicción: {y_hat:.4f}, Pérdida: {L:.4f}")
        print(f"  Gradientes: dw1={dL_dw1:.4f}, dw2={dL_dw2:.4f}, db={dL_db:.4f}")
        print(f"  Pesos -> w1: {w1:.4f} -> {w1_new:.4f}, w2: {w2:.4f} -> {w2_new:.4f}, b: {b:.4f} -> {b_new:.4f}\n")
        
        w1, w2, b = w1_new, w2_new, b_new
```

Salida

```python
Pesos iniciales: w1=0.10, w2=0.20, b=0.00

=== Época 1 ===
 Ejemplo 1: x=(2.0,3.0), y=1.0
  Predicción: 0.8000, Pérdida: 0.0200
  Gradientes: dw1=-0.4000, dw2=-0.6000, db=-0.2000
  Pesos -> w1: 0.1000 -> 0.1400, w2: 0.2000 -> 0.2600, b: 0.0000 -> 0.0200

 Ejemplo 2: x=(1.0,-1.0), y=0.0
  Predicción: -0.1000, Pérdida: 0.0050
  Gradientes: dw1=-0.1000, dw2=0.1000, db=-0.1000
  Pesos -> w1: 0.1400 -> 0.1500, w2: 0.2600 -> 0.2500, b: 0.0200 -> 0.0300

Pérdida promedio fin época 1: 0.002825

=== Época 2 ===
 Ejemplo 1: x=(2.0,3.0), y=1.0
  Predicción: 1.0800, Pérdida: 0.0032
  Gradientes: dw1=0.1600, dw2=0.2400, db=0.0800
  Pesos -> w1: 0.1500 -> 0.1340, w2: 0.2500 -> 0.2260, b: 0.0300 -> 0.0220

 Ejemplo 2: x=(1.0,-1.0), y=0.0
  Predicción: -0.0700, Pérdida: 0.0024
  Gradientes: dw1=-0.0700, dw2=0.0700, db=-0.0700
  Pesos -> w1: 0.1340 -> 0.1410, w2: 0.2260 -> 0.2190, b: 0.0220 -> 0.0290

Pérdida promedio fin época 2: 0.000856

Pesos finales: w1=0.1410, w2=0.2190, b=0.0290
```
## Scikit-learn vs TensorFlow: 
> ¿Cuáles son las principales diferencias entre Scikit-learn y TensorFlow como frameworks de Machine Learning? ¿En qué situaciones es más conveniente usar cada uno, y existe algún caso en que sea recomendable utilizarlos juntos en un mismo proyecto?

Scikit-learn y TensorFlow no compiten en la misma liga. Uno es una caja de herramientas clásica para ML tabular, el otro es un ecosistema para aprendizaje profundo y producción a escala.

### Diferencias fundamentales
#### Filosofía y alcance
**Scikit-learn**

* API uniforme y predecible: `fit`, `predict`, `transform`
* Enfocado en algoritmos clásicos: regresión lineal y logística, SVM, árboles, random forest, gradient boosting, k-means, PCA.
* Todo en CPU, con NumPy/SciPy debajo. Curva de aprendizaje corta.

**TensorFlow**

* Framework de grafos computacionales para deep learning. Hoy con Keras como API de alto nivel.
* Diseñado para redes neuronales: CNN, RNN, Transformers, modelos generativos.
* Ejecución en CPU, GPU y TPU, entrenamiento distribuido, serialización con SavedModel, despliegue en móvil con TFLite y en servidor con TensorFlow Serving.
#### Datos y flujo de trabajo
**Scikit-learn**

* Espera datos tabulares en memoria: arrays NumPy o DataFrames pandas.
* Pipeline nativo para preprocesamiento + modelo.
* Validación cruzada, búsqueda de hiperparámetros y métricas integradas.

**TensorFlow**

* Trabaja con tf.data para pipelines de streaming, aumento de datos y lotes grandes que no caben en RAM.
* Necesitas definir arquitectura, función de pérdida, optimizador y bucle de entrenamiento.
* Herramientas de producción: TensorBoard, checkpoints, mixed precision, XLA.
#### Flexibilidad vs simplicidad
Scikit-learn gana en simplicidad para problemas estándar. TensorFlow gana en expresividad para modelos personalizados y escala.

### ¿Cuándo usar cada uno?
**Usa Scikit-learn cuando:**

* Tus datos son tabulares y de tamaño moderado, miles a millones de filas que caben en RAM.
* Necesitas una baseline rápida: regresión logística para churn, random forest para scoring crediticio, gradient boosting para predicción de demanda.
* La interpretabilidad importa: coeficientes, importancia de features, SHAP sobre árboles.
* El ciclo es experimentación en notebook y despliegue ligero con pickle/joblib.

Casos típicos: detección de fraude con features ingenierizadas, segmentación con k-means, reducción de dimensionalidad con PCA, modelos de propensión en marketing.

**Usa TensorFlow cuando:**

* Trabajas con datos no estructurados: imágenes, audio, texto, video, series temporales largas.
* Necesitas deep learning: visión por computadora, NLP con Transformers, recomendadores basados en embeddings, modelos generativos.
* Requieres GPU/TPU y entrenamiento distribuido.
El despliegue es parte del requisito: TFLite en móvil, TF Serving en producción, pipelines con TFX.

Casos típicos: clasificación de imágenes médicas, detección de objetos en línea de producción, traducción automática, forecasting con LSTM/Temporal Fusion Transformer.

### ¿Se usan juntos? Sí, y es común
Son complementarios en proyectos híbridos.

**Patrón 1: Preprocesamiento clásico + deep learning**
Scikit-learn para limpieza tabular, imputación, escalado y codificación. Luego pasas las matrices a TensorFlow para entrenar la red. `ColumnTransformer` y `Pipeline` de sklearn dejan los datos listos para `tf.data`.

**Patrón 2: Features de deep learning + modelo clásico**
Entrenas un encoder con TensorFlow, por ejemplo un CNN que produce embeddings de imágenes o un BERT que vectoriza texto. Congelas el encoder y usas esos embeddings como features para un modelo de Scikit-learn: logistic regression, SVM o gradient boosting. Es útil cuando tienes pocos datos etiquetados para la tarea final pero un buen representador preentrenado.

**Patrón 3: Stacking híbrido**
Un modelo de TensorFlow genera probabilidades para datos no estructurados, un modelo de Scikit-learn genera probabilidades para datos tabulares. Un meta-modelo simple combina ambos.

**Patrón 4: Validación y análisis**
Entrenas en TensorFlow y evalúas con métricas y herramientas de Scikit-learn: `classification_report`, `roc_auc_score`, `confusion_matrix`. La interoperabilidad vía NumPy lo hace trivial.

#### Regla práctica
Si puedes resolverlo bien con Scikit-learn, empieza ahí. Es más rápido de implementar, depurar y explicar. Pasa a TensorFlow cuando la señal está en imágenes, texto libre o secuencias, o cuando necesitas la escala y el ecosistema de despliegue que ofrece.
