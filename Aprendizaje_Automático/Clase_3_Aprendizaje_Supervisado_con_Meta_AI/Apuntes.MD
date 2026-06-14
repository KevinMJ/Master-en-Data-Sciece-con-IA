# Apuntes

## KNN y la Distancia: 
> ¿Cómo funciona matemáticamente el algoritmo K-Nearest Neighbors? Explica el papel de la distancia euclidiana, cómo impacta la elección del valor de K en el rendimiento del modelo (sobreajuste vs. subajuste), y por qué la normalización de los datos es un paso obligatorio antes de aplicar KNN.

KNN no entrena pesos, no ajusta curvas. Guarda todos los puntos de entrenamiento y, cuando llega uno nuevo, pregunta: ¿quiénes son mis vecinos más cercanos?

### Cómo funciona matemáticamente
Tienes un conjunto de entrenamiento $\{(x^{(i)}, y^{(i)})\}$ donde cada $\ x^{(i)}$ vive en $\mathbb{R}^n.$

Para un punto nuevo z:

1. Calculas la distancia entre $z$ y cada $x^{(i)}$.
2. Ordenas y te quedas con los K más cercanos: N_K(z).
3. Predices:
   - clasificación: $\hat{y} = \text{moda}{ y^{(i)} : x^{(i)} \in N_K(z) }$
   - regresión: $\hat{y} = \frac{1}{K}\sum_{x^{(i)} \in N_K(z)} y^{(i)}$

No hay gradiente, no hay función explícita. La "función" es la geometría local de tus datos.

### El papel de la distancia euclidiana
La euclidiana es la regla de Pitágoras en n dimensiones:

$$ d(p,q) = \sqrt{ \sum_{j=1}^{n} (p_j - q_j)^2 }$$

Mide cuánto tienes que "viajar" en línea recta para ir de p a q. Por eso KNN asume que puntos cercanos en ese espacio deberían compartir etiqueta.

No es la única opción. Manhattan $\sum |p_j-q_j|$ funciona mejor con datos dispersos, y Minkowski generaliza ambas. Pero euclidiana domina porque penaliza diferencias grandes al cuadrado, lo que da fronteras de decisión suaves y circulares en 2D.

El problema: esa suma trata a cada dimensión igual. Si una variable está en miles y otra en décimas, la primera decide casi toda la distancia.

### Elegir K: el control de sesgo y varianza
K es tu perilla de suavizado, no un hiperparámetro decorativo.

- **K pequeño, por ejemplo 1 o 3:** el modelo memoriza. La frontera es dentada, sigue cada punto ruidoso. Alta varianza, bajo sesgo. Funciona en entrenamiento, falla con datos nuevos. Eso es sobreajuste.
- **K grande, por ejemplo 50 o 100:** promedia vecindarios enormes. La frontera se vuelve casi plana. Bajo varianza, alto sesgo. Ignora patrones locales. Eso es subajuste.
- **K intermedio:** equilibra. Una regla práctica es empezar en K \approx \sqrt{N} y luego validar con cross-validation, pero no hay sustituto para probar 3, 5, 7, 11... y mirar la curva de error.

Visualízalo así: K=1 dibuja islas alrededor de cada punto. K=N predice siempre la clase mayoritaria global. Entre esos extremos está tu modelo útil.

### Por qué normalizar no es opcional
KNN depende 100% de distancias, y las distancias dependen 100% de la escala.

Imagina predecir si alguien compra con dos variables:

edad: 18 a 80
ingreso mensual: 20,000 a 200,000 MXN
Sin normalizar:

$$d = \sqrt{(edad_1-edad_2)^2 + (ingreso_1-ingreso_2)^2}$$
La diferencia de ingreso puede ser 180,000, la de edad máximo 62. El término de ingreso domina por factor de miles. Para KNN, la edad casi no existe.

Normalizar reescala todo a la misma cancha. Dos formas comunes:

- Min-max: $x' = \frac{x - \min}{\max - \min} →$ rango[0][1]
- Estandarización: $x' = \frac{x - \mu}{\sigma} →$ media 0, desviación 1

Después de eso, cada dimensión aporta proporcionalmente a su variabilidad real, no a sus unidades. Sin este paso, KNN no es KNN, es "vecino más cercano en la variable más grande".

Un detalle extra: en alta dimensión, todas las distancias se parecen. Por eso KNN sufre la maldición de la dimensionalidad y la normalización ayuda, pero no la cura. Reducir variables o usar distancias ponderadas suele ser necesario después.

En resumen: guarda puntos, mide con euclidiana, vota con K vecinos. K pequeño memoriza, K grande generaliza demasiado. Y si no normalizas, tu modelo mide pesos en kilos contra distancias en kilómetros, y decide mal por construcción.

## Métricas de Clasificación: 
> ¿Qué miden exactamente la precisión, el recall (sensibilidad) y el F1-score en un problema de clasificación? ¿Por qué en un diagnóstico médico como la detección de tumores malignos es preferible optimizar el recall sobre la precisión, incluso si esto aumenta los falsos positivos?

Vamos directo a la matriz de confusión, porque sin ella precisión y recall suenan igual.

Imagina un clasificador binario de tumores:

|                  | Predicho Maligno | Predice Benigno |
|------------------|-------------------|-------------------|
| **Es maligno** | TP (verdarero Positivo) | FN (falso negativo)|
| **Es benigno** | FP (falso positivo) | TN (verdadero negativo) |

### ¿Qué mide cada métrica?
**Precisión**: de todo lo que marcaste como positivo, ¿cuánto acertaste?

$$ \text{Precision} = \frac{TP}{TP + FP} $$

**Mide pureza**. Precisión alta = cuando dices "tiene cáncer", casi siempre tienes razón. No te importa cuántos cánceres se te escaparon, solo que no alarmes en falso.

**Recall o sensibilidad:** de todos los casos reales positivos, ¿cuántos encontraste?

$$ \text{Recall} = \frac{TP}{TP + FN} $$ 

**Mide cobertura**. Recall alto = si hay un tumor maligno, casi seguro lo detectas. No te importa si también marcaste muchos benignos como sospechosos.

Son tensiones opuestas. Subir el umbral de decisión aumenta precisión y baja recall. Bajarlo hace lo contrario.

**F1-score:** la media armónica que castiga desbalances.

$$ F1 = 2 \cdot \frac{\text{Precision} \times \text{Recall}}{\text{Precision} + \text{Recall}} = \frac{2TP}{2TP + FP + FN} $$
No es un promedio simple. Si una de las dos es baja, F1 se hunde. Sirve cuando quieres equilibrio y no tienes una razón de negocio para priorizar una sobre otra.

### ¿Por qué en tumores priorizamos recall?
Porque el costo de los errores no es simétrico.

- FN (falso negativo): le dices "estás sano" a alguien con cáncer. Consecuencia: tratamiento tardío, metástasis, muerte evitable. Costo humano altísimo.
- FP (falso positivo): le dices "parece maligno" a alguien sano. Consecuencia: ansiedad, biopsia, resonancia extra. Costo molesto, pero reversible.

En medicina, preferimos pecar de paranoicos. Un modelo con recall 0.98 y precisión 0.70 detecta 98 de cada 100 tumores reales, aunque genere 30 alarmas falsas por cada 100 positivos. Esas 30 personas harán pruebas adicionales y saldrán limpias.

Un modelo con precisión 0.98 y recall 0.70 haría lo inverso: casi nunca da falsas alarmas, pero deja ir 30 tumores reales sin detectar. En oncología eso es inaceptable.

Por eso se optimiza recall, se mueve el umbral hacia abajo, se aceptan más FP, y luego se usa una segunda etapa clínica para filtrar. El F1 no ayuda aquí porque trata ambos errores como iguales, y en este contexto no lo son.

Regla práctica: si el FN mata y el FP solo cuesta dinero o tiempo, optimiza recall. Si el FP es peligroso, por ejemplo spam filter que bloquea correos críticos, entonces prioriza precisión.

## Clasificación Binaria vs. Multiclase:
>¿Cuáles son los principales desafíos técnicos que distinguen un problema de clasificación binaria de uno multiclase? ¿Cómo adaptan los algoritmos como KNN su funcionamiento para manejar múltiples clases, y qué estrategias existen para mejorar el rendimiento en clases con pocos ejemplos?

Vamos a desarmarlo en tres partes, porque cada una esconde un problema distinto.

### 1. Binaria vs multiclase: lo que realmente cambia
La binaria es cómoda: un solo límite de decisión, dos probabilidades que suman 1, y métricas como accuracy o ROC-AUC que funcionan directo.

Cuando pasas a 3, 10 o 100 clases, la geometría se complica:

- **Límites de decisión múltiples.** Ya no separas "sí/no", sino que navegas un paisaje con muchas fronteras que se tocan. La clasificación multiclase "exige un enfoque matizado para discernir las diferencias sutiles entre las distintas clases" y "la necesidad de algoritmos especializados que puedan manejar la creciente complejidad de los límites". 
- **Estrategias de descomposición.** Muchos algoritmos originalmente binarios se adaptan así:
  - **Uno contra el resto (OvR):** entrenas un clasificador por clase, esa clase es positiva y todas las demás son negativas. Simple, pero con 50 clases tienes 50 modelos. 
  - **Uno contra uno (OvO):** un modelo por cada par, es decir N×(N-1)/2 clasificadores. Muy costoso, útil cuando el algoritmo base no escala con muchas muestras. 
  - **ECOC:** codificas cada clase como palabra binaria y entrenas clasificadores por bit, da tolerancia a errores. 

- **Salida probabilística.** Redes neuronales usan softmax en la capa final para entregar una distribución entre clases, algo natural para multiclase. 
- **Métricas engañosas.** En binario, accuracy, precisión y recall bastan. En multiclase "pueden no traducirse bien... donde el equilibrio entre clases y el costo de una clasificación errónea puede ser más complejo". Por eso pasamos a matriz de confusión, F1 macro, balanced accuracy. 
- **Pérdida de información si fuerzas binario.** Forzar un problema complejo a dos categorías "corre el riesgo de simplificar demasiado la situación" y provoca "pérdida importante de información". Ejemplo clásico: diagnosticar solo "enfermedad/no enfermedad" ignora subtipos con tratamientos distintos. 
### 2. ¿Y KNN? Es multiclase por naturaleza
KNN no necesita OvR ni OvO. Funciona igual con 2 o con 20 clases porque no aprende un hiperplano, solo mira vecinos.

- **Voto por pluralidad.** Para clasificación, "la etiqueta de clase se asigna en función de la mayoría de votos, es decir, se utiliza la etiqueta que se representa con más frecuencia en torno a un punto". 
- El matiz importante: con más de dos clases no necesitas 50% + 1. "Cuando tiene varias clases, por ejemplo cuatro categorías, no necesita necesariamente el 50% de los votos... puede asignar una etiqueta de clase con un voto superior al 25%". 
- **k define el vecindario.** "El valor k define cuántos vecinos se comprobarán... si k=1, la instancia se asignará a la misma clase que su único vecino". En multiclase suele recomendarse k impar para evitar empates, y validación cruzada para elegirlo. 
- Desafíos que se agravan en multiclase:
  - Maldición de la dimensionalidad: "no funciona bien con entradas de datos de alta dimensión... después de alcanzar el número óptimo de características, las características adicionales aumentan errores". 
  - Con muchas clases raras, un k grande "suaviza" y ahoga la minoría; un k pequeño sobreajusta al ruido.

Prácticamente: KNN multiclase = calcula distancia (euclidiana, Manhattan, Minkowski), toma los k más cercanos, cuenta votos por clase, devuelve la más frecuente. Nada más.

### 3. Clases con pocos ejemplos: cómo levantarlas
El desbalance es el dolor real. Si una clase representa menos del 10% suele considerarse altamente desbalanceado, y el modelo aprende a ignorarla. 

Estrategias que funcionan, de lo simple a lo avanzado:

1. **Resampling**

   - **Oversampling:** replicas o sintetizas ejemplos de la minoría. SMOTE es la referencia porque "permite generar ejemplos sintéticos en lugar de replicar los mismos datos". 
   - **Undersampling:** reduces la mayoritaria para equilibrar proporción. Útil si tienes millones de ejemplos, peligroso si pierdes información. 
  
2. **Pesos en la función de pérdida**
No tocas los datos, tocas el costo. "Modificar la función de pérdida asignando pesos mayores a la clase minoritaria... incentiva al modelo a prestar más atención". En scikit-learn: class_weight='balanced'; en XGBoost o Random Forest también está soportado. 

3. **Generación sintética avanzada**
Más allá de SMOTE, "las Redes Generativas Adversariales (GANs) son una opción avanzada para generar datos realistas". Útil en imágenes o series donde SMOTE lineal falla. 

4. **Cambia las métricas, no solo el modelo**
Deja de mirar accuracy. Usa:

- F1-score, "promedio entre precisión y sensibilidad" 
- AUC-ROC
- Matriz de confusión para ver dónde falla cada clase 
5. **Trucos específicos para KNN con clases raras**

- Usa distancia ponderada: vecinos más cercanos pesan más que los lejanos (weights='distance').
- Reduce dimensionalidad antes (PCA, embeddings) para mitigar la maldición.
- Combina con oversampling solo en el set de entrenamiento, nunca en test, para no inflar métricas.
- Considera ensembles: un KNN solo sufre con minorías; un bosque o boosting con pesos de clase suele aguantar mejor.
---
Resumen rápido: binario es un duelo, multiclase es un torneo, y eso cambia límites, métricas y costo computacional. KNN lo resuelve votando entre vecinos sin reentrenar nada, pero sufre si hay muchas dimensiones o clases muy pequeñas. Para esas clases pequeñas, la clave no es "más datos a ciegas", sino equilibrar con SMOTE, pesar el error y medir con F1 en lugar de accuracy.

## Reflexión
 Lo que más me sorprendió de la herramienta es que ahora sea tan precisa al momento de hablar al español y que no demora más de unos segundos para encontrar la información buscada, procesarla y entregarla en un formato fácil de entender o al cuál puedo seguir preguntándo y entrenando porque ahora se ha adaptado a la manera en la que le solicito lás respuestas.