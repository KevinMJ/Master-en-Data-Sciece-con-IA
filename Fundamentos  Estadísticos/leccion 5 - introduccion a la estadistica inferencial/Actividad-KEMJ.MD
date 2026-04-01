# Estadística: 
¿En qué se diferencia la estadística descriptiva con la estadística inferencial?, ¿Qué papel juega el aprendizaje estadístico dentro de la inferencia?

La diferencia está en **qué hacen con los datos** y **hasta dónde llegan las conclusiones**.

## Estadística descriptiva
* **Objetivo**: resumir, organizar y presentar los datos que ya tienes (toda la población o una muestra concreta).
* **Herramientas típicas**: medidas de tendencia central (media, mediana, moda), dispersión (rango, desviación estándar, varianza) y gráficos (histogramas, diagramas de caja).
* **Alcance**: describe “lo que pasó” en ese conjunto observado, sin intentar generalizar más allá. Por ejemplo, “el promedio de edad en este grupo es 35 años”. 
## Estadística inferencial
* **Objetivo**: usar una muestra para hacer afirmaciones sobre una población más grande que no se observa por completo.
* **Herramientas típicas**: estimación puntual e intervalar (intervalos de confianza), pruebas de hipótesis, modelos de regresión y análisis de varianza. Se apoya en la probabilidad para cuantificar la incertidumbre (error de muestreo).
* **Alcance**: permite generalizar y predecir, por ejemplo, “con 95 % de confianza, la media de edad de la población está entre 33 y 37 años”. La definición formal también la describe como la rama que “elabora conclusiones, generalizaciones o predicciones acerca de una población basándose en el análisis de una muestra representativa”. 
## ¿Dónde entra el aprendizaje estadístico?
El **aprendizaje estadístico** (statistical learning) es el conjunto de métodos —muchos provenientes del machine learning— que construyen modelos a partir de datos de muestra para inferir patrones y hacer predicciones sobre nuevos datos. Dentro de la inferencia clásica cumple dos papeles clave:

1. **Modelado flexible de la relación entre variables **(por ejemplo, regresión, árboles, redes). Estos modelos estiman parámetros y permiten cuantificar la incertidumbre (intervalos, pruebas) sobre la población subyacente.
2. **Predicción y generalización**: el foco pasa de solo probar hipótesis a evaluar qué tan bien el modelo aprendido se comporta fuera de la muestra (error de predicción, validación cruzada), lo que es, en esencia, una forma de inferencia predictiva sobre la población de casos futuros.

En resumen: la descriptiva resume lo observado; la inferencial usa la muestra para hablar de la población; y el aprendizaje estadístico es la caja de herramientas moderna que, dentro de la inferencial, construye y evalúa modelos capaces de inferir y predecir más allá de los datos que tenemos.

---

# Significancia: 
¿Por qué el valor de 0.05 (5%) es el estándar de oro para el p-value en la mayoría de las investigaciones? ¿Qué pasaría si usáramos un estándar más relajado, como 0.10 (10%)?

No es una ley de la naturaleza: el 5 % es una convención que se volvió “estándar de oro” por historia y comodidad práctica, no porque sea óptimo en todos los casos.

## ¿De dónde sale el 0,05?
Ronald A. Fisher popularizó el umbral en los años 1920. En Statistical Methods for Research Workers (1925) escribió que “es conveniente tomar como límite el punto en que P = 0,05, o 1 en 20… desviaciones que superen dos veces la desviación estándar se consideran formalmente significativas”. Es decir, 5 % equivale más o menos a 2 σ en una normal y resultaba un punto práctico para decidir si un resultado “merecía” atención. Con el tiempo esa conveniencia se institucionalizó y se repitió como regla en revistas y manuales. 

En la práctica también se usan otros niveles (1 % para “muy significativo”, 0,1 % para “extremadamente significativo”) y no hay obligación de fijarlo a priori; Fisher mismo señalaba que los niveles son flexibles. 

## ¿Qué pasa si lo relajamos a 0,10?
Cambiar α de 0,05 a 0,10 aumenta la probabilidad de cometer un error tipo I (rechazar la hipótesis nula cuando en realidad es cierta). Es el clásico trade-off entre los dos tipos de error:

* **Más falsos positivos**: con α=0,10 esperas que, en promedio, 1 de cada 10 estudios sin efecto real aparezca como “significativo” solo por azar, el doble que con 0,05 (1 de cada 20). Esto infla la tasa de descubrimientos irreproducibles.
* **Más potencia (menos falsos negativos)**: al ser menos estricto, también reduces el error tipo II (β); es más fácil detectar efectos pequeños o con muestras limitadas.
* **Consecuencia práctica**: más hallazgos “significativos” publicados, pero menor confiabilidad; en contextos donde equivocarse al afirmar un efecto es costoso (medicina, políticas), ese aumento de falsos positivos es problemático. Por eso muchas propuestas recientes van en la dirección contraria (bajar el umbral a 0,005), no a relajarlo.
En resumen: 0,05 quedó por conveniencia histórica y porque “1 en 20” suena razonable; subir el umbral a 0,10 hace la prueba menos conservadora: ganas potencia pero pagas con más resultados que parecen reales y no lo son. Por eso la elección de α siempre debería justificarse según el costo relativo de equivocarse en cada dirección, el tamaño muestral y el contexto del estudio.

# Regresión:
¿Cómo se relaciona una correlación de bivariable con una regresión lineal simple?, ¿Cómo actúa el algoritmo del descenso de gradiente como optimizador para una regresión?

**En corto:** la correlación describe la fuerza y dirección de la asociación lineal entre dos variables, mientras que la regresión lineal simple convierte esa asociación en una ecuación predictiva \(\hat{y}=b_0+b_1 x\). El descenso de gradiente es simplemente el método iterativo que ajusta \(b_0\) y \(b_1\) para que la recta minimice el error.

### 1. De la correlación a la recta de regresión
- El **coeficiente de correlación de Pearson** \(r\) mide la intensidad y el signo de la relación lineal entre \(x\) e \(y\) (varía entre \(-1\) y \(+1\)) y se calcula con  
  $$
  r=\frac{\sum (x_i-\bar{x})(y_i-\bar{y})}{\sqrt{\sum (x_i-\bar{x})^2\; \sum (y_i-\bar{y})^2}}.
  $$
- La **regresión lineal simple** usa esos mismos datos para estimar la recta de mejor ajuste. La pendiente y la ordenada están ligadas a \(r\) por  
  $$
  b_1 = r\,\frac{s_y}{s_x},\qquad b_0 = \bar{y} - b_1\bar{x},
  $$  
  donde \(s_x, s_y\) son las desviaciones típicas. Es decir, la pendiente es la correlación “reescalada” por la dispersión de cada variable, y el intercepto hace que la recta pase por \((\bar{x},\bar{y})\). Además, \(r^2\) coincide con el coeficiente de determinación \(R^2\) de la regresión (la proporción de varianza explicada).

### 2. ¿Qué hace el descenso de gradiente en la regresión?
El objetivo es minimizar la función de pérdida, típicamente el **error cuadrático medio (MSE)**:
$$
J(w,b)=\frac{1}{m}\sum_{i=1}^{m}\big(f_{w,b}(x^{(i)})-y^{(i)}\big)^2,\quad \text{con } f_{w,b}(x)=wx+b.
$$

El descenso de gradiente:
1. **Inicializa** \(w\) y \(b\) (por ejemplo en \(0\)) y calcula la pérdida.
2. **Calcula las derivadas parciales** (el gradiente) de \(J\) respecto a los parámetros:  
   $$
   \frac{\partial J}{\partial w}= \frac{1}{m}\sum 2x^{(i)}\big(f_{w,b}(x^{(i)})-y^{(i)}\big),\qquad
   \frac{\partial J}{\partial b}= \frac{1}{m}\sum 2\,\big(f_{w,b}(x^{(i)})-y^{(i)}\big).
   $$
3. **Actualiza** los parámetros moviéndose en la dirección opuesta al gradiente:  
   $$
   w := w - \alpha\,\frac{\partial J}{\partial w},\quad b := b - \alpha\,\frac{\partial J}{\partial b},
   $$  
   donde \(\alpha\) es la tasa de aprendizaje.
4. Repite hasta que la pérdida deje de disminuir (convergencia).

En palabras de Google Developers: “el descenso de gradientes es una técnica… que encuentra iterativamente los pesos y sesgos que producen el modelo con la pérdida más baja”; GeeksforGeeks lo resume como “un algoritmo de optimización usado en regresión lineal para minimizar el error en las predicciones”.

Así, la correlación te dice cuán alineados están los puntos; la regresión traduce eso en una recta predictiva; y el descenso de gradiente es el procedimiento que, paso a paso, ajusta la pendiente y el intercepto de esa recta para que el error cuadrático sea mínimo.
---
# Reflexión
Lo que más me sorprende es la fluidez con la que puedo pasar de un tema técnico a una explicación en lenguaje llano. No es solo que “sepa” las definiciones; es que puedo enlazar conceptos.