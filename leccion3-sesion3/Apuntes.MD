# Apuntes
## Fundamentos de Control de Flujo
¿Podrías explicar detalladamente en qué se diferencian las estructuras de flujo Secuencial, de Selección e Iterativa? Proporciona un ejemplo cotidiano para cada una.

* Secuencial -> Las acciones se ejecutan una después de la otra, en un **orden estricto**, sin saltarse ningún paso.
* Selección -> Bifurcación de un camino. La acción que realizas depende de la respuesta a la pregunta
* Iterativa -> Esta estructura permite repetir una o más acciones varias veces, como en un bucle, la repetición se realiza mientras una condidión específica se siga cumpliendo.

## Estrategia de Bucles
¿Bajo qué criterios técnicos o escenarios específicos debería elegir un ciclo for sobre un ciclo while y viceversa? ¿Existe alguna diferencia de rendimiento o buenas prácticas entre ambos?

* Se usa el **Ciclo For** cuando se conoce de antemano cuántas veces necesitas que se repita un bloque de código y el **ciclo while** cundo el bucle deba continuar hasta que una condición específica se vuelva falsa y no sabes de antemano cuántas veces eso ocurrirá.


## Optimización y Pythonic Way
¿En qué situaciones es preferible utilizar un operador ternario o comprensión de listas / diccionarios en lugar de una estructura tradicional de if/else o un ciclo for? ¿Cuándo afecta esto a la legibilidad del código?

**Operador Ternario**

`valor_si_verdadero if condicion else valor_si_falso`

* Es preferible usar el operador ternario para asignaciones simples y directas, donde solo hay dos posibles resultados.

**Comprensión de Listas/ Diccionarios**

* Son herramientas para crear nuevas listas o diccionarios a partir de un iterable existente. Combinan un ciclo for y, opcionalmente, un if en una sola línea.
* Su objetivo es la transformación  el filtrado.
* La función map() y filter() son alternativas más directas a las comprensiones de listas, provienen del paradigma de la programación funcional.
* 

## Guía de Buenas Prácticas
* Usa for cuando el objetivo es iterar sobre un rango o una colección.
* No anidar operadores ternarios
* Si hay más de un camino, evitar el uso de operadores ternarios.
* Si la condicion es compleja o se afecta la legibilidad, cambiar el uso de los operadores ternarios.