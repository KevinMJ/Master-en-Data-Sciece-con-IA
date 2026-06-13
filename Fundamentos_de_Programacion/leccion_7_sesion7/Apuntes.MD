# Apuntes Estándares de Estilo Y Filosofía en Python

## PEP 20 "Zen de Python" [Tim Peters]

* Son una guía cultural que busca promover la claridad, la simplicidad, la legibilidad y la elegancia en la programación
* Influencia de los Aforismos en la programación.
  * Legibilidad y Claridad
  * Simplicidad y Concisión
  * Gestión de Errores
  * Unicidad y Coherencia
  * Practicidad

### 19 Aforismos

1. Bello es mejor que feo.
2. Explícito es mejor que implícito.
3. Simple es mejor que complejo.
4. Complejo es mejor que complicado.
5. Plano es mejor que anidado.
6. Disperso es mejor que denso.
7. La legibilidad cuenta.
8. Los casos especiales no son tan especiales como para romper las reglas.
9. Aunque lo práctico gana a lo puro.
10. Los errores nunca deberían dejarse pasar silenciosamente.
11. A menos que se silencien explícitamente.
12. Frente a la ambigüedad, rechaza la tentación de adivinar.
13. Debería haber una, y preferiblemente solo una, manera obvia de hacerlo.
14. Aunque esa manera no sea obvia al principio, a menos que seas holandés.
15. Ahora es mejor que nunca.
16. Aunque nunca es a menudo mejor que justo ahora.
17. Si la implementación es difícil de explicar, es una mala idea.
18. Si la implementación es fácil de explicar, puede que sea una buena idea.
19. Los espacios de nombres son una gran idea, ¡hagamos más de esos!

## PEP 8

* Mejora la legibilidad: Un código consistente es más fácil de leer, entender y depurar para todos los miembros del equipo, incluso si no lo escribieron ellos mismos. 
* Facilita la colaboración: Reduce malentendidos y conflictos de estilo entre desarrolladores, haciendo que el trabajo en equipo sea más fluido. 
* Aumenta la mantenibilidad: El código bien estructurado y con un estilo uniforme es más sencillo de mantener y modificar en el futuro. 
* Profesionalismo: Seguir PEP 8 demuestra un compromiso con la escritura de código de alta calidad y profesional. 

### Aquí te dejo algunas de sus reglas más importantes:

* **Sangría**: Utiliza siempre 4 espacios por nivel de sangría. Nunca uses tabulaciones. 
* **Longitud de línea:** Limita todas las líneas a un máximo de 79 caracteres para el código y 72 para docstrings y comentarios. Esto mejora la legibilidad en diferentes pantallas y editores. 
* **Nomenclatura:**
  * **Variables y funciones:** Usa snake_case (minúsculas con guiones bajos). 
  * **Clases:** Usa CamelCase (cada palabra empieza con mayúscula). 
  * **Constantes:** Usa UPPER_SNAKE_CASE (todo mayúsculas con guiones bajos). 
* **Espacios en blanco:**
  * Evita espacios en blanco innecesarios, como dentro de paréntesis o antes de comas. 
  * Usa un solo espacio alrededor de operadores binarios (como =, +, ==). 
  * Agrega dos líneas en blanco para separar definiciones de clases y funciones de nivel superior. 
* **Importaciones:** Agrupa las importaciones en tres categorías, separadas por una línea en blanco:
  1. Módulos de la biblioteca estándar.
  2. Módulos de terceros.
  3. Módulos locales/de tu propio proyecto. 
* Comentarios y Docstrings:
  * Usa comentarios para explicar "por qué" el código hace algo, no solo "qué" hace. 
  * Usa docstrings (cadenas de documentación con triples comillas) para describir módulos, clases y funciones públicas. 
  
> Los operadores ternarios, las funciones lambda y las comprensiones de listas/diccionarios son herramientas súper poderosas que, cuando se usan bien, abrazan totalmente el espíritu de Python, que se enfoca en la legibilidad y la simplicidad.