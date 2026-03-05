# Apuntes

> Prompt : 
"Actúa como tutor de Python para Ciencia de Datos. Explícame detalladamente en qué se diferencian los distintos tipos de funciones en Python (por ejemplo: built-in, definidas por el usuario con def, lambda/anónimas, métodos (instancia/clase/estático), generadoras con yield y funciones async). Para cada tipo: (1) definición breve, (2) un ejemplo corto en Python, y (3) un caso real de cuándo usarla."

## Funciones Built-in
* Funciones preinstaladas de python
  * sum(), len(), round()

## Funciones definidas por el usuario
* Funciones que tu mismo creas usando la palabra clave `def`

`def limpiar_texto(texto):
    texto_limpio = texto.lower().strip()
    return texto_limpio`

## Funciones Lambda (Anónimas)
* Pequeñas funciones de una sola línea, sin nombre.
`multiplicar_por_dos = lambda x: x * 2`