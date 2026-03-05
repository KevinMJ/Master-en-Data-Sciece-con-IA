# Challenge: Introducción a las Funciones
> El objetivo de este challenge es aplicar lo visto en sesión y lecturas sobre funciones en Python para mejorar un CRUD de agenda. Se espera modularizar, definir parámetros/retornos correctamente y aumentar la legibilidad eliminando repetición.

## Competencias a demostrar
* **Modularización de código:** separar el programa en bloques lógicos reutilizables para cada operación de la agenda.
* **Gestión de parámetros y retornos:** funciones que reciben información específica y devuelven resultados/estados (éxito, error, datos procesados).
* **Abstracción y legibilidad:** reducir código duplicado y facilitar mantenimiento futuro del CRUD.

## Cuadernos Colab (material oficial)
* [Cuaderno del challenge (haz una copia para trabajar)](https://colab.research.google.com/drive/1EhhsSw9P7aiPq7U1ZLsk6ZAJ0GyKLvcX?usp=sharing)


* [Versión solucionada (para completar la parte vista en clase si te falta)](https://colab.research.google.com/drive/19WfNFwr8eu0YK3MGQh4RBczbKfbcqvgI?usp=sharing)

## Instrucciones de trabajo (paso a paso)
1. Abre el cuaderno del challenge y crea una copia: File → Save a copy in Drive.
2. Completa primero la sección de código vista en clase. Si no la tienes completa, apóyate en la versión solucionada (copia/pega con criterio y entiende qué hace cada parte).
3. Ve a la sección exclusiva del challenge dentro del cuaderno y sigue las instrucciones detalladas.
4. Refactoriza el CRUD para que cada operación viva en una función (y evita duplicar validaciones/prints).
5. Define firmas claras: qué recibe cada función (parámetros) y qué devuelve (retornos) para que el menú principal solo coordine.
6. Prueba tu programa con varios escenarios: altas, consultas, ediciones, borrados, búsquedas, entradas inválidas, y flujo de salida.

## Sugerencia de arquitectura (guía rápida)
* Funciones “core” del CRUD (ejemplos): create_contact, read_contact, update_contact, delete_contact, list_contacts, search_contacts.
* Helpers reutilizables: print_menu, get_non_empty_input, validate_phone, find_index_by_id (o por nombre), confirm_action.
* Parámetros recomendados: pasa la estructura de datos (ej. agenda), el identificador (id/nombre), y los nuevos datos. Evita depender de variables globales cuando puedas.
* Retornos recomendados: usa bool (éxito/fracaso), dict/list (datos), o tuplas (resultado + mensaje). Deja el print en un solo lugar si tu diseño lo permite.

## Checklist técnico (auto-evaluación)
- [X] Cada opción del menú llama a una función (no hay bloques gigantes repetidos).
- [X] Mis funciones reciben solo lo necesario (parámetros claros, sin “magia”).
- [X] Mis funciones regresan algo útil (datos/estado) en vez de depender solo de prints.
- [X] Evité repetir validaciones (tengo helpers para input/validación).
- [X] El flujo principal (main) se lee como “orquestación”, no como lógica detallada.
- [X] Probé casos de error (contacto inexistente, inputs vacíos, formatos inválidos).

## Entregable (link público)
> La entrega es la liga del cuaderno (tu copia), con permisos de lector. Debe apuntar al archivo .ipynb (no a la carpeta de Drive).

  * En tu copia del Colab: Share → cambia permisos a Anyone with the link → Viewer (o el esquema que te pidan en campus, pero debe poder abrirse).
  * Copia el link del Colab y pégalo como entrega en el campus.
  * Opcional (recomendado): exporta el notebook y súbelo a GitHub (como respaldo) y conserva el link público del repositorio/archivo: File → Download → .ipynb y luego súbelo a tu repo.

## Evidencia sugerida (para tu organización)
* Link público a tu copia de Colab (principal).
* Link a tu GitHub con el .ipynb (respaldo).
* Si tu campus pide comentarios: agrega un bloque final en el notebook explicando qué funciones creaste y qué retorna cada una.

## Cierre
Sigue practicando: refactorizar código hacia funciones pequeñas, con parámetros claros y retornos útiles, es una de las habilidades que más rápido mejora tu calidad como dev. Haz iteraciones: primero que funcione, luego que se entienda, luego que sea fácil de mantener.