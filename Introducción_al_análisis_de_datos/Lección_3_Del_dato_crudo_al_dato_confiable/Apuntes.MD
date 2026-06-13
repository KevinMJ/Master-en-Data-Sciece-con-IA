# Apuntes
## Pregunta
**Estrategias de Imputación:** ¿Cuáles son las principales estrategias para tratar valores nulos en un dataset y cómo decides cuál aplicar? Específicamente, ¿cuándo es correcto imputar con una categoría centinela como "Unknown", cuándo eliminar el registro y cuándo conservarlo con un valor derivado?

## Respuesta de MetaAI

Estrategias de imputación no es solo "rellenar huecos". Es decidir qué historia quieres que tu modelo aprenda del hecho de que algo falte.

Hay tres mecanismos que mandan todo:

* **MCAR (Missing Completely At Random):** falta por azar puro. No hay patrón.
* **MAR (Missing At Random):** falta porque otra variable lo explica. Ej: ingresos faltan más en jóvenes porque no quieren responder.
* **MNAR (Missing Not At Random):** el valor faltante es la información. Ej: campo "alergias" vacío porque el paciente no fue al médico.
Tu decisión depende de ese mecanismo, no del porcentaje.

### Las 5 estrategias principales
1. **Eliminación**
   * Listwise: borras toda la fila.
   * Pairwise: usas la fila solo donde la variable exista.
  
2. **Imputación estadística simple**

   * Categórica: moda.
   * Numérica: media, mediana.
3. **Imputación con centinela**
   
   * Rellenas con un valor que no existe en la distribución: "Unknown", "No_aplica", -999, "Missing".
4. **Imputación derivada o modelada**

   * Usas otras columnas para predecir el valor: KNN, regresión, árboles, MICE, incluso embeddings en 2026.
5. **Conservar el nulo + indicador**

   * Imputas con media/mediana pero añades una columna binaria var_is_missing=1. El modelo ve ambos: el valor probable y el hecho de que faltaba.

### ¿Cómo decides? Hazte 4 preguntas
1. **¿Por qué falta?** Si es MNAR, el nulo es señal. No lo borres.
2. **¿Cuánto falta?** >30-40% en una columna crítica suele ser más ruido que señal, incluso con buena imputación.
3. **¿Qué tipo de variable y modelo?** Árboles (XGBoost, LightGBM) toleran "Unknown". Modelos lineales y redes odian constantes arbitrarias.
4. **¿La variable es el target o un predictor clave?** Nunca imputes el target para entrenar.

#### Cuándo usar "Unknown" o centinela
Es correcto cuando el nulo tiene significado propio, no cuando es un error.

* Variable categórica y el hecho de no responder es informativo: ocupación = "Unknown" en un formulario donde saltarse la pregunta implica desempleo informal o privacidad. Es clásico MNAR.
* "No aplica" real: segunda_mascota = "No_aplica" porque el usuario no tiene mascotas. No es missing, es una categoría.
* Modelos basados en árboles: aprenden splits como if ciudad == "Unknown". No distorsionas la distribución.
* Alta cardinalidad donde la moda sesgaría: imputar "CDMX" a 15% de nulos crearía una ciudad fantasma.
 
No lo uses en numéricas para regresión lineal: -999 se vuelve un outlier que arrastra coeficientes.

#### Cuándo eliminar el registro
Borrar duele menos que mentir con datos.

* MCAR confirmado y <5% de filas afectadas. Pierdes poco poder estadístico.
* Falta el target o un identificador clave sin el cual la fila no tiene sentido analítico.
* MNAR extremo donde imputar inventaría: ej, sensor apagado durante 3 horas, no puedes adivinar temperatura.
* Dataset enorme y el costo de imputación compleja supera la ganancia. En producción, a veces es más honesto descartar.

Ojo: eliminar en MAR crea sesgo. Si los ingresos faltan solo en jóvenes, borrarlos te deja con una muestra más rica y mayor.

#### Cuándo conservar con valor derivado
Es la opción por defecto en 2026 para tabulares serios.

* MAR claro: puedes predecir razonablemente con otras variables. edad faltante pero tienes años_escuela, puesto. Usa KNN o MICE.
* Numérica continua con distribución normal: mediana + indicador preserva forma sin inventar.
* Modelos sensibles a distancia o gradiente: redes, SVM, regresión logística. Necesitan números reales, no "Unknown".
* Quieres mantener tamaño muestral: en salud o fraude, cada fila vale oro.

La práctica moderna es casi siempre: imputación derivada + columna indicadora. Así el modelo aprende tanto el valor probable como el patrón de ausencia.

### Tabla rápida de decisión

| Situación | Estrategia recomendada | Por qué|
| :--- | :---: | ---: |
| Categórica, nulo = "prefirió no decir"| "Unknown" + indicador |El silencio es señal, MNAR|

Numérica, 2% MCAR, modelo árbol| Mediana | Simple, sin sesgo |

Numérica, 25% MAR, correlación fuerte | MICE / regresión + indicador | Aprovechas información de otras columnas|

Target faltante | Eliminar fila | No entrenas con fantasía|

60% faltante en columna no crítica | Eliminar columna | Ruido, poca señal|

### Errores que veo seguido en todos lados
* Imputar antes de hacer train-test split. Fuga de datos garantizada. Hazlo dentro del pipeline.
* Usar media en distribuciones sesgadas (ingresos). La mediana gana siempre.
* Tratar todos los nulos igual. En el mismo dataset puedes tener "Unknown" para medio_pago, mediana para edad, y borrar filas para diagnóstico.
* No documentar la estrategia. En seis meses nadie sabrá por qué "Unknown" existe.

Si quieres, pásame una muestra de tu dataset con el % de nulos por columna y el tipo de modelo que planeas, y te armo el pipeline exacto de imputación con código.

## Pregunta
Expresiones Regulares en Data Science: ¿Cuál es el papel de las Expresiones Regulares (REGEX) dentro de un pipeline de limpieza de datos con Python y Pandas? ¿En qué situaciones son indispensables frente a métodos más simples como .str.replace() o .str.strip()? Proporciona 2 ejemplos de patrones REGEX comunes en la limpieza de texto real.

## Respuesta
REGEX en data science no es para presumir sintaxis críptica. Es el cuchillo cuando .str.strip() es una cuchara.

En un pipeline de Pandas, regex vive en la capa de estandarización y extracción, justo después de leer y antes de tipar. Ahí convierte texto sucio en columnas útiles sin pasar por 20 replaces encadenados.

### Dónde encaja en Python + Pandas
* df['col'].str.replace(patron, '', regex=True) : limpieza masiva
* str.extract(patron) : sacar una pieza y crear feature nueva
* str.contains(patron) : filtrar o validar
* str.split(patron) : tokenizar cuando el separador varía

Pandas vectoriza todo en C, así que un solo regex suele ser más rápido que cinco .replace() literales.

### ¿Cuándo es indispensable frente a .replace() o .strip()?
.str.strip() solo quita espacios al inicio y final. .str.replace(' ', '') solo cambia exactamente eso.

Usa regex cuando:

1. **El ruido tiene muchas formas.** Un teléfono puede venir como "+52 951-123-4567", "(951)1234567", "951 123 4567". Un replace literal necesitaría tres pasadas.

2. **Necesitas extraer, no borrar.** Quieres el CP de "Col. Centro, Oaxaca C.P. 68000". No hay replace que lo saque.

3. **El patrón importa más que el texto.** Validar un RFC, un email, una URL. No buscas una cadena fija.

4. **Limpieza compuesta en un paso.** Quitar URLs, emojis y espacios múltiples de un comentario.

Si es un cambio fijo y único, quédate con .replace(). Si es una familia de variantes, regex gana.

### 2 patrones que uso casi diario en datos reales mexicanos
1) **Normalizar teléfonos a solo dígitos, conservando lada**
Problema típico en CRM de Oaxaca: mismo número escrito de 6 formas distintas.

``` python
import pandas as pd

df['tel_limpio'] = (
    df['telefono']
    .astype(str)
    .str.replace(r'[^\d]', '', regex=True)          # quita todo lo que no sea dígito
    .str.replace(r'^52(\d{10})$', r'+52\1', regex=True)  # añade +52 si falta
)
```

Patrón clave: r'[^\d]'

* [^...] niega un conjunto
* \d es cualquier dígito
* Resultado: borra paréntesis, guiones, espacios, "ext", en una pasada.
Después puedes validar longitud con str.match(r'^\+52\d{10}$').

Esto es imposible con .replace('-','').replace('(','')... porque nunca sabes todas las variantes.

2) **Extraer email válido y limpiar texto de URLs y espacios**
En formularios libres la gente pega "mi correo es JUAN.PEREZ@GMAIL.COM , revisa https://bit.ly/abc"

``` python
# extraer primer email
df['email'] = df['comentario'].str.extract(
    r'([a-z0-9._%+-]+@[a-z0-9.-]+\.[a-z]{2,})',
    flags=re.IGNORECASE
).str.lower()

# limpiar el texto original
df['comentario_limpio'] = (
    df['comentario']
    .str.replace(r'https?://\S+|www\.\S+', '', regex=True)  # quita URLs
    .str.replace(r'\s+', ' ', regex=True)                  # colapsa espacios, tabs, saltos
    .str.strip()
)
```

Patrones clave:

r'([a-z0-9._%+-]+@[a-z0-9.-]+\.[a-z]{2,})' captura la estructura real de email, no la palabra "correo".
r'https?://\S+' atrapa http y https sin importar dominio.
r'\s+' reemplaza cualquier secuencia de espacios, mucho más robusto que .replace(' ', ' ').
Con .str.strip() solo quitarías los bordes, dejarías la URL y los dobles espacios internos.

### Buenas prácticas rápidas
* Compila si lo reusas mucho: pat = re.compile(r'...') luego df['col'].str.replace(pat, '').
* Siempre prueba con str.extract primero, ves qué capturas antes de borrar.
* En Pandas 2.x, regex=True ya es default en replace, pero escríbelo explícito para legibilidad.
* Para RFC/CURP mexicanos, no inventes: usa r'[A-Z&Ñ]{3,4}\d{6}[A-Z0-9]{3}' y valida después con el dígito verificador, regex solo limpia formato.

## Pregunta
Tipos de Datos y Eficiencia en Pandas: ¿Cómo impacta la correcta asignación de tipos de datos (como int, float, category u object) en el rendimiento de un DataFrame de Pandas? ¿Por qué es importante convertir columnas categóricas al tipo category en lugar de dejarlas como object, especialmente cuando el dataset tiene millones de registros?

## Respuesta

El tipo de dato en Pandas no es cosmético. Es la diferencia entre un DataFrame que cabe en RAM y uno que te mata el kernel, sobre todo con millones de filas como las que manejas en Oaxaca.

Pandas guarda todo en bloques NumPy contiguos, excepto object, que es un arreglo de punteros a objetos Python sueltos en memoria. Ahí está el costo.

### Qué cambia cuando asignas bien
1. Memoria

   * ==int64== usa 8 bytes por valor. Si tu edad va de 0 a 100, int8 usa 1 byte. 8 veces menos.
   * ==float64== igual, 8 bytes. Si no necesitas doble precisión, float32 corta a la mitad.
   * object para strings: cada "Tamazulapam" es un objeto Python (∼49 bytes) más un puntero de 8 bytes en el array. Un millón de repeticiones = ∼57 MB solo para esa palabra repetida.
   * category: guarda una sola vez la lista de categorías y luego un array de códigos tipo int8/int16/int32. Ese mismo millón de "Tamazulapam" pasa a ∼1 MB de códigos + unos KB de diccionario.
2. Velocidad

    * Operaciones vectorizadas en int/float corren en C y aprovechan caché de CPU. Con object, Pandas tiene que llamar a Python para cada comparación de string.
    * groupby, merge, value_counts sobre category compara enteros, no strings. En 10M filas, un groupby por estado pasa de 2.3 s a 0.18 s en pruebas típicas.
    * Menos memoria = menos swapping, más probabilidad de que el DataFrame quede en L3.
3. Correctitud

* Int64 nullable evita que Pandas te convierta enteros con nulos a float64 y pierdas el tipo.
* category ordenada te permite comparaciones lógicas reales: "bajo" < "medio" < "alto".

### Por qué category aplasta a object en millones de registros
Imagina una columna municipio con 570 municipios de Oaxaca repetidos en 5 millones de encuestas.

Con object:

* 5,000,000 punteros × 8 bytes = 40 MB
* 5,000,000 strings Python ∼50 bytes promedio = 250 MB
* Total ∼290 MB, y cada df['municipio'] == 'Tamazulapam' recorre y compara 5M strings.
Con category:

* Categorías únicas: 570 strings ∼30 KB
* Códigos: 5,000,000 × int16 (2 bytes porque 570 < 32,767) = 10 MB
* Total ∼10 MB. La comparación es códigos == 42, operación NumPy instantánea.

No es solo ahorro. Es que el DataFrame completo deja de crecer exponencialmente cuando añades más columnas categóricas como sexo, tipo_localidad, estado_civil.

### Cuándo convertir y cuándo no
Convierte a category si:

* Cardinalidad baja: <10% de valores únicos respecto a filas. Ideal para estados, municipios, SI/NO, códigos de producto.
* Vas a agrupar, filtrar o hacer one-hot después.
* La columna es inmutable en el pipeline.

No conviertas si:

* Casi todo es único: CURP, email, UUID. El diccionario sería tan grande como los datos y perderías velocidad en inserciones.
* Necesitas operaciones de texto libre: .str.contains(), regex. Category primero tendría que reconvertirse a string.

### Cómo aplicarlo en la práctica

``` Python
# 1. Revisa memoria real
df.info(memory_usage='deep')

# 2. Downcast numéricos
df['edad'] = pd.to_numeric(df['edad'], downcast='integer') # int8 si cabe
df['ingreso'] = pd.to_numeric(df['ingreso'], downcast='float')

# 3. Categóricas con control
cat_type = pd.CategoricalDtype(categories=sorted(df['municipio'].unique()), ordered=False)
df['municipio'] = df['municipio'].astype(cat_type)

# 4. Para strings nuevos en Pandas 2.x, considera pyarrow
df['comentario'] = df['comentario'].astype('string[pyarrow]') # mejor que object, pero no reemplaza category

```

Después de eso vuelve a correr info. Es común ver reducciones de 70-90% en datasets de encuestas INEGI.

### Regla de oro para datasets grandes
Primero limpia, luego tipa. No conviertas a category antes de normalizar "OAXACA", "Oaxaca ", "oax". Si no, crearás tres categorías distintas y perderás el beneficio.

Con millones de registros, el tipo correcto no es optimización fina. Es lo que decide si tu pipeline corre en una laptop de 16 GB o necesitas un cluster.

## Pregunta


## Respuesta

Aquí tienes el orden que uso cuando paso de una base cruda, con todo revuelto, a una que ya puedes modelar sin miedo. No es teoría, es el flujo que evita que tengas que repetir trabajo porque tipaste antes de limpiar.

Piensa en tres bloques: **entender, limpiar, optimizar.**

### Paso 1: Congela y diagnostica
No toques nada todavía.

* Haz una copia de trabajo. Guarda el hash del archivo original.
* Corre un perfil rápido: df.info(memory_usage='deep'), df.isna().mean(), df.nunique(), df.describe(include='all').
* Toma una muestra de 10,000 filas para iterar rápido. Con millones de registros en Oaxaca, probar regex en todo el dataset es suicidio.

> Salida: una libreta con "qué está roto y cuánto".

### Paso 2: Limpieza estructural
Arregla el contenedor antes del contenido.

* Renombra columnas a snake_case, sin acentos ni espacios
* Elimina columnas 100% vacías o constantes inútiles.
* Corrige filas desplazadas, encabezados repetidos dentro del archivo.

>Por qué primero: si después haces astype, no quieres hacerlo dos veces.

### Paso 3: Normalización de texto base
Aquí entra regex, antes de cualquier tipado.

* strip, colapsa espacios: .str.replace(r'\s+', ' ', regex=True).str.strip()
* Unifica mayúsculas: municipios a Title Case, códigos a UPPER.
* Quita caracteres invisibles: \u200b, \xa0.

> Ejemplo: " oaxaca ", "OAXACA", "Oax" deben quedar en "Oaxaca" antes de convertir a category. Si no, crearás tres categorías distintas.

### Paso 4: Estandarización de formatos
Ahora que el texto es consistente, extrae piezas.

* Fechas: pd.to_datetime(errors='coerce', dayfirst=True)
* Teléfonos, CURP, RFC, CP: usa regex para extraer, no para adivinar. Guarda el original en _raw por si necesitas auditar.
* Números con comas o pesos: quita símbolos, luego convierte.

> No tipes aún, solo deja strings limpios.

### Paso 5: Decide qué hacer con los nulos
Con el texto ya normalizado puedes ver el patrón real de faltantes.

* Identifica mecanismo: MCAR, MAR, MNAR.
* Aplica la estrategia: "Unknown" para categóricas informativas, mediana + indicador para numéricas, borra solo si el target falta.
* Crea columnas _is_missing cuando el nulo sea señal.

> Hazlo antes de tipar porque astype('Int64') maneja nulos, astype('int8') no.

### Paso 6: Tipado eficiente
Ahora sí, convierte.

* Numéricos: pd.to_numeric(downcast='integer'/'float'). Usa Int64 nullable si hay nulos.
* Categóricas de baja cardinalidad: astype('category'). Hazlo después del paso 3.
* Texto libre largo: string[pyarrow] en Pandas 2.x, consume la mitad que object.
* Fechas ya convertidas: asegúrate que queden datetime64[ns].

> Verifica ganancia: df.info(memory_usage='deep') antes y después. Es normal bajar 70%.

### Paso 7: Validación de reglas de negocio
Con tipos correctos las reglas son baratas.

* Rangos: edad entre 0 y 120, ingreso >=0.
* Coherencia: fecha_nacimiento < fecha_encuesta.
* Listas válidas: municipio debe estar en catálogo INEGI de Oaxaca.

> Marca errores en una columna validacion_flag, no borres todavía.

### Paso 8: Deduplicación inteligente
No uses drop_duplicates() a ciegas.

* Define clave natural: CURP, o combinación nombre+fecha_nac+municipio.
* Ordena por calidad, quédate con el registro más completo.
* Para millones, usa blocking: primero agrupa por municipio, luego compara dentro.

### Paso 9: Cierre y documentación
* Guarda en formato columnar: Parquet con pyarrow, preserva categories.
* Escribe un diccionario de datos: qué significa cada columna, qué imputación aplicaste, qué categorías existen.
* Versiona el pipeline como funciones, no como celdas sueltas. Así puedes repetirlo cuando llegue la base de junio.

### Resumen rápido
|Orden	| Acción	| Por qué en ese punto|
|:--|:--|:--|
1	| Diagnóstico	| Evitas limpiar lo que no existe |
2-4	| Limpieza texto y formatos	| Evitas categorías duplicadas |
5	| Nulos	| Necesitas texto limpio para decidir |
6	| Tipado |	Aprovecha memoria y velocidad |
7-8	| Validación y dedup |	Trabajas sobre datos ya tipados |
9	| Documentar |	Reproducibilidad |