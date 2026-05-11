# Apuntes
## Evolución: Explica la relación técnica entre NumPy y Pandas. ¿Por qué se dice que Pandas es una evolución de NumPy para el análisis de datos tabulares?

### NumPy vs Pandas: relación técnica

NumPy y Pandas no son competidores, son capas distintas de la misma pila. NumPy es el cimiento; Pandas es la casa que se construyó encima para vivir con datos tabulares del mundo real.

#### NumPy: el motor numérico

NumPy aporta una sola idea poderosa: el `ndarray`.

- Es un bloque contiguo de memoria en C, con un solo `dtype` homogéneo (todo int64, todo float64, etc.).
- Sobre ese bloque corren las ufuncs vectorizadas, escritas en C y Fortran, que hacen sumas, productos, álgebra lineal y broadcasting sin bucles de Python.
- No tiene etiquetas, no entiende de nombres de columnas, y el valor faltante clásico es `np.nan`, que solo funciona bien para floats.

Por eso NumPy es esencial para la mayoría de las bibliotecas de ciencia de datos, ya que proporciona una base eficiente para el almacenamiento y el cálculo de datos numéricos.

#### Pandas: la capa tabular construida sobre NumPy

Pandas no reescribió ese motor, lo reutilizó. En la documentación oficial se dice literalmente: *for most data types, pandas uses NumPy arrays as the concrete objects contained with a Index, Series, or DataFrame*.

Técnicamente:

- **Series** = un `ndarray` 1D + un `Index` (etiquetas). Es la versión de NumPy con nombre en cada posición.
- **DataFrame** = colección de Series que comparten el mismo Index. Internamente, hasta pandas 1.x, un BlockManager agrupaba columnas del mismo dtype en un solo ndarray; desde pandas 2.0 hay también ArrayManager y soporte Arrow, pero el caso por defecto sigue siendo NumPy.
- Cada columna conserva su dtype homogéneo de NumPy, pero el DataFrame en conjunto puede ser heterogéneo: una columna int64, otra object, otra datetime64[ns].

Por eso la descripción clásica encaja: Pandas proporciona estructuras para manipular datos y su DataFrame es similar a una tabla de base de datos o a Excel, mientras NumPy se queda en arrays multidimensionales.

#### ¿Por qué se dice que Pandas es una evolución para datos tabulares?

No es marketing, son cuatro saltos técnicos que NumPy solo no resuelve:

##### 1. De posición a etiqueta
- NumPy: `arr[3]` significa "cuarta fila".
- Pandas: `s['2024-03-15']` o `df.loc['cliente_42']` significa "busca por significado". La alineación es automática: si sumas dos Series con índices distintos, Pandas empareja por etiqueta, no por orden.

##### 2. Datos sucios del mundo real
- Valores faltantes tipados: `Int64` nullable, `BooleanDtype`, `StringDtype`, `NaT` para fechas. NumPy solo tenía `nan` para float.
- Tipos temporales ricos: `Timestamp` con zona horaria, `Period`, `Interval`, que NumPy no representa nativamente.

##### 3. Semántica de tabla
- NumPy no sabe qué es un `groupby`, un `merge` o un `pivot`. Pandas implementa esas operaciones en Cython, pero delega el cálculo numérico al ndarray subyacente.
- Ejemplo: `df.groupby('país').ingreso.mean()` ejecuta por debajo un bucle en C sobre arrays NumPy.

##### 4. Interoperabilidad sin fricción
- `.values` y `.to_numpy()` te devuelven el ndarray crudo cuando necesitas velocidad pura.
- Al revés, puedes crear una Series desde cualquier función NumPy:
  ```python
  pd.Series(np.random.normal(size=1000))
  ```

#### El costo de la evolución
No es gratis. Añadir etiquetas, alineación y tipos nullable introduce overhead de memoria y CPU. Para operaciones puramente numéricas sobre matrices enormes y homogéneas, NumPy sigue siendo más rápido.

Pandas gana cuando el problema es tabular, heterogéneo y con contexto: limpiar CSVs, unir tablas, manejar fechas, imputar nulos, hacer análisis exploratorio.

---

**En resumen**: NumPy te da el vector rápido. Pandas toma esos vectores, les pone nombre, calendario y reglas de negocio, y te deja pensar en filas y columnas en vez de en offsets de memoria. Por eso se le llama evolución: no reemplaza a NumPy, lo envuelve para que el análisis de datos tabulares se parezca más a cómo pensamos los datos, no a cómo los guarda la máquina.

## Objetos: Dame un ejemplo de la vida real (fuera de la programación) que ilustre la diferencia entre una Series y un DataFrame de Pandas.

### Series vs DataFrame en la vida real

Piensa en el puesto de doña Rosa en el mercado.

#### 1. Series: una sola libreta, una sola medida

Doña Rosa lleva una libreta donde anota **solo el precio del kilo de maíz** cada día de la semana.

La fecha es la etiqueta, el precio es el valor.

| Fecha (índice) | Precio maíz |
|----------------|-------------|
| lun 14 abr     | $28         |
| mar 15 abr     | $27         |
| mié 16 abr     | $29         |
| jue 17 abr     | $30         |

Esto es una **Series**:
- Una dimensión.
- Cada fila tiene una etiqueta (la fecha).
- Solo mide una cosa.

Si le preguntas "¿cuánto costaba el martes?", busca por la etiqueta, no por "la segunda línea".

#### 2. DataFrame: el cuaderno completo del puesto

El sábado, doña Rosa abre su cuaderno grande. En la misma página anota, para cada día, el precio de **varios productos**.

Todas las filas comparten las mismas fechas, pero ahora hay varias columnas.

| Fecha (índice) | Maíz | Frijol | Chile costeño |
|----------------|------|--------|---------------|
| lun 14 abr     | $28  | $42    | $95           |
| mar 15 abr     | $27  | $42    | $98           |
| mié 16 abr     | $29  | $43    | $100          |
| jue 17 abr     | $30  | $44    | $102          |

Esto es un **DataFrame**:
- Es una colección de Series que comparten el mismo índice (las fechas).
- Cada columna es una Series independiente (la Series "Maíz" es exactamente la libreta de antes).
- Puedes mezclar tipos: precios en pesos, pero podrías añadir una columna "¿Hubo lluvia?" con Sí/No.

#### La diferencia clave

- **Series = una lista etiquetada.** Como la tira de precios del maíz pegada en la pared.
- **DataFrame = la tabla entera.** Como el cuaderno donde cada página alinea varias tiras por la misma fecha.

En Pandas pasa igual: un DataFrame no es otra cosa que varias Series viviendo juntas, alineadas por su índice. Por eso puedes tomar la columna "Maíz" del DataFrame y trabajar con ella como si fuera la libreta original.

## Comparación: ¿En qué escenarios es superior utilizar Pandas en lugar de Microsoft Excel para el análisis de datos? Menciona al menos 3 ventajas técnicas relacionadas con el volumen de datos y la automatización.

### ¿Cuándo gana Pandas frente a Excel?

Excel es excelente para explorar rápido y mostrar una tabla bonita. Pandas toma la delantera cuando el análisis deja de ser "una hoja que abro" y se convierte en "un proceso que debe correr solo, con muchos datos".

### Escenarios típicos donde Pandas es superior

- **Logs o ventas diarias de más de 1 millón de filas.** Una cadena de tiendas en Oaxaca que exporta 3 millones de tickets al mes.
- **ETL nocturno.** Descargar 50 CSVs de proveedores, limpiarlos, unirlos y subir el resultado a una base de datos a las 3am.
- **Análisis reproducible en equipo.** Tres analistas deben obtener exactamente el mismo reporte cada lunes, sin depender de quién hizo clic dónde.

#### 3 ventajas técnicas (volumen y automatización)

##### 1. Volumen de datos sin el techo de 1,048,576 filas
- **Excel**: límite duro de 1,048,576 filas por hoja y cerca de 2 GB de memoria antes de colgarse.
- **Pandas**: lee por chunks (`pd.read_csv(..., chunksize=500_000)`), usa formatos columnares como Parquet, y reduce memoria con dtypes (`category`, `Int32`, `float32`).
- **Resultado**: puedes procesar 10 o 50 millones de registros en una laptop, o escalar a Dask sin reescribir la lógica.

> Caso real: limpiar un histórico de 12 GB de precios de mercado. En Excel es imposible abrirlo; en Pandas lo lees en trozos, filtras y guardas solo lo útil.

##### 2. Automatización nativa, sin clicks
- **Excel**: dependes de macros VBA o Power Query, que viven dentro del archivo y se rompen si alguien mueve una columna.
- **Pandas**: es código Python plano. Lo programas una vez:
  ```python
  df = pd.read_sql("SELECT * FROM ventas WHERE fecha = CURDATE()", conn)
  resumen = df.groupby("producto").ingreso.sum()
  resumen.to_excel("reporte_diario.xlsx")
  ```

Puedes agendarlo con cron, Task Scheduler o Airflow, conectarlo a una API, y enviar el reporte por correo sin abrir nunca una ventana.
Ventaja técnica: ejecución headless, sin sobrecarga de interfaz gráfica, y tiempos predecibles.

##### 3. Reproducibilidad y control de versiones
- **Excel**: el historial está en "versión_final_v3_DEFINITIVA.xlsx". Dos personas cambian fórmulas y nadie sabe cuál es la buena.
- **Pandas**: el script vive en Git. Cada transformación (merge, pivot, fillna) queda documentada línea por línea. Puedes hacer git diff y ver exactamente qué cambió en la lógica, no solo en el resultado.
- Además, las operaciones se ejecutan sobre NumPy en C, no celda por celda. Un groupby de 5 millones de filas tarda segundos, no minutos recalculando fórmulas.
##### Ventaja extra: joins y limpieza complejos a escala
- Unir 12 tablas por claves compuestas, aplicar ventanas móviles (rolling), o normalizar texto con regex en millones de filas es una línea en Pandas. En Excel necesitarías Power Pivot, varias hojas intermedias y fórmulas propensas a error.

| Característica | Excel | Pandas |
|----------------|-------|--------|
| Filas máximas prácticas | ~1M | decenas de millones con chunking |
| Automatización | manual o VBA frágil | script programable, sin GUI |
| Trazabilidad | archivo binario | código versionable en Git |
| Velocidad de cálculo | recalcula celdas | vectorizado en C vía NumPy |

Usa Excel para prototipar y presentar. Usa Pandas cuando el dato es grande, llega todos los días y no puedes permitirte hacerlo a mano.

## Reflexión
Me sorprendió que se adaptó a mi nivel de conocimiento, así mismo usó mi contexto para los ejemplos, me permitió convertir las respuestas a un formato diferente y que reconoce cuando se equivoca o miente.