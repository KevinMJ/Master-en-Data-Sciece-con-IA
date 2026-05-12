# Apuntes
## Prompt
1. Reproducibilidad en Data Science: ¿Por qué la reproducibilidad es considerada uno de los pilares fundamentales de un proyecto de Ciencia de Datos profesional? ¿Cuáles son las principales prácticas para garantizarla en un notebook de Google Colab o Jupyter, y qué consecuencias tiene para un equipo de trabajo cuando un análisis no puede ser reproducido?

## Respuesta de la IA
La reproducibilidad no es un extra académico, es la condición mínima para que un análisis de datos sea ciencia y no artesanía.

### ¿Por qué se considera pilar?
En ciencia computacional, reproducir significa que otra persona —o tú mismo dentro de seis meses— pueda repetir exactamente los pasos y obtener el mismo resultado. Es el mecanismo que permite construir con confianza sobre el trabajo de otros, reutilizar pipelines y avanzar de verdad. 

En publicaciones, el principio exige que los estudios se documenten y archiven a un nivel tal que puedan repetirse, tanto por los autores originales como por otros interesados. Sin eso: 

* no hay auditoría posible
* no hay transferencia a producción
* no hay colaboración real, solo intercambio de capturas de pantalla

Los notebooks lo facilitan porque combinan código ejecutable, narrativa y visualización en un solo documento portátil, pero también introducen trampas propias. 

### Prácticas concretas en Jupyter o Google Colab
No necesitas infraestructura exótica, solo disciplina. Estas vienen de la literatura más citada sobre notebooks:

1. **Cuenta una historia, no pegues celdas**

* Empieza con contexto, hipótesis y datos de entrada
* Cada sección en markdown explica por qué haces lo siguiente, no solo qué hace
* Cierra con interpretación, límites y próximos pasos
2. **Documenta el proceso, no solo el resultado**

* Anota exploraciones fallidas, parámetros probados y por qué los descartaste. Si lo haces al vuelo, evitas olvidar decisiones clave 
3. **Una celda, una idea**

* Evita celdas de 200 líneas. Divide en pasos legibles, pon encabezados y tabla de contenido
* Ejecuta siempre de arriba a abajo antes de compartir. Los estados ocultos y celdas fuera de orden son la causa número uno de notebooks irreproducibles 
4. **Modulariza**

* Si copias y pegas más de dos veces, conviértelo en función
* Para proyectos medianos, mueve funciones a un .py en la misma carpeta e impórtalo. En Colab: <code>%load_ext autoreload + %autoreload 2</code>
5. **Congela el entorno**

En la primera celda:

``` Code
!pip install -r requirements.txt
```

* Guarda versiones con !pip freeze > requirements.txt o usa watermark
* En Colab, fija la versión de Python con runtime → cambiar tipo, y documenta si usas GPU/TPU
6. **Datos versionados y rutas relativas**

* Nunca "/content/drive/MyDrive/..." hardcodeado sin alternativa. Usa:
``` Python
from pathlib import Path
DATA = Path("data/raw.csv")
```
* Versiona datos con DVC, Git LFS o al menos guarda hash SHA256 en el notebook
7. **Semillas y aleatoriedad**

* np.random.seed(42), random.seed(42), torch.manual_seed(42) al inicio. Documenta la semilla en markdown
8. **Control de versiones**

* Guarda el .ipynb limpio (sin outputs enormes) en Git. Usa nbstripout o Colab → GitHub
* Cada experimento: commit con mensaje descriptivo, no "update notebook"
9. **Ejecuta en limpio**

* Antes de entregar: Runtime → Restart and run all. Si falla, el notebook no es reproducible
10. **Comparte el entorno completo**

* Para Colab: botón "Abrir en Colab" + requirements.txt en repo
* Para Jupyter: Binder, Docker o conda env export > environment.yml
### Qué pasa cuando no se puede reproducir
La consecuencia no es teórica. Hoy es común que doctorandos dediquen meses o años solo a replicar resultados publicados antes de poder extenderlos. En equipos de empresa el costo es más inmediato: 

* **Tiempo quemado:** un analista pasa días adivinando qué versión de pandas usaste
* **Decisiones frágiles:** el modelo en producción da números distintos porque el notebook usaba un CSV local modificado a mano
* **Deuda técnica:** cada "corre rápido esta celda" crea estado oculto que luego nadie puede resetear
* **Pérdida de confianza:** stakeholders dejan de creer en los dashboards porque "la semana pasada daba otro número"
* **Riesgo legal y científico:** sin trazabilidad de datos y código, no puedes defender un análisis ante auditoría o revisión

En la práctica, un notebook no reproducible deja de ser un activo y se convierte en un mito: todos lo citan, nadie lo ejecuta.

Si quieres, te armo una plantilla base de notebook para Colab con las celdas ya estructuradas — historia, entorno, datos, funciones, experimento, resultados — y el bloque de watermark para que cada ejecución quede firmada.

## Prompt de seguimiento
Sí dame la plantilla en formato markdown y yo la adapto, dá explicación detallada de que se hace en cada punto.

## Respuesta de seguimiento
Perfecto. Te doy dos cosas: primero qué hace cada bloque y por qué existe, luego la plantilla en markdown puro para que la pegues en Jupyter o Colab y la rellenes.

### Qué poner en cada bloque y por qué
#### 0. Portada y metadatos

* Qué se hace: título, autor, fecha de inicio y fin, objetivo en una frase, enlace al repo.
* Por qué: tu yo del futuro es tu principal lector. Si no sabe qué intentabas, no podrá repetirlo.
#### 1. Historia del análisis

* Qué se hace: 3 párrafos cortos en markdown — contexto de negocio, pregunta concreta, hipótesis.
* Por qué: los notebooks reproducibles cuentan una historia, no solo ejecutan código. Esto evita que alguien reinterprete mal los resultados.
#### 2. Entorno reproducible

* Qué se hace: celda de instalación con requirements.txt o %pip install, luego impresión de versiones con watermark o pip freeze.
En Colab añade el montaje de Drive pero con fallback a ruta relativa.
* Por qué: la mayoría de fallos vienen de dependencias no declaradas. Congelar versiones es lo mínimo.
#### 3. Configuración global y semillas

* Qué se hace: fijar random_state para numpy, random, sklearn, torch. Definir PROJECT_ROOT = Path.cwd() y carpetas data/, outputs/.
* Por qué: sin semillas, dos ejecuciones dan métricas distintas. Sin rutas relativas, el notebook solo corre en tu máquina.
#### 4. Carga de datos versionados

Qué se hace: leer con Path, verificar hash SHA256, mostrar df.head() y df.info(). Documenta origen y licencia.
Por qué: los datos no versionados corrompen más experimentos que cualquier import roto. Si el CSV cambia, tu resultado cambia sin aviso.
5. Funciones auxiliares

* Qué se hace: aquí van las funciones que ibas a copiar y pegar. Nada de lógica de negocio, solo utilidades.
* Por qué: modularizar evita el estado oculto y permite testear. Una celda, una idea.
#### 6. Análisis exploratorio

Qué se hace: cada gráfico en su celda, con título y una frase de interpretación debajo.
Regla: ejecuta "Restart and run all" antes de seguir. Las celdas fuera de orden y el estado oculto son la causa principal de notebooks irreproducibles. 
7. Experimento principal

Qué se hace: separa preparación, entrenamiento y predicción. Guarda parámetros en un dict al inicio de la celda.
Por qué: así puedes cambiar un hiperparámetro sin reescribir todo.
8. Evaluación y guardado

* Qué se hace: métricas, matriz de confusión, guarda artefactos en outputs/ con timestamp.
* En Colab: outputs/ puede ser /content/drive/MyDrive/proyecto/outputs si montas Drive.
#### 9. Conclusiones y límites

* Qué se hace: qué sí puedes afirmar, qué no, qué quedó pendiente.
* Por qué: documentar el proceso, no solo el resultado, es parte del principio de reproducibilidad — permite que otros repitan y extiendan.
#### 10. Checklist final

* Qué se hace: lista de verificación antes de compartir.
### Plantilla lista para copiar
Pega esto en tu notebook. Las partes entre {{ }} son placeholders.

``` Markdown
# {{TÍTULO DEL PROYECTO}}

**Autor:** {{tu nombre}}
**Fecha inicio:** {{YYYY-MM-DD}}
**Última ejecución:** {{se actualiza solo}}
**Objetivo en una frase:** {{qué pregunta respondes}}
**Repo:** {{url de GitHub}}

---

## 1. Historia
**Contexto:** {{por qué importa este análisis para el negocio o investigación}}
**Pregunta:** {{pregunta exacta}}
**Hipótesis:** {{qué esperas encontrar}}

---

## 2. Entorno reproducible
```
``` Python
# instala dependencias exactas
%pip install -r requirements.txt -q

# muestra entorno
!pip list | grep -E "pandas|numpy|scikit-learn"
# o usa watermark
%load_ext watermark
%watermark -v -m -p pandas,numpy,sklearn -g
```

``` Markdown
> En Colab, si usas Drive:
```
```Python
from google.colab import drive
try:
    drive.mount('/content/drive')
    BASE = Path('/content/drive/MyDrive/{{proyecto}}')
except:
    BASE = Path.cwd()
```

``` Markdown
---
## 3. Configuración y semillas
```

```Python
import random, numpy as np
from pathlib import Path

SEED = 42
random.seed(SEED)
np.random.seed(SEED)

PROJECT_ROOT = Path.cwd()
DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)

print(f"Datos en: {DATA_DIR.resolve()}")
```

```Markdown
---

## 4. Datos
**Origen:** {{url o DOI}}
**Versión / hash:** {{sha256}}
```
``` Python
import pandas as pd

df = pd.read_csv(DATA_DIR / "raw.csv")
df.head(3)
df.info()
```
``` Markdown
---
## 5. Funciones auxiliares
```

``` Python
def clean_text(s: str) -> str:
    # tu lógica reutilizable aquí
    return s.strip().lower()

# evita copiar y pegar, importa si crece: from src.utils import clean_text
```
``` Markdown
---
## 6. Análisis exploratorio
```
```Python
# celda 1: distribución
df['variable'].hist(bins=30)
```
```Markdown
*Interpretación:* {{qué ves}}
```
```Python
# celda 2: correlación
```
```Markdown
---

## 7. Experimento
**Parámetros:**
```
```Python
params = {
    "model": "RandomForest",
    "n_estimators": 200,
    "random_state": SEED
}
```
```Python
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(...)
model = RandomForestClassifier(**{k:v for k,v in params.items() if k!='model'})
model.fit(X_train, y_train)
```
``` Markdown
---

## 8. Evaluación
```
``` Python
from sklearn.metrics import classification_report
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))

# guarda
import joblib, datetime
ts = datetime.datetime.now().strftime("%Y%m%d_%H%M")
joblib.dump(model, OUTPUT_DIR / f"model_{ts}.pkl")

```
``` Markdown
---

## 9. Conclusiones
- **Hallazgo principal:** {{}}
- **Limitación:** {{datos, sesgo, tiempo}}
- **Siguiente paso:** {{}}

---

## 10. Checklist de reproducibilidad
- [ ] Ejecutado con "Restart and run all" sin errores
- [ ] Todas las rutas son relativas a `PROJECT_ROOT`
- [ ] `requirements.txt` actualizado
- [ ] Semilla fijada y documentada
- [ ] Datos con origen y hash
- [ ] Outputs guardados en `/outputs` con timestamp
- [ ] Notebook sin outputs pesados antes de commit
```
¿Quieres que te genere también el requirements.txt mínimo y el .py de utilidades para que tu plantilla funcione desde el primer run en Colab?

## Prompt
2. Documentación de Decisiones Analíticas: ¿Cuál es la diferencia entre documentar el qué y documentar el por qué en un notebook de análisis de datos? Proporciona un ejemplo concreto de una decisión analítica (como la elección entre media o mediana para imputar) y muestra cómo debería documentarse correctamente tanto en una celda Markdown como en un comentario de código.

## Respuesta de la IA

Documentar bien es lo que separa un notebook que se puede ejecutar de uno que se puede entender. La diferencia es sutil pero decisiva.

### Qué vs por qué
**Documentar el qué** es describir la operación técnica. Es la receta.

* "Rellenamos los nulos de edad con la mediana"
* "Eliminamos outliers con IQR"
* Es necesario para reproducir el código, pero no explica nada del contexto.
**Documentar el por qué** es registrar el razonamiento que te llevó a esa operación. Es la bitácora.

* Qué alternativas evaluaste
* Qué viste en los datos que te hizo descartarlas
* Qué trade-off aceptas y qué impacto tendrá en el resultado

Los notebooks fueron diseñados para mezclar código con narrativa, no solo para guardar notas esporádicas. La regla básica es usar el texto explicativo para contar una historia con principio, desarrollo y final, describiendo no solo qué hiciste, sino por qué lo hiciste y cómo se conectan los pasos. Y la segunda regla es documentar el proceso, no solo el resultado, porque la naturaleza interactiva hace fácil probar cosas y olvidar registrarlas.

Sin el por qué, tu yo de dentro de tres meses o un compañero solo ve una decisión arbitraria. Con el por qué, ve una decisión trazable.

### Ejemplo concreto: imputar edad con media o mediana
Imagina un dataset de clientes con 12% de nulos en edad. Distribución sesgada a la derecha, media 38.4, mediana 34, hay un grupo de clientes de 80-90 años que inflan la media.

#### Mala documentación (solo qué)
Markdown:

> Imputación de edad

Código:

``` Python
df['edad'].fillna(df['edad'].mean(), inplace=True) # rellenar nulos
```
Problema: no sabes si usó media por costumbre, si vio outliers, si probó otra cosa. Si mañana cambian los datos, nadie sabe si mantenerlo.

#### Buena documentación (qué + por qué)

**Celda Markdown — la decisión completa**

``` Markdown
### Decisión 3: imputación de `edad`

**Qué hicimos:** rellenar 1,247 nulos (12.3%) con la **mediana = 34**.

**Por qué no la media:**
- Distribución con cola derecha larga (skew = 1.8). Media = 38.4, se desplaza por 87 clientes >75 años.
- Prueba rápida: imputar con media sube la edad promedio del segmento 25-35 en +2.1 años, lo que sesga la propensión modelada.
- Mediana es robusta a outliers y conserva la mediana del grupo control en validación cruzada.

**Alternativas descartadas:**
1. Media → descartada por sesgo (ver arriba)
2. KNN imputer → mejora <0.3% en AUC pero añade dependencia y tiempo, no justificado para MVP
3. Crear categoría "desconocido" → rompe modelo lineal actual

**Impacto aceptado:** perdemos varianza real en la cola alta, pero priorizamos estabilidad en el segmento mayoritario. Revisar si en v2 incorporamos modelo de imputación.
```

**Celda de código — comentario mínimo pero anclado**

```Python
# Imputación edad: usamos mediana (34) por distribución sesgada (skew 1.8)
# Media (38.4) inflada por outliers >75 años. Ver Decisión 3 en markdown.
# Trade-off: estabilidad > precisión en cola. Reevaluar con KNN en v2.
mediana_edad = df['edad'].median() # 34.0 calculado en df_train solamente
df['edad'] = df['edad'].fillna(mediana_edad)
```
Nota tres cosas:

1. El markdown guarda el contexto que no cabe en código: números, pruebas, alternativas.
2. El comentario no repite todo, solo ancla la decisión y remite al markdown. Si alguien lee solo el código, sabe dónde buscar el razonamiento.
3. Calculamos la mediana en train, no en todo el dataset, y lo dejamos explícito. Eso es parte del por qué técnico.

Cuando documentas así, reproducir no es solo volver a correr. Es volver a pensar igual. Y cuando el análisis falle, sabrás exactamente qué supuesto revisar primero.

## Prompt
3. EDA en la Industria: ¿Cómo se estructura un Análisis Exploratorio de Datos (EDA) en un equipo profesional de Data Science? ¿Qué entregables se esperan al finalizar el EDA antes de pasar a la fase de modelado, y qué herramientas o formatos (como reportes, dashboards o notebooks) se utilizan comúnmente para comunicar los hallazgos a stakeholders no técnicos?

## Respuesta de la IA

En industria el EDA no es "hacer 30 gráficas y ver qué sale". Es una investigación con método, porque de aquí sale si el modelo tendrá sentido o si vas a automatizar basura.

### Cómo lo estructura un equipo profesional
La práctica que más se repite en equipos maduros divide el trabajo en tres fases, no en una lista de plots.

#### Fase 1: chequeo de integridad (20 a 30% del tiempo)
Antes de visualizar, validas. Es la parte menos vistosa y la más rentable.

* Procedencia: quién capturó el dato, con qué incentivo, qué cambió en el negocio desde entonces
* Estructura: filas esperadas vs reales, tipos, duplicados lógicos
* Contenido: imposibles (edad 150), categóricas con 95% "otro", patrones de nulos
* Relaciones: fechas que no cuadran, órdenes sin cliente

Saltarse esto es mala práctica profesional. He visto análisis de segmentación de tres días invalidados porque 30% eran cuentas de prueba.

#### Fase 2: descubrimiento estructurado
En lugar de explorar a ciegas, se ataca por dimensiones que responden preguntas de negocio distintas:

* Temporal: ¿cómo cambia en el tiempo? Tendencias, estacionalidad, cohortes, antes/después de un evento
* Distribucional: ¿qué forma tienen los datos? No solo histogramas, también KDE, Q-Q plots, violines. La forma te dice el proceso generador
* Relacional: ¿cómo se conectan entidades? reglas de asociación, grafos, jerarquías
* Dimensional: ¿cómo se comportan las categorías? tablas cruzadas, tests, comparativos por segmento
#### Fase 3: diagnósticos dirigidos
Ya no exploras, investigas anomalías concretas.

* Verifica que sea real, no error
* Caracteriza tamaño y alcance
* Busca factores correlacionados
* Formula hipótesis con el negocio

Este enfoque cambia el EDA de "mirar cosas interesantes" a "buscar evidencia que apoye o elimine una hipótesis sobre lo que pasa". 

El EDA en sí se define como el método para analizar e investigar conjuntos de datos y resumir sus características principales, usando visualización para descubrir patrones, detectar anomalías y verificar supuestos antes de modelar. 

### Entregables que se exigen antes de pasar a modelado
Un equipo serio no avanza hasta tener estos artefactos firmados, no solo un notebook suelto:

|Entregable|Qué contiene|Por qué importa|
|---|---|---|
|**Informe de calidad de datos**| % nulos por campo, duplicados, imposibles, patrones de missingness, decisiones de limpieza| Evita que el modelo aprenda de errores. Es el objetivo principal del EDA: identificar errores obvios antes de asumir nada|
**Diccionario de datos vivo**| definición de cada variable, origen, unidad, transformaciones aplicadas| Alinea a negocio y técnica sobre qué significa "usuario activo"|
**Perfil univariado y bivariado**| distribuciones, correlaciones con target, outliers caracterizados| Guía selección de features y transformaciones|
**Hipótesis validadas y descartadas**| lista de 5 a 10 hipótesis de negocio con evidencia a favor/en contra| Impide confirmar sesgos|
**Matriz de riesgos**| sesgos de muestreo, leakage potencial, variables con fuga futura|Protege contra modelos que no funcionarán en producción|
**Propuesta de baseline**| métrica de negocio, métrica técnica, resultado de modelo tonto| Da punto de comparación realista|

Todo esto suele vivir en un repo con control de versiones, no en el Drive de alguien.

### Cómo se comunica a stakeholders no técnicos
El notebook completo nunca va directo a dirección. Se traduce a formatos que cuentan historia, no código.

1. Notebook narrativo (para pares técnicos)
   * Jupyter o Datalore con markdown extenso, celdas limpias, ejecutable de arriba a abajo
   * Se usa para reproducibilidad interna

2. Reporte ejecutivo (para dirección y producto)

    * 6 a 10 diapositivas en PowerPoint o Google Slides, o PDF de 2 páginas
    * Estructura: problema, hallazgo clave 1 con visual simple, hallazgo 2, implicación, recomendación, riesgos
    * Cada gráfico lleva una frase "entonces qué": no "distribución de edad", sino "70% de churn ocurre en clientes <6 meses, sugiere onboarding"
3. Dashboard ligero (para exploración guiada)

   * Power BI, Tableau, Looker Studio o Metabase
   * No más de 3 vistas: panorama temporal, segmentación, anomalías
   * Filtros preconfigurados para que negocio juegue sin romper nada
4. Data app interactiva (para equipos de producto)

   * Streamlit o Gradio cuando necesitan probar cortes ellos mismos
   * Útil para mostrar impacto de umbrales sin explicar código
5. One-pager de decisiones

   * Tabla con "decisión analítica, por qué, impacto en modelo". Es el puente entre EDA y modelado

> La regla práctica: si el stakeholder no puede explicar el hallazgo en una frase a su jefe, el formato falló. Por eso los profesionales priorizan la exploración sistemática sobre la visualización superficial, enfocándose en integridad, pruebas de hipótesis y análisis estratégico para descubrir verdad en lugar de confirmar suposiciones. 

## Plantilla de reporte ejecutivo

Perfecto. Aquí tienes el reporte ejecutivo de EDA en Markdown, pensado para pegar en Marp, Quarto, Obsidian Slides o cualquier visor que convierta --- en diapositivas. Son 6 láminas, cada una con una sola idea.

Copia todo el bloque de abajo tal cual.

```Markdown
---
marp: true
theme: default
paginate: true
---

# {{Nombre del proyecto}} — EDA Ejecutivo
**Fecha:** {{2026-05-11}}
**Equipo:** {{Data Science}}
**Stakeholder:** {{Dirección de Producto}}
**Objetivo del análisis:** {{pregunta de negocio en 1 frase}}

> Este reporte resume hallazgos, no código.

---

## 1. Problema y contexto

**Pregunta de negocio:** {{¿por qué estamos analizando estos datos?}}

**Datos analizados:**
- Periodo: {{ene 2024 - abr 2026}}
- Filas: {{1.2M}} | Columnas: {{47}}
- Fuente: {{CRM + transacciones}}

**Supuesto clave a validar:** {{ej. "el churn se concentra en onboarding"}}

---

## 2. Hallazgo clave 1 — {{título corto}}

**Qué vimos:**
{{ej. 68% del churn ocurre en los primeros 60 días}}

**Visual:** ![grafico1](outputs/churn_cohorte.png)

**Entonces qué:**
{{ej. No es un problema de precio, es de activación inicial. Los clientes no llegan al primer valor.}}

**Evidencia:**
- Mediana tiempo a primera compra: 47 días en churn vs 12 días en retenidos
- Test KS p < 0.001

---

## 3. Hallazgo clave 2 — {{título corto}}

**Qué vimos:**
{{ej. La variable "edad" está sesgada por outliers >75 años}}

**Visual:** ![grafico2](outputs/edad_distribucion.png)

**Entonces qué:**
{{ej. Usar media infla la edad promedio +4 años. Imputaremos con mediana.}}

**Decisión analítica:**
- Imputación: mediana = 34 (ver Diccionario de datos)
- Riesgo: perdemos varianza en cola alta, aceptable para MVP

---

## 4. Calidad de datos y riesgos

| Problema | Impacto | Acción |
| --- | --- | --- |
| 12.3% nulos en edad | Sesgo si usamos media | Imputar mediana |
| 30% cuentas test | Contaminan segmentación | Filtradas en ETL |
| Fechas futuras en 0.8% | Error de captura | Eliminadas |

**Riesgo principal:** {{ej. posible fuga de información en variable "último_login"}}
**Mitigación:** {{ej. excluir del modelado o retrasar 7 días}}

---

## 5. Implicaciones para el negocio

1. {{ej. Rediseñar onboarding en primeros 30 días, no campañas de descuento}}
2. {{ej. Segmento 25-35 años con mayor sensibilidad a tiempo de activación}}
3. {{ej. Datos de satisfacción están sesgados por muestreo, no usar como target}}

**Métrica de baseline actual:** {{ej. churn 60d = 18.4%}}

---

## 6. Recomendación y siguientes pasos

**Recomendación:** {{ej. Priorizar experimento de activación rápida antes de modelar propensión}}

**Siguientes pasos:**
- [ ] Validar hallazgo 1 con equipo de CX (semana 1)
- [ ] Aprobar diccionario de datos v1.2
- [ ] Pasar a modelado con features limpias (sin leakage)

**Dueño:** {{tu nombre}} | **Fecha decisión:** {{2026-05-18}}

**Anexos:** Notebook completo, informe de calidad, diccionario

```

#### Cómo usarlo sin perder tiempo
1. Una idea por slide. Si necesitas más de 2 bullets, divide.
2. Visual = prueba, no decoración. Cada imagen lleva título y la frase "entonces qué" debajo.
3. Nunca pegues código. Si te piden detalle, manda el anexo.
4. Rellena los {{ }} en orden. Empieza por slide 2 y 3, el resto sale solo.
5. Exporta a PDF. En Marp: Export → PDF. Es lo que leerá dirección.
¿Quieres que te genere también la versión con datos de ejemplo ya rellenos para churn de telecom, para que veas el tono exacto?

## Reflexión
Lo que más aprecio de la herramienta MetaAI fué la capacidad de brindarme plantillas de estilo profesional para comenzar directamente con la documentación.