# Apuntes
## Sesgo Algorítmico: 
> ¿Qué es el sesgo algorítmico en Machine Learning y cómo pueden los datos de entrenamiento introducir discriminación sistémica en un modelo? Proporciona un ejemplo real documentado donde un modelo de ML haya causado un daño social concreto (en contratación, crédito, justicia penal, etc.) y explica qué tipo de sesgo lo originó y cómo podría haberse mitigado.

El sesgo algorítmico no es un "error de cálculo", es un error sistemático que hace que un modelo de machine learning produzca resultados injustos o discriminatorios, y casi siempre refleja prejuicios socioeconómicos, raciales o de género que ya existían en el mundo real. 

Es especialmente peligroso cuando la IA decide sobre cosas que cambian vidas: salud, justicia penal o contratación. 

### Cómo los datos de entrenamiento meten discriminación
El algoritmo no nace sesgado, lo sesgamos nosotros al recopilar y codificar los datos. Pasa de varias formas: 

- **Datos históricos con sesgo.** Si entrenas con el pasado, el modelo aprende el pasado. Datos "defectuosos, no representativos o históricamente sesgados" llevan a resultados parciales que amplifican ese sesgo. IBM lo ilustra con Oakland: los datos de arrestos reflejaban la marginación histórica de afroamericanos, y un modelo predictivo policial entrenado con eso replicaría el mismo sesgo racial. 
- **Subrepresentación.** Cuando un grupo casi no aparece, el modelo no aprende a reconocerlo bien. No es maldad, es estadística: aprende de correlaciones, no de causalidad.
- **Proxies.** A veces quitamos "raza" o "género" del dataset, pero dejamos sustitutos como código postal, universidad o verbos usados en el CV. Esos proxies pueden correlacionarse accidentalmente con atributos protegidos y discriminar igual. 
- **Ciclo de retroalimentación.** El modelo toma decisiones sesgadas, esas decisiones se convierten en nuevos datos de entrenamiento, y el sesgo se refuerza con el tiempo.

### Ejemplo real: el reclutador de Amazon que no quería mujeres
Es uno de los casos mejor documentados, reportado primero por Reuters en 2018.

Amazon empezó en 2014 a construir un motor de IA para revisar currículums y puntuar candidatos de 1 a 5 estrellas, como si fueran productos. 

El modelo fue entrenado observando patrones en los currículums enviados a Amazon durante 10 años. La mayoría venían de hombres, reflejo del dominio masculino en tecnología. 

¿Qué aprendió?

- Que "hombre = buen candidato". Penaliza currículums que incluían la palabra "women's", como "women's chess club captain", y degradaba a graduadas de dos universidades solo para mujeres. 
- Favorecía verbos más comunes en CV masculinos, como "executed" y "captured". 

El daño social fue concreto: mujeres calificadas eran filtradas automáticamente antes de que un humano las viera, perpetuando la brecha de género en ingeniería.

**¿Qué tipo de sesgo fue?**

1. Sesgo histórico y de representación en los datos de entrenamiento.
2. Sesgo de muestreo: la población de entrenamiento no reflejaba la población objetivo.
3. Sesgo de correlación espuria: el modelo confundió "ser hombre" con "ser competente".

Amazon intentó parchear borrando términos explícitos, pero los ingenieros admitieron que no había garantía de que el sistema no encontrara otras formas de discriminar. Al final, la dirección perdió la esperanza y disolvió el equipo a principios de 2018. Nunca se usó solo para contratar, pero sí influyó en recomendaciones. 

Un caso paralelo en justicia penal es COMPAS, usado en tribunales de EE. UU. ProPublica mostró en 2016 que clasificaba a acusados negros con el doble de probabilidad de error como "alto riesgo" de reincidencia violenta frente a blancos, también por entrenar con datos históricos sesgados del sistema penal. 

### Cómo podría haberse mitigado
No hay una bala de plata, pero la práctica actual apunta a intervenir en todo el ciclo:

- **Datos diversos y representativos desde el inicio.** El aprendizaje automático "es tan bueno como los datos que lo capacitan", y esos datos deben ser integrales, equilibrados y reflejar la demografía real. Para Amazon: balancear activamente el dataset con currículums de mujeres contratadas con éxito, o usar muestreo estratificado por género. 
- **Detección continua, no solo al final.** Auditorías, pruebas de impacto y monitoreo antes de desplegar. Ningún sistema está "terminado". Herramientas de fairness miden paridad de falsos positivos/negativos por grupo. 
Human-in-the-loop. Que un reclutador revise las recomendaciones antes de descartar, como de hecho hacía Amazon parcialmente, pero formalizado.
- **Transparencia e interpretabilidad.** Documentar qué datos se usaron, qué proxies se eliminaron y por qué el modelo puntúa como puntúa. Si no puedes explicar por qué "women's chess club" baja puntos, no deberías usarlo. 
- **Gobernanza.** Definir métricas de equidad antes de optimizar precisión, y aceptar que a veces hay que sacrificar un poco de performance global para no dañar a un grupo. 

La lección de Amazon no es que la IA sea mala para contratar, sino que aprende exactamente lo que le enseñamos. Si le enseñamos una década de contrataciones sesgadas, nos devolverá ese sesgo, pero más rápido y a escala.

## Data Leakage: 
> ¿Qué es el data leakage (fuga de datos) en Machine Learning y por qué es uno de los errores más comunes y peligrosos al construir modelos? Explica la diferencia entre fuga de datos en el preprocesamiento (como escalar con datos de prueba) y fuga de datos conceptual (como incluir variables que en producción no estarían disponibles), y cómo prevenirlos.

La fuga de datos es cuando tu modelo aprende con información que, en el mundo real, no tendría en el momento de predecir. No es un bug de código, es una trampa metodológica: en entrenamiento ves el futuro, entonces tus métricas se ven espectaculares, pero en producción el modelo se cae.

Por eso es tan común y tan peligroso. Es común porque en un notebook todo está en una sola tabla y es tentador hacer ```df.fillna()```, scaler.```fit_transform(df)``` o ```selectKBest()``` antes de separar train y test. Es peligroso porque no lo notas hasta que está en producción: el AUC pasa de 0.95 en validación a 0.62 en vivo, y ya tomaste decisiones con él. Te da confianza falsa.

Hay dos familias distintas, y se previenen de forma distinta.

### 1. Fuga en el preprocesamiento
Aquí la información del set de prueba se cuela durante la preparación, no durante el modelado.

El clásico: escalar con datos de prueba.

``` Python
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)  # mal: fit con TODO
X_train, X_test = train_test_split(X_scaled)
```
Al hacer ```fit``` en todo X, la media y desviación de test influyen en train. Tu modelo ya sabe indirectamente cómo se distribuye el futuro.

Otros ejemplos igual de frecuentes:

- Imputar nulos con la media global antes del split
- Hacer target encoding (media de y por categoría) usando todo el dataset
- Aplicar SMOTE o cualquier oversampling antes de separar
- Seleccionar variables con correlación contra y usando todo el dataset

El modelo no ve la etiqueta directamente, pero ve estadísticas que incluyen el test. Resultado: sobreajuste invisible.

### 2. Fuga conceptual o temporal
Aquí el problema no es el código, es el diseño de features. Incluyes una variable que en producción no existiría en el momento de la predicción.

Ejemplos típicos:

- Predecir si un cliente hará churn este mes, y usar como feature "cantidad de tickets de soporte en los últimos 30 días" incluyendo los del día 30, cuando la predicción se hace el día 1
- Predecir impago de crédito usando "días de atraso máximo", que solo existe después del impago
- Diagnóstico médico usando resultados de laboratorio que se piden porque ya sospechas la enfermedad
- En series de tiempo, usar el precio de cierre de mañana para predecir hoy

Es una fuga de lógica de negocio. No importa qué tan perfecto sea tu pipeline, el modelo está aprendiendo la respuesta disfrazada de feature.

Cómo distinguirlos rápido
|Tipo| De dónde viene la info| Síntoma|
|---|---|---|
|Preprocesamiento| Del set de test/validación durante fit| Métricas demasiado buenas en CV, caen poco en test pero mucho en producción|
Conceptual| Del futuro o de una consecuencia del target| Métricas imposiblemente buenas incluso con modelos simples, features con importancia sospechosa|

### Cómo prevenirlos sin volverte loco
No necesitas más datos, necesitas disciplina de orden.

#### Para fuga de preprocesamiento
1. **Separa primero, siempre.** Haz train_test_split o tu split temporal antes de tocar cualquier transformador.
2. Usa **Pipelines**. En sklearn:
    ```Python
    pipe = Pipeline([('imputer', SimpleImputer()), ('scaler', StandardScaler()), ('model', LogisticRegression())])
    cross_val_score(pipe, X_train, y_train)
    ```
    El fit solo ve train en cada fold.
3. **Para validación cruzada, anida.** Si haces selección de features o tuning, que esté dentro del CV, no fuera.
4. **Nunca toques el test.** Guárdalo como si fuera producción. Solo transform, nunca fit ni fit_transform.

#### Para fuga conceptual
1. **Define el momento de predicción.** Escribe literal: "El modelo predice el martes a las 8am usando solo datos hasta el lunes 23:59". Todo feature debe poder calcularse con esa regla.
2. **Haz un inventario de disponibilidad.** Para cada columna pregunta: ¿esta información existe en ese momento para un cliente nuevo? Si la respuesta es "después", fuera.
3. **Usa splits temporales reales.** No random split si hay tiempo. Entrena con 2023, valida con 2024 Q1, test con 2024 Q2.
4. **Revisa con alguien de negocio.** Tú ves correlación 0.9, ellos ven "claro, eso es el resultado del evento". Es la revisión más barata.
5. **Simula producción.** Crea un script que toma un solo registro nuevo y pasa por el pipeline. Si falla porque falta una columna, tenías fuga.

La regla de oro que me repito: si una feature te hace pensar "wow, con esto es obvio", probablemente es fuga. Los problemas reales casi nunca son obvios.

## MLOps y el Modelo en Producción: 
> ¿Qué es MLOps y por qué es fundamental para llevar un modelo de Machine Learning del entorno de investigación a producción? Explica los conceptos de data drift, model drift y model monitoring, y describe qué estrategias se utilizan en la industria para detectar y responder cuando un modelo comienza a degradarse en producción

### MLOps y el Modelo en Producción
Pasar de un notebook que funciona en tu laptop a un modelo que decide precios, aprueba créditos o detecta fallas 24/7 no es solo "copiar el.pkl". Ahí entra MLOps.

#### ¿Qué es MLOps y por qué es fundamental?
**MLOps** es el paradigma que combina machine learning, DevOps e ingeniería de datos para implementar y mantener modelos en producción de manera confiable y eficiente. No es una herramienta, es una cultura de ingeniería que cierra la brecha entre el desarrollo (Dev) y las operaciones (Ops). 

Incluye:

- automatización CI/CD para código, datos y modelo
- orquestación del flujo de trabajo
- control de versiones de datos, modelo y código
- entrenamiento y evaluación continua, seguimiento de metadatos, monitoreo continuo y circuitos de retroalimentación 

¿Por qué sin esto los modelos mueren en el laboratorio?

- **Los modelos no son estáticos.** Los datos del mundo real cambian cada día, y un modelo entrenado con datos históricos se estanca rápido. 
- **La complejidad es operativa, no algorítmica.** Hasta el 88% de las iniciativas corporativas de ML no superan la fase de prueba, pero las que llegan a producción con MLOps ven aumentos de margen del 3-15%. 
- **Regulación y confianza.** MLOps aporta reproducibilidad, gobernanza y trazabilidad, lo que permite auditar por qué un modelo decidió algo, no solo que funcionó una vez.

En resumen: investigación crea precisión en un dataset; MLOps crea precisión sostenida en el mundo.

#### Data drift, model drift y model monitoring
Son tres piezas del mismo problema: el entorno cambia.

1. **Data drift (deriva de datos)**
   
   También llamado cambio de covariable. Ocurre cuando cambia la distribución subyacente de los datos de entrada. Ejemplo: tu modelo de e-commerce se entrenó con usuarios de 18-25 años, pero ahora lo usan mayores de 50; o un sensor que antes reportaba en Celsius ahora manda Fahrenheit. 

2. **Model drift / concept drift (deriva del modelo)**
   
   Es la degradación del rendimiento porque cambian las relaciones entre entrada y salida. IBM lo divide en:

   - **Conceptual:** la definición de "éxito" cambia. Estacional (ventas de quitanieves en invierno), repentino (pandemia, boom de ChatGPT), o gradual (spammers que evolucionan). 
   - **Data drift:** ya explicado.
   - **Cambio upstream:** alguien cambia la pipeline, por ejemplo dólares en vez de euros. 
  
3. Model monitoring
    
    Es la práctica de observar continuamente no solo la métrica de negocio (accuracy, AUC), sino también la calidad de los datos, la infraestructura y las predicciones. Es el corazón de la observabilidad de datos, y sin él la deriva avanza silenciosamente hasta causar decisiones erróneas. 

#### Estrategias que usa la industria para detectar y responder
No hay una sola bala de plata; las empresas maduras combinan detección estadística, automatización y cultura.

**Detección automática**

- Pruebas de distribución: la industria usa métodos no paramétricos porque no asumen forma de los datos:
  - **Kolmogorov-Smirnov (K-S):** compara si dos muestras vienen de la misma distribución. 
  - **Distancia de Wasserstein (Earth Mover's Distance):** mide "cuánto trabajo" cuesta transformar la distribución de entrenamiento en la de producción, útil con outliers. 
  - **PSI (Population Stability Index):** estándar en banca para variables categóricas; PSI alto = recalibrar. 
- **Monitoreo en tiempo real:** plataformas como Vertex AI o Watson comparan datos de entrenamiento vs producción continuamente y disparan alertas cuando la precisión cae bajo un umbral. 
- **Monitoreo más allá del accuracy:** se vigila latencia, tasa de errores, sesgo, y "data quality" (valores nulos, rangos).

**Respuesta orquestada**

1. **Alertas con contexto.** No solo "el modelo bajó", sino rastrear qué transacciones causaron la deriva para re-etiquetarlas y usarlas en reentrenamiento. 
2. **CI/CD de modelos.** Automatizar pruebas en preproducción (sesgo, deriva) y promover la misma configuración a producción. Azure ML lo implementa con pipelines reproducibles y registro de modelos. 
3. **Reentrenamiento continuo.**
   - *Batch retraining:* semanal/mensual con datos nuevos.
   - *Online learning:* actualizar el modelo con cada lote de datos reales en cuanto llegan. 
4. **Estrategias de despliegue seguro:**
   - *Shadow mode:* el nuevo modelo predice en paralelo sin afectar usuarios.
   - *Canary:* 5% del tráfico va al nuevo modelo; si las métricas aguantan, se escala.
5. **Entorno unificado.** Gestionar todo desde un panel central rompe silos y permite ver linaje completo de datos, modelo y métricas. Forrester destaca que esto es clave para mantener modelos justos y conformes. 
6. **Análisis de causa raíz.** Revisar si la deriva fue gradual o repentina con series temporales, y aplicar IA explicable para entender por qué cambió. 
7. **Verificación de inputs.** A veces no es deriva, es que entrenaste con escáneres de alta resolución y en producción solo hay baja resolución. 

En la práctica, un pipeline MLOps típico en 2026 luce así: datos ingestados → validación de esquema y drift (K-S/PSI) → entrenamiento versionado en MLflow → pruebas automáticas → despliegue canary → monitor en producción que compara predicciones vs realidad → si drift > umbral, dispara reentrenamiento y notifica al equipo.

Llevar un modelo a producción sin MLOps es como lanzar un satélite sin telemetría: funciona al inicio, pero no sabrás cuándo ni por qué se desvía. Con MLOps, la deriva deja de ser sorpresa y se convierte en señal para mejorar.

## Reflexión
Me sorprendió la capacidad de buscar entre artículos de diferentes diarios para brindarme un caso con el que estoy más familiarizado por el tamaño de la empresa y su contexto.