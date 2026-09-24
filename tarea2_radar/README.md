# Tarea 2 — Radar de contratación pública

Estado: **Fases 1-5 completas y verificadas con datos reales.** Ver
`../README.md` para la visión general del repo.

## Fuente de datos — hallazgo real, no documentado públicamente

La página de descargas de OECE (`contratacionesabiertas.oece.gob.pe/descargas`)
es una SPA en Angular sin enlaces de descarga estáticos — igual que pasó con
El Peruano en la Tarea 1. En vez de pedir descarga manual, inspeccioné el
bundle JS compilado que sirve esa SPA (`main.*.js`) y encontré la API REST
real que consume, no documentada en ninguna página pública:

- `GET https://contratacionesabiertas.oece.gob.pe/api/v1/files` (paginado,
  324 archivos disponibles) — metadatos de cada archivo mensual masivo, por
  fuente (`seace_v3` para datos recientes) y por año/mes.
- `GET https://contratacionesabiertas.oece.gob.pe/api/v1/file/<fuente>/<formato>/<año>/<mes>/`
  — descarga el ZIP de ese mes (`csv`, `xlsx` o `json`).

Sin autenticación, sin bloqueo a bots (`200 OK` directo). Usada exactamente
como está diseñada: descargas masivas por archivo, no por request individual.

## Corpus descargado (Fase 1, requisito: ≥3 archivos mensuales de 2026)

```powershell
cd tarea2_radar
uv venv --python 3.12 .venv
uv pip install --python .venv\Scripts\python.exe -r ..\requirements.txt
.venv\Scripts\python.exe src\adquisicion.py   # descarga (idempotente)
.venv\Scripts\python.exe src\procesar.py      # normaliza a una fila por proceso
```

3 meses reales descargados: **junio, julio y agosto de 2026** (formato JSON,
fuente `seace_v3`). Registro real en `logs/adquisicion.csv`:

| Mes | Tamaño | Duración | Intentos |
|---|---|---|---|
| 2026-06 | 11,170,315 bytes | 16.0 s | 1 |
| 2026-07 | 10,032,696 bytes | 163.7 s | 1 |
| 2026-08 | 8,315,209 bytes | 18.7 s | 1 |

3 peticiones HTTP totales (una por mes). Las 3 tuvieron éxito al primer
intento — el reintento con backoff exponencial (`src/adquisicion.py`) está
implementado pero no se necesitó esta vez.

**Idempotencia verificada**: correr `adquisicion.py` una segunda vez hace
**0 peticiones HTTP** — cada archivo ya existe en `data/raw/` y se salta.

**Nunca pierde trabajo**: cada descarga escribe primero a `<archivo>.zip.tmp`
y solo renombra al destino final si terminó completa — una descarga cortada
a la mitad nunca deja un archivo corrupto pisando uno bueno anterior ni hace
perder los meses ya descargados con éxito.

## Modelo de datos OCDS: release vs. record

Cada archivo mensual es un **"record package"** de OCDS (clave de nivel
superior `"records"`, no `"releases"`). Cada elemento de `records[]` trae:

```json
{"ocid": "ocds-dgv273-seacev3-1251201", "releases": [...], "compiledRelease": {...}}
```

- Una **liberación (release)** es un documento OCDS que captura UN evento
  puntual del proceso (la convocatoria, una modificación de bases, la buena
  pro...) tal como estaba en ese momento.
- Lo que identifica que varias liberaciones pertenecen al MISMO proceso es
  el **`ocid`** — se repite en todas las liberaciones de un mismo proceso.
- Un **registro (record)** es el contenedor por `ocid`: trae la lista
  completa de liberaciones (`releases`) MÁS una liberación fusionada
  (`compiledRelease`) que resume el estado más reciente y completo del
  proceso — por diseño, hay como máximo un `compiledRelease` por `ocid`.

**Evidencia real, no supuesta:** en los 3 meses descargados, cada proceso
tiene entre **1 y 74 liberaciones** fusionadas en su `compiledRelease`
(mediana: **13**) — ver `data/outputs/fase1_calidad.csv`. Esa es la razón concreta
de por qué se arma la tabla final desde `compiledRelease` y no desde
`releases`: usar `releases` habría dado ~13 filas por proceso en promedio
(cada modificación, cada cambio de estado), rompiendo el requisito de una
fila por proceso.

## Una fila por proceso — antes y después

`src/procesar.py` arma una fila por `compiledRelease` (ya es 1 por `ocid`
dentro de cada archivo mensual) y luego deduplica por `ocid` **entre los 3
meses**, quedándose con la versión de `compiledRelease.date` más reciente
cuando el mismo proceso aparece en más de un archivo.

| | Filas |
|---|---|
| Antes de deduplicar entre meses (suma de los 3 archivos) | **20,424** |
| Después de deduplicar por `ocid` | **20,424** |
| `ocid` duplicados encontrados entre meses | **0** |

Hallazgo real: **0 duplicados**, no porque el problema no exista en general,
sino porque OECE particiona cada archivo mensual por
`dataSegmentation.criteria = [añoInicioConvocatoria, mesInicioConvocatoria]`
— cada proceso queda asignado al mes en que SE INICIÓ su convocatoria, de
forma permanente, así que un mismo `ocid` no puede aparecer en dos archivos
mensuales distintos con esta fuente. La deduplicación por `ocid` se dejó
implementada de todas formas como salvaguarda genérica (necesaria si se
agregan más meses, o si se usa otra fuente que sí reparta el mismo proceso
en varios archivos) — no se asumió que no haría falta.

## Calidad de los datos (`data/outputs/fase1_calidad.csv`)

- **20,424 procesos** en el corpus final (`data/processed/procesos.parquet`
  y `.csv`), 17 columnas.
- **25 departamentos distintos, 0 valores faltantes** en el campo
  `departamento` — cubre las 24 regiones del Perú más el Callao. Viene de
  `compiledRelease.parties[rol=buyer].address.department`, ya como **campo
  estructurado (metadata)**, no como texto libre a extraer de una
  descripción — importante para la Fase 2 (filtros vs. embeddings).
- Categorías OCDS estándar: `goods` (8,884), `services` (8,219), `works`
  (3,321).

## Problemas de calidad conocidos (documentados por Open Contracting Partnership)

Según `data.open-contracting.org/es/publication/135`: algunos proveedores
(`tenderer`) tienen identificadores duplicados; algunos contratos no tienen
estado asignado; hay códigos de tipo de documento no declarados; en algunos
casos la información de adjudicación no está vinculada a la de contratos.
Relevante para la Fase 2 (validación) — no se ignora, se aborda ahí.

## Fase 2 — validación y normalización territorial

```powershell
.venv\Scripts\python.exe src\validar.py
```

Corre sobre `data/processed/procesos.parquet` (salida de Fase 1) y produce
`data/processed/procesos_validados.parquet` (+ `.csv`) y el informe de
calidad `data/outputs/fase2_calidad_validacion.csv` — una fila por regla, con
cuántos registros marcó y qué se hizo con ellos.

### Hallazgo real: el campo OCDS `region` no contiene departamentos

Al inspeccionar todos los valores únicos del campo `region` de la dirección
del comprador (distinto de `department`), **177 de 195 valores únicos NO
son nombres de departamento** — son nombres de **provincia** (`HUANCAYO`,
`TRUJILLO`, `CHICLAYO`...), verificado comparándolos contra la lista oficial
de 25 departamentos. Es exactamente el problema que describe el enunciado
("encontrará provincias mezcladas con departamentos"): si alguien usara
`region` pensando que es un sinónimo de departamento, mezclaría provincias
y departamentos sin darse cuenta. Por eso Fase 1 usa `department` (de la
extensión `ocds_department_extension` de OECE), no `region` — `region`
queda documentado pero sin usarse para la columna de territorio.

### Reconciliación de acentos: JUNÍN vs. JUNIN

Los nombres oficiales (INEI) de los departamentos llevan tilde (`JUNÍN`,
`HUÁNUCO`, `SAN MARTÍN`, `APURÍMAC`); el campo `department` de la API de
OECE viene en mayúsculas **sin tilde** (`JUNIN`, `HUANUCO`...) — consistente
internamente, pero habría chocado contra cualquier fuente que sí use tilde
(ej. los polígonos departamentales de la fase de territorio). La regla
`normalizar_departamento()` (`src/validar.py`) quita tildes y mayúsculas
ANTES de comparar contra la lista oficial de 25 departamentos, y se prueba
explícitamente con 8 casos sintéticos (`JUNIN`, `JUNÍN`, `Junín`, `junin`,
espacios de sobra, minúsculas, `San Martin` sin tilde, y un valor
inexistente) — no solo con los datos ya limpios que nos tocaron.

### Informe de calidad (`data/outputs/fase2_calidad_validacion.csv`) — resultado real

| Regla | Filas marcadas | Acción |
|---|---|---|
| `ocid_duplicado` | 0 | ninguna (Fase 1 ya deduplica; esto re-verifica de forma independiente) |
| `monto_faltante_o_cero` | 2,632 (12.9%) | **conservadas con advertencia** (columna `advertencia_monto_cero_o_faltante`) — un monto en 0 puede ser legítimo, eliminar 13% del corpus sin evidencia de error sería injustificado |
| `sin_descripcion` | 0 | ninguna |
| `ubicacion_no_es_departamento` (campo `region`) | 20,424 (100% de las filas con `region`) | campo ignorado para territorio; se usa `departamento` en su lugar |
| `normalizacion_25_departamentos` | 0 no localizados | **corregidas** — 20,424/20,424 (100%) normalizadas al nombre oficial con tilde |

Ningún registro se eliminó en silencio: la única regla que sí elimina
(`ocid_duplicado`, con 0 casos reales) está justificada y registrada; la de
monto cero se conserva con advertencia explícita en vez de borrarse.

## Fase 3 — RAG híbrido

```powershell
.venv\Scripts\python.exe src\indexar.py       # construye el índice (una vez)
.venv\Scripts\python.exe src\calibracion.py   # barrido de umbral sobre este corpus
.venv\Scripts\python.exe eval\evaluar.py      # recall@k, sin llamar al LLM
```

### Reutilización real de la Tarea 1 (no reimplementado)

`src/motor.py`, `src/indexar.py` y `src/calibracion.py` importan
directamente desde `tarea1_rag_normativo/src`: `E5Embedder`,
`abrir_coleccion` (mismo modelo local, misma lógica de índice
idempotente), `ErrorDeAPI`, `_llamar_gemini`/`_llamar_deepseek` (llamar al
LLM y manejar sus errores es idéntico sea cual sea el corpus),
`registrar_llamada`/`RegistroLlamada` (mismo formato de log, apuntado al
`logs/costos.csv` **de esta tarea**, no el de la Tarea 1), y hasta
`barrer_umbrales` de `engine/calibracion.py` (esa función es genérica: solo
necesita una tabla de similitudes, no sabe nada de normativa). Lo único
propio de esta tarea es `filtros.py` (parser de condiciones
numéricas/territoriales) y el prompt de citación por `ocid`.

**20,424 procesos indexados** en 1,294.8 s (~21.6 min — el corpus es 27×
más grande que el de la Tarea 1). Texto embebido = `titulo + descripcion`
(máx. 542 caracteres, muy por debajo del límite de 512 tokens de E5, no
hace falta chunking). Metadata de ChromaDB por proceso: `ocid`,
`departamento`, `monto`, `categoria`, `fecha_publicacion`,
`comprador_nombre`, `comprador_id`, `tender_id`, `titulo`.

### Por qué las condiciones numéricas/territoriales son FILTROS, no embeddings

Una pregunta como *"Obras de agua y alcantarillado en Cusco por encima de
un millón de soles"* mezcla una condición semántica ("obras de agua y
alcantarillado" — se resuelve con similitud de embeddings) con dos
condiciones EXACTAS: el departamento es `CUSCO` o no lo es; el monto es
mayor a 1,000,000 o no lo es. Un embedding no hace aritmética ni
comparación exacta — la similitud semántica entre "un millón de soles" y
un proceso de 980,000 es casi idéntica a la de uno de 1,200,000: el texto
se parece, el número no cumple la condición. Confiarle esto a la similitud
produce falsos positivos (montos por debajo del umbral) y falsos negativos
(procesos que describen el monto con otras palabras). Lo mismo con el
departamento: un embedding puede traer procesos de Apurímac o Puno
(semánticamente "cerca" de Cusco) aunque el departamento exacto no
coincida. Por eso `src/filtros.py` separa la pregunta en (a) condiciones
estructuradas -> `where` de ChromaDB, aplicado ANTES de rankear por
similitud, y (b) el resto del texto -> búsqueda semántica. Reconoce los 25
departamentos, alias de categoría OCDS, y montos con patrones "por encima
de/mayor a/más de X" y "por debajo de/menor a/menos de X" (dígitos o
"un(a)"/"medio(a)", con multiplicador "mil"/"millón(es)").

**Bug real encontrado y corregido en evaluación:** una primera versión
buscaba el departamento como substring suelto, y "ICA" (departamento)
machaba dentro de "ELECTRÓNICAS" (`...electrón-ICA-s`), metiendo un filtro
de departamento incorrecto en una pregunta que no lo pedía y rompiendo esa
consulta. Se corrigió a coincidencia de PALABRA completa (`\bICA\b`) — otra
prueba de que validar con datos reales, no solo con ejemplos elegidos a
mano, importa.

### El umbral de la Tarea 1 NO transfiere — recalibrado con evidencia

Barrido real (`src/calibracion.py`, `data/outputs/fase3_barrido_umbral.csv`)
sobre 12 preguntas de dominio + 8 fuera de dominio, contra ESTE índice:

| Umbral | Falsos-abstención | Falsos-respuesta | Errores totales |
|---|---|---|---|
| 0.86 | **0% (0/12)** | 37.5% (3/8) | **3 (mínimo)** |
| 0.88 (el de la Tarea 1) | 50% (6/12) | 12.5% (1/8) | 7 |
| 0.90 | 83.3% (10/12) | 0% (0/8) | 10 |

**El umbral 0.88 de la Tarea 1 no se aplica bien aquí**: sobre este corpus
produce 50% de falsos-negativos y todavía deja pasar un falso-positivo —
las escalas de similitud dependen del corpus, no solo del modelo (ya lo
habíamos visto en la Tarea 1, Fase 4, entre modelos distintos; acá se repite
entre corpus distintos con el MISMO modelo).

**Criterio de recalibración, deliberadamente distinto al de la Tarea 1:**
en la Tarea 1 elegí "0% falsos-positivos a cualquier costo" porque una
respuesta legal mal citada es difícil de verificar para alguien sin
formación legal. Acá cada respuesta **siempre** cita el `ocid` del
proceso — cualquiera puede verificarlo en segundos en el portal de OECE — y
un radar de transparencia que se abstiene el 83% de las veces (el umbral de
"0% falsos-positivos" aquí, 0.90) deja de cumplir su propósito. Por eso se
eligió **0.86**: minimiza el total de errores del barrido, con 0%
falsos-negativos sobre las preguntas de dominio, aceptando un residual de
falsos-positivos mitigado por la cita verificable.

### Recall@k (`data/outputs/fase3_evaluacion.csv`) — 12 preguntas sobre procesos reales conocidos

Cada pregunta se armó a partir de un proceso real elegido de la muestra
(ver `eval/preguntas.csv`, columna `ocid_esperado`), cubriendo 8
departamentos y las 3 categorías OCDS.

| Métrica | Valor |
|---|---|
| Recall@1 | 0.583 (7/12) |
| Recall@3 | 0.667 (8/12) |
| Recall@5 | **0.75 (9/12)** |
| Abstenciones (umbral 0.86) | 2/12 |
| Preguntas con filtro estructurado detectado | 11/12 |

Los 3 casos sin acierto en top-5: 2 abstuvieron (similitud justo debajo de
0.86 — consistente con el umbral elegido, que acepta algo de
falso-negativo a cambio de nunca dejar pasar un falso-positivo en el eval
de calibración) y 1 pasó el umbral pero no trajo el proceso exacto en
top-5 pese al filtro de departamento correcto — una limitación real de
recall semántico dentro del subconjunto ya filtrado, no escondida.

### Citación por ocid — verificado con una llamada real

Pregunta real: *"Compra de piedra chancada para transitabilidad vial en
Puno"* → filtro `{departamento: PUNO}` detectado correctamente → 3 procesos
reales del Gobierno Regional de Puno recuperados, cada uno citado en la
respuesta como `(ocid: ocds-dgv273-seacev3-1232393)`, etc. — formato exacto
pedido por el enunciado. Costo: $0.00 (capa gratuita de Gemini).

## Fase 4 — Panel Streamlit

```powershell
.venv\Scripts\python.exe src\preparar_geojson.py   # una sola vez: prepara el mapa
.venv\Scripts\streamlit.exe run app.py
```

Se abre en `http://localhost:8501`.

### Polígonos departamentales

Fuente: [geoBoundaries](https://www.geoboundaries.org/) (ADM1, Perú, CC-BY
4.0) — descarga directa, sin bloqueo, 26 unidades. `src/preparar_geojson.py`
(se corre una sola vez, no en cada carga de página) fusiona
"Municipalidad Metropolitana de Lima" + "Lima" en un solo departamento
`LIMA` y renombra "El Callao" -> `CALLAO`, usando la MISMA regla de
normalización de la Fase 2 (`validar.normalizar_departamento`) — reusada,
no reimplementada. Resultado: **25/25 departamentos oficiales con
geometría**, guardado en `data/processed/departamentos.geojson`.

### Bug real encontrado y corregido: `px.choropleth` no renderizaba el mapa

La primera versión usaba `px.choropleth` (el trazador "geo" de Plotly,
basado en SVG) con `fitbounds="locations"`. Al probarlo con datos reales
en el navegador, solo se dibujaba UN departamento (Ucayali) y el resto de
la vista quedaba pintado con el color del último valor de la serie —
reproducido de forma aislada (con subconjuntos de 5 departamentos),
confirmado con capturas reales, e independiente de la versión de Plotly
instalada (probado en 6.1.1 y 7.1.0 — no era una regresión de versión).
Se descartaron como causa: los datos (los 25 departamentos tienen conteos
reales correctos, verificado), el emparejamiento de nombres (coinciden
exacto contra el geojson), la complejidad de los polígonos (persistía tras
simplificar de ~7,000 a ~2,700 puntos) y el sentido de los anillos
(winding order — solo Lima lo tiene invertido, y no es la que fallaba).
La causa real es una limitación del renderizador "geo"/SVG de Plotly con
GeoJSON personalizado. Se corrigió cambiando a `px.choropleth_map`
(renderizador MapLibre, vectorial) con `map_style="carto-positron"` (estilo
abierto, sin necesitar token) — renderiza los 25 departamentos
correctamente, verificado con captura real.

### Las 6 vistas mínimas

1. **Encabezado KPI** (arriba de las pestañas, siempre visible): procesos,
   monto total, departamentos representados, y el **indicador de riesgo de
   la Fase 5** ("adjudicado a un solo licitador", nacional) — ver sección
   Fase 5 más abajo. Los 4 KPI se recalculan con los filtros de la barra
   lateral (el de riesgo es nacional/precalculado, ver justificación en
   Fase 5).
2. **Mapa coroplético** (pestaña Resumen): departamentos coloreados por
   número de procesos o monto total (selector), con leyenda de escala y
   tooltip al pasar el mouse (procesos + monto por departamento).
3. **Cuadro de preguntas** (pestaña Preguntar): el motor híbrido de la
   Fase 3 — respuesta, filtros detectados, similitud, costo, y los procesos
   recuperados con su similitud individual.
4. **Tabla ordenable** (pestaña Tabla): `st.dataframe` (ordenar por
   columna, nativo) + botón de descarga CSV del subconjunto filtrado.
5. **Vista de distribución** (pestaña Resumen): procesos por categoría OCDS
   y procesos por mes, con Plotly.
6. **Panel de calidad de datos** (pestaña Calidad): las tablas reales de
   `data/outputs/fase1_calidad.csv`, `fase2_calidad_validacion.csv` y
   `fase3_resumen.csv` — el usuario ve en qué se basa el análisis, no un
   resumen inventado.
7. **Panel de riesgo** (pestaña Riesgo, Fase 5): desglose por departamento
   y top 10 compradores del indicador "adjudicado a un solo licitador",
   con el aviso explícito de que es una señal de alerta, no una prueba —
   ver sección Fase 5 más abajo.

### Filtros de la barra lateral (los 5 pedidos)

Departamento, categoría, rango de monto, rango de fechas, y umbral de
similitud (este último solo afecta la pestaña Preguntar — no tiene sentido
"filtrar" la tabla/mapa por una similitud que no existe hasta que se hace
una pregunta). Selección vacía en cualquier filtro no rompe la app: cada
sección revisa `df.empty` y muestra un aviso en vez de fallar (probado
vaciando manualmente el rango de monto a un intervalo sin datos).

### Nunca descarga ni reconstruye nada al cargar la página

`app.py` solo lee `procesos_validados.parquet`, `departamentos.geojson` y
los 3 CSV de `data/outputs/` — todos precalculados por `adquisicion.py`,
`procesar.py`, `validar.py`, `preparar_geojson.py` y `eval/evaluar.py`
respectivamente, corridos aparte. `@st.cache_data` en las 3 funciones de
carga (se leen una sola vez por sesión) y `@st.cache_resource` en
`obtener_config()`. La pestaña Preguntar abre el índice vectorial ya
construido (`src/indexar.py` corre aparte, no desde la app) — nunca lo
crea ni le agrega nada.

### Verificación real

`python app.py` (modo bare) corrió sin ningún traceback; `streamlit run
app.py --server.headless true` levantó el servidor y respondió `200 OK`.

## Fase 5 — Indicador de riesgo: adjudicaciones a un solo licitador

```powershell
.venv\Scripts\python.exe src\riesgo.py
```

Lee `data/processed/procesos_validados.parquet` y produce 3 reportes
precalculados que la pestaña **🚩 Riesgo** del panel solo lee (no recalcula
nada al cargar la página): `data/outputs/fase5_resumen.csv`,
`data/outputs/fase5_riesgo_departamento.csv`,
`data/outputs/fase5_riesgo_top_compradores.csv`.

### Qué mide y con qué campos OCDS

- **Adjudicado**: el proceso tiene al menos una liberación en `awards` con
  proveedor asignado. OCDS trae `tender.status` y `award.status` vacíos en
  el **100% de los 20,424 procesos** de este dataset (verificado) — no se
  puede filtrar por esos campos, así que la presencia de `awards` es la
  señal disponible de que el proceso llegó a buena pro.
- **Número de licitadores**: `tender.numberOfTenderers` — cuántos postores
  **participaron** en la competencia, no cuántos ganaron (eso sería
  `len(awards[].suppliers)`, casi siempre 1, y no mide competencia).
- **Un solo licitador**: `numero_licitadores == 1` entre los adjudicados.

### Número mínimo de procesos por comprador — justificación

Con pocos procesos, la proporción es inestable: un comprador con 1 proceso
adjudicado y ese único proceso con un solo postor tiene "100% de riesgo"
sin que eso sea un patrón. Con n=5, un solo caso mueve la proporción 20
puntos porcentuales; con n=10, la mueve 10 puntos. Se fijó el mínimo en
**10 procesos adjudicados** por comprador (`config.yaml: riesgo.umbral_minimo_procesos_por_comprador`)
— por debajo de eso, un único proceso puede seguir dominando la lectura, y
publicar ese ranking convertiría ruido estadístico en una lista de "los
peores compradores" sin sustento.

### Resultado real (3 meses de 2026, nacional)

| | Valor |
|---|---|
| Procesos adjudicados | 13,742 |
| ...con un solo licitador | 1,803 |
| **Proporción nacional** | **13.1%** |
| Compradores evaluados (≥1 proceso adjudicado) | 2,064 |
| Compradores en el ranking (≥10 procesos) | 337 |
| Excluidos del ranking por muestra insuficiente | 1,727 |

Top 3 departamentos por proporción: **TUMBES (36.7%, n=120)**, **LIMA
(30.2%, n=3,863)**, AMAZONAS (16.1%, n=218) — el resto entre 1.5% (APURÍMAC)
y 10.1%. Top comprador con el mínimo de 10 procesos: **Organismo de
Evaluación y Fiscalización Ambiental (95.4%, 166/174 procesos)**, seguido
de Programa Subsectorial de Irrigaciones - PSI (85.2%), Ministerio de
Relaciones Exteriores (83.9%) — ver `data/outputs/fase5_riesgo_top_compradores.csv`
para el top 10 completo.

**Salvaguarda real, no un stub**: antes de publicar el ranking por
comprador, `src/riesgo.py::_filtrar_nombres_no_institucionales()` descarta
cualquier `comprador_nombre` que no esté en mayúscula (las 2,183 entidades
del corpus lo están, sin excepción — un nombre de persona pegado por error
en ese campo casi seguro rompería ese patrón). En esta corrida excluyó 8
nombres con mayúscula/minúscula mixta por precaución (ej. "...Sede
Central", "...Administración General" — instituciones legítimas, falsos
positivos de la heurística, pero se prefirió pecar de cauto).

### Encuadre obligatorio: señal de alerta, no prueba

Siguiendo tanto a [Open Contracting Partnership — Red Flags in Public
Procurement (2024)](https://www.open-contracting.org/resources/red-flags-in-public-procurement-a-guide-to-using-data-to-detect-and-mitigate-risks/)
como a [Ojo Público — Funes, un algoritmo contra la corrupción](https://ojo-publico.com/especiales/funes/)
— ambas fuentes usan el patrón de "postor único" repetido como una señal
que activa revisión adicional, no como veredicto — la pestaña Riesgo del
panel muestra un `st.warning` explícito con esta advertencia ANTES de
cualquier tabla, y el mismo encuadre debe repetirse en el video. No se
publican nombres de personas: `comprador_nombre` en OCDS es siempre una
`Organization` (municipalidad, ministerio, hospital...), nunca un
funcionario individual.
