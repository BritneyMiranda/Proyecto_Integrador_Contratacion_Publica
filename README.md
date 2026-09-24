# Proyecto Integrador — RAG normativo y radar de contratación pública

Dos tareas conectadas sobre contratación pública en Perú:

- **[Tarea 1 — RAG normativo](tarea1_rag_normativo/README.md)** ("¿Qué dice
  la ley?"): asistente RAG sobre la Ley N.° 32069 y el Decreto Supremo N.°
  001-2026-EF, con motor desacoplado de la interfaz, abstención calibrada,
  gestión de versiones entre normas y costo observable. **Completa
  (Fases 1-5), verificada con corridas reales.**
- **[Tarea 2 — Radar de contratación pública](tarea2_radar/README.md)**
  ("¿Qué compra el Estado y dónde?"): datos abiertos de la OECE, reutilizando
  el motor RAG de la Tarea 1. **Completa (Fases 1-5)** — 20,424 procesos
  reales de 3 meses de 2026 (vía una API no documentada, descubierta
  inspeccionando el bundle de la SPA de OECE), validados, normalizados a
  los 25 departamentos del Perú, indexados y consultables con un motor RAG
  híbrido (filtros estructurados + búsqueda semántica, citas por `ocid`,
  umbral recalibrado con evidencia), con panel Streamlit (mapa
  coroplético, KPIs, preguntas, tabla descargable, calidad de datos, e
  indicador de riesgo de adjudicaciones a un solo licitador — explícitamente
  encuadrado como señal de alerta, no como prueba, según OCP y Ojo Público).

Video de presentación (≤12 min, ambas tareas): _pendiente — se agrega el
enlace aquí una vez grabado_.

## Estructura del repositorio

```
README.md                  # este archivo
requirements.txt           # dependencias de ambas tareas
.env.example                # solo nombres de variables — sin credenciales
.gitignore
docs/
  pipeline.md                # diagramas Mermaid de ambas tareas (offline/online)
tarea1_rag_normativo/       # ver su propio README.md — completa
  config.yaml                 # toda la configuración: rutas, modelo, umbral, prompts, documentos fuente
  build_index.py               # proceso fuera de línea: PDF -> índice vectorial
  app.py                        # Streamlit — interfaz, solo llama al motor
  src/                          # extracción, limpieza, chunking, embeddings, índice, motor, costos
    interfaces/cli.py             # interfaz de terminal — solo llama a engine.motor.responder()
  eval/                         # conjunto de evaluación + scripts (recall@k, comparación de embeddings)
  data/raw/                     # PDF fuente
  data/processed/               # texto procesado
  data/outputs/                 # informes de calidad (Fase 1) y evaluación (Fase 4)
  logs/costos.csv                # registro de costo real por llamada al LLM
tarea2_radar/                # ver su propio README.md — Fases 1-5 completas
  config.yaml                    # toda la configuración: rutas, modelo, umbral, filtros, riesgo, prompts
  app.py                          # Streamlit: KPIs, mapa, preguntar, tabla, riesgo, calidad — solo lee precalculados
  src/
    adquisicion.py               # descarga mensual OECE: caché, reintento, rate-limit, log real
    procesar.py                  # OCDS -> una fila por proceso (release vs. record, deduplicado)
    validar.py                   # 5 reglas de calidad + normalización a los 25 departamentos
    indexar.py                    # índice vectorial (reusa embeddings/vectorstore de Tarea 1)
    filtros.py                    # separa condiciones numéricas/territoriales de la búsqueda semántica
    motor.py                      # RAG híbrido — reusa ErrorDeAPI/costos/pricing de Tarea 1
    calibracion.py                 # barrido de umbral (reusa barrer_umbrales de Tarea 1)
    preparar_geojson.py            # fusiona/normaliza polígonos departamentales (geoBoundaries)
    riesgo.py                      # Fase 5: indicador "adjudicado a un solo licitador" (señal, no prueba)
  eval/                          # preguntas.csv (ocid conocido), evaluar.py (recall@k)
  data/raw/                      # 3 ZIP mensuales + polígonos geoBoundaries
  data/processed/                # procesos_validados.parquet/.csv, departamentos.geojson
  data/outputs/                  # calidad de Fases 1-2, evaluación de Fase 3, riesgo de Fase 5
  data/index/chroma_storage/      # índice vectorial (20,424 procesos; no va a git)
  logs/{adquisicion,costos}.csv    # registros reales de cada corrida
```

## Cómo correr cada tarea

Cada carpeta de tarea es autocontenida (su propio entorno virtual, su propio
`.env`, su propio `config.yaml`) — ver el README de cada una para los pasos
exactos:

- **Tarea 1:** [`tarea1_rag_normativo/README.md`](tarea1_rag_normativo/README.md)
  — instalación, cómo conseguir los 2 PDF, cómo construir el índice, cómo
  correr la app de Streamlit, tablas de resultados de las 5 fases.
- **Tarea 2:** [`tarea2_radar/README.md`](tarea2_radar/README.md) — Fases
  1-5 completas (adquisición, validación, RAG híbrido, panel Streamlit,
  indicador de riesgo); instrucciones para reproducir cada una ahí mismo.

`requirements.txt` y `.env.example` están en la raíz porque cubren ambas
tareas (evita duplicar/desincronizar versiones de dependencias); cada tarea
instala su propio `.venv` apuntando a este mismo archivo (`-r ..\requirements.txt`).

## Diagramas de flujo

Ver [`docs/pipeline.md`](docs/pipeline.md) — el mismo diagrama de la Tarea 1
también está embebido en `tarea1_rag_normativo/README.md`.

## Nota sobre el historial de commits

Los commits de este repositorio se hacen de forma incremental a medida que
avanza cada fase, no todos de una vez al final.
