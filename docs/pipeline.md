# Diagramas de flujo

Diagramas de referencia para el video de presentación. El mismo diagrama de Tarea 1 vive
también en `tarea1_rag_normativo/README.md`.

## Tarea 1 — RAG normativo

Dos procesos independientes: uno **fuera de línea** (se corre una vez, o
cuando cambian los documentos) y uno **en línea** (se corre en cada
pregunta y nunca vuelve a tocar los PDF).

```mermaid
flowchart TB
    subgraph OFFLINE["FUERA DE LÍNEA — build_index.py (una vez)"]
        A1["PDF crudos<br/>data/raw/*.pdf"] --> A2["Verificar calidad<br/>(páginas sin texto,<br/>orden de lectura)"]
        A2 --> A3["Limpiar encabezado/pie<br/>(regla estructural)"]
        A3 --> A4["Texto por página + metadata<br/>data/processed/*.jsonl<br/><b>número de página va aquí,<br/>como metadata — no como texto</b>"]
        A4 --> A5["Segmentar en fragmentos<br/>chunk_size=128, overlap=24"]
        A5 --> A6["Embeddings locales<br/>(E5, CPU)"]
        A6 --> A7[("Índice vectorial<br/>ChromaDB<br/>data/index/")]
    end

    subgraph ONLINE["EN LÍNEA — motor.responder(pregunta), cada consulta"]
        B1["Pregunta del usuario"] --> B2["Embeber la pregunta"]
        B2 --> B3["Buscar top-k en el índice<br/>(solo lee A7, nunca reconstruye)"]
        B3 --> B4{"¿Mejor similitud<br/>≥ umbral 0.88?"}
        B4 -->|"No"| B5["ABSTENERSE<br/><b>aquí el sistema decide<br/>NO llamar al LLM</b><br/>costo = $0.00"]
        B4 -->|"Sí"| B6["Armar prompt con fragmentos<br/>+ instrucciones de versión"]
        B6 --> B7["LLM (Gemini)<br/>genera la respuesta"]
        B7 --> B8["Respuesta + citas<br/>(documento, página, similitud)<br/>+ costo real"]
    end

    A7 -.->|"índice ya construido"| B3
```

## Tarea 2 — Radar de contratación pública

Mismo patrón que la Tarea 1: un proceso **fuera de línea** (adquisición ->
validación -> indexado -> indicador de riesgo, cada uno se corre aparte y
una sola vez) y uno **en línea** (el panel Streamlit, que solo LEE lo que
el proceso fuera de línea ya calculó).

```mermaid
flowchart TB
    subgraph OFFLINE["FUERA DE LÍNEA — se corre aparte, no al abrir el panel"]
        C1["API OECE (SEACE v3)<br/>no documentada, descubierta<br/>en el bundle JS de la SPA"] --> C2["adquisicion.py<br/>descarga mensual: caché,<br/>reintento, rate-limit"]
        C2 --> C3["procesar.py<br/>compiledRelease -> 1 fila/ocid<br/>(release vs. record)"]
        C3 --> C4["validar.py<br/>5 reglas de calidad +<br/>normalización a 25 departamentos<br/>(JUNÍN vs. JUNIN)"]
        C4 --> C5["indexar.py<br/>embeddings locales E5<br/>(mismo modelo que Tarea 1)"]
        C5 --> C6[("Índice vectorial<br/>ChromaDB<br/>20,424 procesos")]
        C4 --> C7["riesgo.py (Fase 5)<br/>adjudicado a un solo licitador<br/>por departamento y comprador"]
        C7 --> C8[("Reportes precalculados<br/>data/outputs/fase*.csv")]
        C4 --> C9["preparar_geojson.py<br/>polígonos de 25 departamentos"]
        C9 --> C10[("departamentos.geojson")]
    end

    subgraph ONLINE["EN LÍNEA — app.py (streamlit run), cada carga de página"]
        D1["Pregunta del usuario"] --> D2["filtros.py<br/>separa condiciones exactas<br/>(depto/monto/categoría)<br/>del resto del texto"]
        D2 --> D3["Buscar top-k en el índice<br/>(where=filtros + similitud)<br/>solo lee C6, nunca reconstruye"]
        D3 --> D4{"¿Mejor similitud<br/>≥ umbral 0.86?"}
        D4 -->|"No"| D5["ABSTENERSE<br/>costo = $0.00"]
        D4 -->|"Sí"| D6["LLM (Gemini)<br/>respuesta citada por ocid"]
        D7["Panel Streamlit<br/>KPIs, mapa, tabla,<br/>riesgo, calidad"] -.->|"solo lee"| C8
        D7 -.->|"solo lee"| C10
    end

    C6 -.->|"índice ya construido"| D3
```
