# AEM3 / PIM3 — Sistema Multiagente RAG con LangChain, LangGraph y Langfuse

Sistema de routing inteligente para una empresa SaaS (i2T): clasifica
automáticamente las consultas de clientes por departamento (RR. HH.,
Tecnología y Finanzas) y las deriva a agentes especializados que responden
con información fundamentada en documentación interna real, usando RAG.

## Arquitectura

```
Consulta del usuario
        |
        v
  Orquestador (LangChain, clasifica intención)
        |
        v
  Routing condicional (LangGraph, add_conditional_edges)
        |
   +----+---------+-----------+
   v    v          v           v
  HR   Tech     Finance     Unknown
 (RAG) (RAG)     (RAG)     (fallback)
        |
        v
  Respuesta fundamentada + traza en Langfuse
```

- **LangChain**: carga de documentos, chunking, embeddings, vector store
  (FAISS) y retrievers por dominio; prompts y chains de generación.
- **LangGraph**: `StateGraph` con `add_conditional_edges` para el routing
  entre el orquestador y los agentes de dominio.
- **Langfuse**: `CallbackHandler` para trazar cada ejecución end-to-end, y
  Score API para el evaluador automático (bonus).

## Estructura del repositorio

```
.
├── data/
│   ├── hr_docs/        # Documentos reales de i2T (RR. HH.)
│   ├── tech_docs/      # Documentos de Tecnología / Soporte
│   └── finance_docs/   # Documentos de Finanzas
├── src/
│   ├── config.py        # Rutas, variables de entorno, parámetros
│   ├── rag.py            # Chunking, embeddings, FAISS, retrievers
│   ├── agents.py         # Orquestador y agentes HR/Tech/Finance/Unknown
│   ├── graph.py          # StateGraph y routing condicional (LangGraph)
│   ├── langfuse_setup.py # CallbackHandler y Score API
│   ├── evaluator.py      # Evaluador automático (bonus, LLM-as-judge)
│   ├── main.py           # Punto de entrada por CLI
│   └── stress_test.py    # Test de consistencia (repite consultas y mide estabilidad)
├── notebooks/
│   └── multi_agent_system.ipynb
├── test_queries.json
├── stress_test_results/   # (generado, no versionado) resultados de src/stress_test.py
├── pyproject.toml
├── uv.lock
├── .python-version
├── requirements.txt      # alternativa para quienes prefieren pip + venv
└── .env.example
```

## Instalación

### Opción recomendada: uv

[`uv`](https://docs.astral.sh/uv/) resuelve e instala todas las dependencias
(incluidas las de Jupyter) en segundos, usando el lockfile `uv.lock` para
builds reproducibles.

```bash
# Instalar uv (una sola vez), si no lo tenés:
curl -LsSf https://astral.sh/uv/install.sh | sh

# Instalar dependencias del proyecto (crea .venv automáticamente)
uv sync --extra notebook

cp .env.example .env       # Completar OPENAI_API_KEY y, opcionalmente, Langfuse

# Ejecutar cualquier comando dentro del entorno del proyecto:
uv run python -m src.main "¿Cuántos días de vacaciones me corresponden con 6 años de antigüedad?"
uv run jupyter notebook notebooks/multi_agent_system.ipynb
```

> **Nota sobre la versión de Python:** el archivo `.python-version` fija
> Python 3.12, porque `faiss-cpu` (usado como vector store) todavía no
> publica wheels para Python 3.14. Si `uv` no tiene esa versión disponible,
> la descarga automáticamente al correr `uv sync` (no requiere instalarla
> manualmente).

### Alternativa: pip + venv

```bash
python -m venv venv
source venv/bin/activate  # En Windows: venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env
```

> **Nota sobre Langfuse:** al crear el proyecto en Langfuse Cloud elegís
> una región (US o EU), y las claves solo son válidas en esa región. Usá
> el valor de `LANGFUSE_BASE_URL` (o `LANGFUSE_HOST`, alias soportado por
> compatibilidad) tal cual te lo muestra el propio dashboard en
> *Project Settings > API Keys > pestaña .env* — copiarlo así evita
> errores silenciosos donde las trazas nunca llegan a aparecer.

## Ejecución

Con `uv` anteponé `uv run` a cualquier comando; con `pip + venv`, activá el
entorno virtual primero. A partir de acá los comandos son los mismos:

```bash
python -m src.main "¿Cuántos días de vacaciones me corresponden con 6 años de antigüedad?"
```

Validar el routing contra el set de pruebas (`test_queries.json`):

```bash
python -m src.main --validate
```

También puede ejecutarse el flujo completo desde
`notebooks/multi_agent_system.ipynb`.

## Ejemplos de uso

| Consulta | Intención esperada |
|---|---|
| "¿Cuántos días de vacaciones me corresponden con 6 años de antigüedad?" | hr |
| "No puedo conectarme a la VPN desde mi notebook" | tech |
| "¿Cuándo se paga el reembolso de una factura de viáticos ya aprobada?" | finance |
| "¿Cuál es la capital de Francia?" | unknown |

## Resultados de validación

Última corrida de `python -m src.main --validate` sobre las 12 consultas
de `test_queries.json`:

- **Routing:** 12/12 consultas correctamente enrutadas (100%).
- **Calidad de respuesta** (evaluador automático, sobre las 9 consultas de
  dominio real — se excluyen las 3 de control `unknown`): `correctness`,
  `clarity` y `grounding` en 1.00 de promedio.

## Evaluador automático (bonus)

`src/evaluator.py` implementa un **Evaluator Agent**: un LLM secundario que
actúa como juez y puntúa cada respuesta generada por un agente de dominio
(hr/tech/finance) según una rúbrica de tres criterios, en escala 0.0 a 1.0:

- **correctness** (Corrección): ¿la respuesta es objetivamente correcta
  respecto del contexto recuperado?
- **clarity** (Claridad): ¿la respuesta es clara y fácil de entender?
- **grounding** (Fundamentación): ¿todo lo afirmado está respaldado por el
  contexto, sin inventar información?

El promedio de los tres compone un `overall_score`. Los cuatro valores se
envían a Langfuse mediante la Score API y quedan anclados a la traza de la
ejecución (visibles en la pestaña **Scores** del dashboard).

El evaluador se dispara **automáticamente** después de cada consulta (en
`src/main.py`, tanto en una consulta suelta como en `--validate`), siempre
que Langfuse esté configurado. El nodo `unknown` queda excluido de la
evaluación, porque su respuesta es un mensaje fijo del código y no una
respuesta generada por el LLM.

## Limitaciones conocidas

- Los documentos de `tech_docs` y `finance_docs` son ficticios (creados con
  el mismo estilo y tono que los documentos reales de RR. HH. de i2T), ya
  que no se contó con documentación fuente real para esos dominios.
- El orquestador clasifica con un LLM mediante prompt (few-shot implícito
  en las instrucciones del sistema), no con un clasificador entrenado; en
  consultas muy ambiguas puede requerir ajuste de prompt.
- Los índices FAISS se persisten en `.vectorstores/` y se invalidan
  automáticamente por hash de contenido cuando cambian los documentos
  fuente de un dominio.
- No se implementó autenticación ni una interfaz web: el sistema se
  ejecuta por CLI o notebook, según el alcance del proyecto académico.
- El agente HR mostró una inconsistencia ocasional al leer tablas con
  rangos numéricos límite. Se midió con un test de estrés dedicado
  (`src/stress_test.py`, 20 repeticiones de la misma consulta intercaladas
  con otras del golden set): **95% de consistencia (19/20)** en calcular
  correctamente los días de vacaciones según antigüedad. Es un
  comportamiento conocido de los LLMs de OpenAI, que no garantizan
  determinismo total incluso con `temperature=0`. El agente evaluador
  (bonus) detectó correctamente el caso fallido, lo que confirma el valor
  de esa capa de control de calidad.
