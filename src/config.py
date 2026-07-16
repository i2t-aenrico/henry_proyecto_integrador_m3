"""
config.py — Configuración central del sistema multiagente AEM3/PIM3.

Centraliza rutas, variables de entorno y parámetros de los modelos,
para que el resto de los módulos (rag.py, agents.py, graph.py,
langfuse_setup.py) no dependan de detalles de infraestructura.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# ---------------------------------------------------------------------------
# Rutas del proyecto
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"

DOMAIN_PATHS = {
    "hr": DATA_DIR / "hr_docs",
    "tech": DATA_DIR / "tech_docs",
    "finance": DATA_DIR / "finance_docs",
}

TEST_QUERIES_PATH = BASE_DIR / "test_queries.json"

# Carpeta donde se persisten los índices FAISS por dominio (se generan en
# el primer arranque y se reutilizan en los siguientes para no recalcular
# embeddings innecesariamente).
VECTORSTORE_DIR = BASE_DIR / ".vectorstores"

# ---------------------------------------------------------------------------
# Modelos
# ---------------------------------------------------------------------------
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
LLM_MODEL = os.getenv("LLM_MODEL", "gpt-4o-mini")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")

# ---------------------------------------------------------------------------
# Langfuse
# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# Langfuse
# ---------------------------------------------------------------------------
LANGFUSE_PUBLIC_KEY = os.getenv("LANGFUSE_PUBLIC_KEY")
LANGFUSE_SECRET_KEY = os.getenv("LANGFUSE_SECRET_KEY")

# LANGFUSE_BASE_URL es el nombre que usa el propio dashboard de Langfuse
# (Project Settings > API Keys > .env) al generar el snippet para copiar.
# Se mantiene LANGFUSE_HOST como alias por compatibilidad, por si alguien
# ya lo configuró con ese nombre en versiones anteriores de este proyecto.
LANGFUSE_HOST = os.getenv("LANGFUSE_BASE_URL") or os.getenv(
    "LANGFUSE_HOST", "https://cloud.langfuse.com"
)
LANGFUSE_ENABLED = bool(LANGFUSE_PUBLIC_KEY and LANGFUSE_SECRET_KEY)

# ---------------------------------------------------------------------------
# Parámetros de chunking y recuperación
# ---------------------------------------------------------------------------
CHUNK_SIZE = 280
CHUNK_OVERLAP = 40
RETRIEVER_TOP_K = 4

# Dominios válidos que puede devolver el orquestador
VALID_INTENTS = ("hr", "tech", "finance", "unknown")
