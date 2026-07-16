"""
rag.py — Capa de recuperación de información (RAG) con LangChain.

Responsabilidades:
  - Cargar los documentos de cada dominio (hr / tech / finance).
  - Dividirlos en chunks (RecursiveCharacterTextSplitter).
  - Generar embeddings y construir un vector store FAISS por dominio.
  - Persistir el índice en disco para no recalcular embeddings en cada
    arranque (se invalida automáticamente si cambian los documentos fuente).
  - Exponer un retriever por dominio, listo para usar en las chains RAG.
"""

from __future__ import annotations

import hashlib
import logging
from pathlib import Path

from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.vectorstores import VectorStoreRetriever

from src.config import (
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    DOMAIN_PATHS,
    EMBEDDING_MODEL,
    RETRIEVER_TOP_K,
    VECTORSTORE_DIR,
)

logger = logging.getLogger(__name__)


def _domain_fingerprint(domain_path: Path) -> str:
    """Hash del contenido de un dominio, para saber si el índice persistido
    sigue siendo válido o si los documentos fuente cambiaron."""
    h = hashlib.sha256()
    for path in sorted(domain_path.glob("*.md")):
        h.update(path.name.encode("utf-8"))
        h.update(path.read_bytes())
    return h.hexdigest()[:16]


def _load_and_split(domain_path: Path) -> list:
    """Carga todos los .md de un dominio y los divide en chunks."""
    if not domain_path.exists():
        raise FileNotFoundError(
            f"No existe la carpeta de documentos para el dominio: {domain_path}"
        )

    loader = DirectoryLoader(
        str(domain_path),
        glob="*.md",
        loader_cls=TextLoader,
        loader_kwargs={"encoding": "utf-8"},
        show_progress=False,
    )
    raw_docs = loader.load()

    if not raw_docs:
        raise ValueError(f"El dominio {domain_path} no tiene documentos .md")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )
    chunks = splitter.split_documents(raw_docs)

    # Metadata útil para trazabilidad en Langfuse y para citar fuentes
    domain_name = domain_path.name.replace("_docs", "")
    for chunk in chunks:
        chunk.metadata["domain"] = domain_name
        chunk.metadata["source_file"] = Path(chunk.metadata.get("source", "")).name

    return chunks


def build_or_load_vectorstore(domain: str, embeddings: OpenAIEmbeddings) -> FAISS:
    """Construye el índice FAISS de un dominio, o lo carga desde disco si ya
    existe y los documentos fuente no cambiaron."""
    domain_path = DOMAIN_PATHS[domain]
    fingerprint = _domain_fingerprint(domain_path)
    index_dir = VECTORSTORE_DIR / f"{domain}_{fingerprint}"

    if index_dir.exists():
        logger.info("Cargando índice FAISS existente para dominio '%s'", domain)
        return FAISS.load_local(
            str(index_dir), embeddings, allow_dangerous_deserialization=True
        )

    logger.info("Construyendo índice FAISS para dominio '%s' (nuevo o desactualizado)", domain)
    chunks = _load_and_split(domain_path)
    vectorstore = FAISS.from_documents(chunks, embeddings)

    index_dir.parent.mkdir(parents=True, exist_ok=True)
    vectorstore.save_local(str(index_dir))
    return vectorstore


class RAGRegistry:
    """Punto único de acceso a los retrievers de cada dominio.

    Se instancia una sola vez (ver agents.py) y expone `get_retriever(domain)`
    para que cada agente recupere contexto de su propia base documental.
    """

    def __init__(self) -> None:
        self.embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL)
        self._vectorstores: dict[str, FAISS] = {}
        self._retrievers: dict[str, VectorStoreRetriever] = {}

    def _ensure_domain(self, domain: str) -> None:
        if domain not in self._vectorstores:
            self._vectorstores[domain] = build_or_load_vectorstore(domain, self.embeddings)
            self._retrievers[domain] = self._vectorstores[domain].as_retriever(
                search_kwargs={"k": RETRIEVER_TOP_K}
            )

    def get_retriever(self, domain: str) -> VectorStoreRetriever:
        if domain not in DOMAIN_PATHS:
            raise ValueError(f"Dominio desconocido: {domain}")
        self._ensure_domain(domain)
        return self._retrievers[domain]

    def build_all(self) -> None:
        """Fuerza la construcción/carga de los índices de todos los dominios
        (útil para precalentar el sistema al arrancar la aplicación)."""
        for domain in DOMAIN_PATHS:
            self._ensure_domain(domain)
