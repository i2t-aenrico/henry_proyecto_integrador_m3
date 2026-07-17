"""
langfuse_setup.py — Observabilidad de las ejecuciones del sistema con Langfuse.

Expone:
  - get_langfuse_handler(): CallbackHandler para pasar a las chains/LLM calls
    de LangChain y que cada paso quede trazado automáticamente.
  - score_trace(): helper para el evaluador (bonus) que puntúa una traza ya
    ejecutada usando la Score API de Langfuse.
"""

from __future__ import annotations

import logging

from src.config import (
    LANGFUSE_ENABLED,
    LANGFUSE_HOST,
    LANGFUSE_PUBLIC_KEY,
    LANGFUSE_SECRET_KEY,
)

logger = logging.getLogger(__name__)

_handler = None
_langfuse_client = None


def get_langfuse_handler():
    """Devuelve un CallbackHandler de Langfuse listo para usar, o None si
    Langfuse no está configurado (permite correr el sistema sin observabilidad
    durante el desarrollo local)."""
    global _handler

    if not LANGFUSE_ENABLED:
        logger.warning(
            "Langfuse no está configurado (faltan LANGFUSE_PUBLIC_KEY / "
            "LANGFUSE_SECRET_KEY). El sistema funcionará sin tracing."
        )
        return None

    if _handler is None:
        from langfuse.callback import CallbackHandler

        _handler = CallbackHandler(
            public_key=LANGFUSE_PUBLIC_KEY,
            secret_key=LANGFUSE_SECRET_KEY,
            host=LANGFUSE_HOST,
            # debug=True,  # TEMPORAL: para diagnosticar por qué no llegan trazas
        )
    return _handler


def get_langfuse_client():
    """Cliente de Langfuse para operaciones que no son de callback, como el
    envío de scores (Score API) usado por el evaluador (bonus)."""
    global _langfuse_client

    if not LANGFUSE_ENABLED:
        return None

    if _langfuse_client is None:
        from langfuse import Langfuse

        _langfuse_client = Langfuse(
            public_key=LANGFUSE_PUBLIC_KEY,
            secret_key=LANGFUSE_SECRET_KEY,
            host=LANGFUSE_HOST,
        )
    return _langfuse_client


def score_trace(trace_id: str, name: str, value: float, comment: str | None = None) -> None:
    """Envía un puntaje asociado a una traza ya ejecutada (Score API).

    Se usa desde src/evaluator.py (bonus) para calificar automáticamente
    la calidad de las respuestas del sistema.
    """
    client = get_langfuse_client()
    if client is None:
        logger.warning("Langfuse no configurado: se omite el score '%s'", name)
        return

    client.score(trace_id=trace_id, name=name, value=value, comment=comment)