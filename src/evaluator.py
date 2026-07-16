"""
evaluator.py — Evaluador automático de calidad (bonus).

Usa un LLM como juez para puntuar, en una escala 0-1, qué tan fundamentada
está la respuesta generada en el contexto recuperado (groundedness) y qué
tan relevante es respecto de la consulta original. Los puntajes se envían
a Langfuse mediante la Score API para quedar asociados a la traza de cada
ejecución.
"""

from __future__ import annotations

import json
import logging

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

from src.config import LLM_MODEL
from src.langfuse_setup import score_trace

logger = logging.getLogger(__name__)

_judge_llm = ChatOpenAI(model=LLM_MODEL, temperature=0)

_JUDGE_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "Sos un evaluador objetivo de un sistema de atención al cliente. "
            "Vas a recibir una consulta, el contexto recuperado de la base "
            "documental y la respuesta generada. Evaluá dos aspectos en una "
            "escala de 0.0 a 1.0:\n\n"
            "- groundedness: qué tan fundamentada está la respuesta en el "
            "contexto provisto (1.0 = todo lo afirmado está respaldado por "
            "el contexto; 0.0 = la respuesta inventa información no "
            "presente en el contexto).\n"
            "- relevance: qué tan bien responde la respuesta a la consulta "
            "del usuario (1.0 = responde completamente; 0.0 = no responde "
            "la consulta).\n\n"
            "Respondé ÚNICAMENTE con un JSON válido, sin texto adicional, "
            'con este formato: {{"groundedness": <float>, "relevance": '
            '<float>, "comment": "<justificación breve>"}}',
        ),
        (
            "human",
            "Consulta:\n{query}\n\nContexto:\n{context}\n\nRespuesta:\n{answer}",
        ),
    ]
)

_judge_chain = _JUDGE_PROMPT | _judge_llm | StrOutputParser()


def evaluate_result(result: dict, trace_id: str | None = None) -> dict:
    """Evalúa un resultado del grafo (dict con query/context/answer) y,
    si se provee trace_id, envía los scores a Langfuse."""
    raw = _judge_chain.invoke(
        {
            "query": result.get("query", ""),
            "context": result.get("context") or "(sin contexto: nodo unknown)",
            "answer": result.get("answer", ""),
        }
    )

    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        logger.warning("El juez no devolvió JSON válido, se omite el score: %s", raw)
        return {"groundedness": None, "relevance": None, "comment": raw}

    if trace_id:
        score_trace(trace_id, "groundedness", parsed.get("groundedness", 0.0), parsed.get("comment"))
        score_trace(trace_id, "relevance", parsed.get("relevance", 0.0), parsed.get("comment"))

    return parsed
