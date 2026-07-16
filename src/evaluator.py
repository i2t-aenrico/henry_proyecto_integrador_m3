"""
evaluator.py — Agente evaluador (bonus).

Sigue el patrón: Agente Primario -> Respuesta Generada -> Evaluator Agent
(rúbrica: Corrección, Claridad, Grounding) -> Score -> Traza en Langfuse.

Un LLM secundario actúa como juez, puntúa la respuesta del agente primario
según una rúbrica de tres criterios, y el resultado se ancla a la traza
original mediante la Score API de Langfuse.
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
            "Sos un evaluador objetivo (Evaluator Agent) de un sistema de "
            "atención al cliente. Vas a recibir una consulta, el contexto "
            "recuperado de la base documental y la respuesta generada por "
            "el agente primario. Evaluá la respuesta según esta rúbrica, "
            "puntuando cada criterio en una escala de 0.0 a 1.0:\n\n"
            "- correctness (Corrección): ¿la respuesta es objetivamente "
            "correcta respecto de lo que dice el contexto? (1.0 = sin "
            "errores; 0.0 = afirma algo incorrecto o contradice el "
            "contexto).\n"
            "- clarity (Claridad): ¿la respuesta es clara, concisa y fácil "
            "de entender para el colaborador que preguntó? (1.0 = muy "
            "clara; 0.0 = confusa o mal redactada).\n"
            "- grounding (Fundamentación): ¿todo lo afirmado está "
            "respaldado por el contexto provisto, sin inventar información "
            "que no está ahí? (1.0 = totalmente fundamentada; 0.0 = "
            "inventa datos no presentes en el contexto).\n\n"
            "Respondé ÚNICAMENTE con un JSON válido, sin texto adicional, "
            'con este formato exacto: {{"correctness": <float>, "clarity": '
            '<float>, "grounding": <float>, "comment": "<justificación '
            'breve, una frase>"}}',
        ),
        (
            "human",
            "Consulta:\n{query}\n\nContexto:\n{context}\n\nRespuesta:\n{answer}",
        ),
    ]
)

_judge_chain = _JUDGE_PROMPT | _judge_llm | StrOutputParser()


def evaluate_result(result: dict, trace_id: str | None = None) -> dict:
    """Evalúa el resultado de un agente primario (dict con query/context/
    answer) según la rúbrica Corrección/Claridad/Grounding, calcula un score
    global (promedio de los tres) y, si se provee trace_id, ancla los cuatro
    valores a la traza en Langfuse vía Score API.

    Devuelve un dict:
        {
            "correctness": float,
            "clarity": float,
            "grounding": float,
            "overall_score": float,
            "comment": str,
        }
    """
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
        return {
            "correctness": None,
            "clarity": None,
            "grounding": None,
            "overall_score": None,
            "comment": raw,
        }

    correctness = parsed.get("correctness", 0.0)
    clarity = parsed.get("clarity", 0.0)
    grounding = parsed.get("grounding", 0.0)
    comment = parsed.get("comment", "")
    overall_score = round((correctness + clarity + grounding) / 3, 4)

    if trace_id:
        score_trace(trace_id, "correctness", correctness, comment)
        score_trace(trace_id, "clarity", clarity, comment)
        score_trace(trace_id, "grounding", grounding, comment)
        score_trace(trace_id, "overall_score", overall_score, comment)

    return {
        "correctness": correctness,
        "clarity": clarity,
        "grounding": grounding,
        "overall_score": overall_score,
        "comment": comment,
    }