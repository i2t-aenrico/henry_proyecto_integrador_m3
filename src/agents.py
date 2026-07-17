"""
agents.py — Orquestador y agentes especializados (HR / Tech / Finance / Unknown).

Cada función de este módulo es un "nodo" de negocio que luego se conecta en
el grafo de LangGraph (ver graph.py). Mantenerlos acá, desacoplados del
grafo, permite testearlos de forma aislada (ver test_queries.json).

Los nodos reciben el `config` (con el callback de Langfuse) que LangGraph
les reenvía automáticamente desde compiled_graph.invoke(..., config=config)
en main.py, y lo forwardean a sus llamadas internas. Esto es lo que permite
que Langfuse trace tanto la capa de routing de LangGraph (orchestrator,
route_by_intent, __start__) como las llamadas internas de LangChain
(chains, retriever, LLM).
"""

from __future__ import annotations

import logging
from typing import Literal, TypedDict

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableConfig
from langchain_openai import ChatOpenAI

from src.config import LLM_MODEL, VALID_INTENTS
from src.rag import RAGRegistry

logger = logging.getLogger(__name__)

Intent = Literal["hr", "tech", "finance", "unknown"]


class AgentState(TypedDict, total=False):
    """Estado compartido que viaja por todos los nodos del grafo.

    - query: consulta original del usuario.
    - intent: dominio clasificado por el orquestador (hr/tech/finance/unknown).
    - reason: breve justificación de la clasificación (para depuración/trazas).
    - context: texto recuperado del vector store del dominio correspondiente.
    - sources: nombres de archivo de los chunks usados como contexto.
    - answer: respuesta final generada para el usuario.
    - trace_steps: lista de nodos ejecutados, para inspección y testing.
    """

    query: str
    intent: Intent
    reason: str
    context: str
    sources: list[str]
    answer: str
    trace_steps: list[str]


# ---------------------------------------------------------------------------
# Recursos compartidos (se instancian una sola vez por proceso)
# ---------------------------------------------------------------------------
_llm = ChatOpenAI(model=LLM_MODEL, temperature=0)
_rag_registry = RAGRegistry()


# ---------------------------------------------------------------------------
# Orquestador: clasificación de intención
# ---------------------------------------------------------------------------
_CLASSIFIER_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "Sos el orquestador de un sistema de atención al cliente de una "
            "empresa de software. Tu única tarea es clasificar la consulta "
            "del usuario en uno de estos cuatro dominios:\n\n"
            "- hr: vacaciones, licencias, beneficios, horarios, sueldo, "
            "clima laboral, capacitación, ingreso o baja de personal, "
            "documentación personal y legajo (DNI, CUIL, actas de "
            "matrimonio o nacimiento, constancias), estructura "
            "organizacional.\n"
            "- tech: VPN, contraseñas, autenticación de dos factores, "
            "soporte técnico, equipos, SuiteCRM, seguridad informática.\n"
            "- finance: reembolsos de gastos, viáticos, facturación a "
            "clientes, pagos a proveedores, presupuesto de proyectos.\n"
            "- unknown: cualquier consulta que no encaje claramente en los "
            "dominios anteriores, o que sea ambigua, o que no esté "
            "relacionada con la empresa.\n\n"
            "Respondé ÚNICAMENTE con una línea en este formato exacto:\n"
            "INTENT: <hr|tech|finance|unknown>\n"
            "REASON: <una frase breve que justifique la clasificación>",
        ),
        ("human", "{query}"),
    ]
)

_classifier_chain = _CLASSIFIER_PROMPT | _llm | StrOutputParser()


def _parse_classifier_output(raw: str) -> tuple[Intent, str]:
    intent: Intent = "unknown"
    reason = "No se pudo interpretar la clasificación del modelo."

    for line in raw.strip().splitlines():
        line = line.strip()
        if line.upper().startswith("INTENT:"):
            candidate = line.split(":", 1)[1].strip().lower()
            if candidate in VALID_INTENTS:
                intent = candidate  # type: ignore[assignment]
        elif line.upper().startswith("REASON:"):
            reason = line.split(":", 1)[1].strip()

    return intent, reason


def orchestrator_node(state: AgentState, config: RunnableConfig | None = None) -> AgentState:
    """Clasifica la intención de la consulta. No accede a ningún vector
    store: es una decisión puramente de enrutamiento."""
    raw = _classifier_chain.invoke({"query": state["query"]}, config=config)
    intent, reason = _parse_classifier_output(raw)

    logger.info("Orquestador clasificó '%s' como intent=%s", state["query"], intent)

    return {
        **state,
        "intent": intent,
        "reason": reason,
        "trace_steps": state.get("trace_steps", []) + ["orchestrator"],
    }


def route_by_intent(state: AgentState) -> Intent:
    """Función de routing condicional para LangGraph: lee el intent ya
    calculado por el orquestador y decide a qué nodo derivar."""
    return state.get("intent", "unknown")


# ---------------------------------------------------------------------------
# Agentes de dominio (RAG)
# ---------------------------------------------------------------------------
_DOMAIN_AGENT_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "Sos un agente de atención al cliente especializado en el área "
            "de {domain_label} de la empresa i2T. Respondé la consulta del "
            "colaborador basándote EXCLUSIVAMENTE en el siguiente contexto "
            "recuperado de la documentación interna. Si el contexto no "
            "alcanza para responder con certeza, decilo explícitamente en "
            "lugar de inventar información.\n\n"
            "Contexto:\n{context}",
        ),
        ("human", "{query}"),
    ]
)

_DOMAIN_LABELS = {
    "hr": "Recursos Humanos",
    "tech": "Tecnología / Soporte",
    "finance": "Finanzas",
}


def _make_domain_node(domain: Intent):
    """Fábrica de nodos de dominio: devuelve una función de nodo que
    recupera contexto del dominio indicado y genera la respuesta."""

    def _node(state: AgentState, config: RunnableConfig | None = None) -> AgentState:
        retriever = _rag_registry.get_retriever(domain)
        docs = retriever.invoke(state["query"], config=config)

        context = "\n\n---\n\n".join(d.page_content for d in docs)
        sources = sorted({d.metadata.get("source_file", "desconocido") for d in docs})

        chain = _DOMAIN_AGENT_PROMPT | _llm | StrOutputParser()
        answer = chain.invoke(
            {
                "domain_label": _DOMAIN_LABELS[domain],
                "context": context if context else "(sin contexto relevante encontrado)",
                "query": state["query"],
            },
            config=config,
        )

        return {
            **state,
            "context": context,
            "sources": sources,
            "answer": answer,
            "trace_steps": state.get("trace_steps", []) + [f"agent_{domain}"],
        }

    return _node


hr_agent_node = _make_domain_node("hr")
tech_agent_node = _make_domain_node("tech")
finance_agent_node = _make_domain_node("finance")


# ---------------------------------------------------------------------------
# Nodo Unknown (fallback): no inventa, deriva a soporte humano
# ---------------------------------------------------------------------------
def unknown_node(state: AgentState) -> AgentState:
    answer = (
        "No pude identificar con certeza a qué área corresponde tu consulta "
        "(RR. HH., Tecnología o Finanzas). Para no darte una respuesta "
        "incorrecta, te recomiendo reformular la consulta con más detalle, "
        "o escribir directamente a soporte@i2t.com.ar para que un humano te "
        "oriente."
    )
    return {
        **state,
        "context": "",
        "sources": [],
        "answer": answer,
        "trace_steps": state.get("trace_steps", []) + ["agent_unknown"],
    }