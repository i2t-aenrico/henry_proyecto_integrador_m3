"""
graph.py — Orquestación del flujo con LangGraph.

Reemplaza un router manual (if/else) por un grafo declarativo: el nodo
`orchestrator` clasifica la intención y `add_conditional_edges` deriva la
ejecución al agente de dominio correspondiente (o al fallback `unknown`).
Esto deja el flujo de decisiones visible, testeable y fácil de extender
(por ejemplo, agregando un nuevo dominio solo agrega un nodo y una rama).
"""

from __future__ import annotations

from langgraph.graph import END, StateGraph

from src.agents import (
    AgentState,
    finance_agent_node,
    hr_agent_node,
    orchestrator_node,
    route_by_intent,
    tech_agent_node,
    unknown_node,
)


def build_graph():
    """Construye y compila el StateGraph del sistema multiagente."""
    graph = StateGraph(AgentState)

    graph.add_node("orchestrator", orchestrator_node)
    graph.add_node("hr", hr_agent_node)
    graph.add_node("tech", tech_agent_node)
    graph.add_node("finance", finance_agent_node)
    graph.add_node("unknown", unknown_node)

    graph.set_entry_point("orchestrator")

    # Routing condicional: la función route_by_intent lee state["intent"]
    # (ya calculado por orchestrator_node) y decide la rama a seguir.
    graph.add_conditional_edges(
        "orchestrator",
        route_by_intent,
        {
            "hr": "hr",
            "tech": "tech",
            "finance": "finance",
            "unknown": "unknown",
        },
    )

    # Todos los agentes de dominio terminan la ejecución del grafo.
    graph.add_edge("hr", END)
    graph.add_edge("tech", END)
    graph.add_edge("finance", END)
    graph.add_edge("unknown", END)

    return graph.compile()


# Instancia compilada, lista para invocar desde main.py o desde el notebook.
compiled_graph = build_graph()
