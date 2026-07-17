"""
main.py — Punto de entrada por línea de comandos del sistema multiagente.

Uso:
    python -m src.main "¿Cuántos días de vacaciones me corresponden con 6 años de antigüedad?"
    python -m src.main --validate   # corre test_queries.json contra el grafo
"""

from __future__ import annotations

import argparse
import json
import logging
import sys

from src.config import TEST_QUERIES_PATH
from src.graph import compiled_graph
from src.langfuse_setup import get_langfuse_handler
from src.evaluator import evaluate_result

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)


def run_query(query: str) -> dict:
    """Ejecuta una consulta a través del grafo completo. El config con el
    callback de Langfuse se arma una sola vez acá y LangGraph lo reenvía a
    cada nodo (que a su vez lo reenvía a sus llamadas internas), lo que
    permite trazar tanto la capa de routing (orchestrator, route_by_intent)
    como las llamadas internas de LangChain. Al final, dispara el agente
    evaluador (bonus) para puntuar la respuesta y anclar el score a la
    misma traza."""
    handler = get_langfuse_handler()
    config = {"callbacks": [handler]} if handler else {}

    result = compiled_graph.invoke({"query": query}, config=config)

    if handler is not None:
        trace_id = handler.get_trace_id()

        # El nodo "unknown" devuelve un string fijo del código, no una
        # respuesta generada por el LLM: no tiene sentido pedirle a un
        # juez que evalúe "corrección" o "fundamentación" de algo que no
        # fue generado por un modelo.
        if result.get("intent") != "unknown":
            evaluation = evaluate_result(result, trace_id=trace_id)
            result["evaluation"] = evaluation

        # El SDK de Langfuse envía las trazas de forma asíncrona en background.
        # En un script de vida corta (como este CLI) el proceso puede terminar
        # antes de que el batch se envíe, perdiendo la traza. Forzamos el flush
        # para asegurar que quede registrada antes de salir.
        handler.flush()

    return result


def _print_result(result: dict) -> None:
    print("-" * 70)
    print(f"Consulta:  {result.get('query')}")
    print(f"Intención: {result.get('intent')}  ({result.get('reason', '')})")
    print(f"Fuentes:   {', '.join(result.get('sources', [])) or '(ninguna)'}")
    print(f"Respuesta: {result.get('answer')}")

    evaluation = result.get("evaluation")
    if evaluation:
        print(
            f"Evaluación: correctness={evaluation.get('correctness')}  "
            f"clarity={evaluation.get('clarity')}  "
            f"grounding={evaluation.get('grounding')}  "
            f"overall={evaluation.get('overall_score')}"
        )
        print(f"Comentario del evaluador: {evaluation.get('comment')}")
    print("-" * 70)


def validate_against_test_queries() -> None:
    """Corre todas las consultas de test_queries.json contra el grafo,
    reporta si la intención detectada coincide con la esperada, y agrega
    un resumen de los scores del evaluador (correctness/clarity/grounding)
    promediados sobre todas las consultas."""
    if not TEST_QUERIES_PATH.exists():
        logger.error("No se encontró %s", TEST_QUERIES_PATH)
        sys.exit(1)

    test_cases = json.loads(TEST_QUERIES_PATH.read_text(encoding="utf-8"))
    total = len(test_cases)
    correct = 0
    scores = {"correctness": [], "clarity": [], "grounding": [], "overall_score": []}

    for case in test_cases:
        result = run_query(case["query"])
        expected = case["expected_intent"]
        got = result.get("intent")
        ok = expected == got
        correct += int(ok)

        status = "OK " if ok else "FAIL"
        print(f"[{status}] esperado={expected:<8} obtenido={got:<8} | {case['query']}")

        evaluation = result.get("evaluation") or {}
        for key in scores:
            value = evaluation.get(key)
            if value is not None:
                scores[key].append(value)

        if evaluation:
            print(
                f"       -> correctness={evaluation.get('correctness')} "
                f"clarity={evaluation.get('clarity')} "
                f"grounding={evaluation.get('grounding')}"
            )
            print(f"       -> comentario: {evaluation.get('comment')}")

    print()
    print(f"Routing:   {correct}/{total} consultas correctamente enrutadas "
          f"({correct / total:.0%})")

    if scores["overall_score"]:
        print("Calidad (promedio del evaluador, sobre las consultas evaluadas):")
        for key, values in scores.items():
            if values:
                print(f"  {key:<15} {sum(values) / len(values):.2f}  (n={len(values)})")
    else:
        print("Calidad: sin datos del evaluador (Langfuse no está configurado).")


def main() -> None:
    parser = argparse.ArgumentParser(description="Sistema multiagente AEM3/PIM3 (i2T)")
    parser.add_argument("query", nargs="?", help="Consulta en lenguaje natural")
    parser.add_argument(
        "--validate",
        action="store_true",
        help="Corre test_queries.json contra el grafo y reporta precisión de routing",
    )
    args = parser.parse_args()

    if args.validate:
        validate_against_test_queries()
        return

    if not args.query:
        parser.error("Debés indicar una consulta, o usar --validate")

    result = run_query(args.query)
    _print_result(result)


if __name__ == "__main__":
    main()