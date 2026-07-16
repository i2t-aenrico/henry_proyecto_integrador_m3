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

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)


def run_query(query: str) -> dict:
    """Ejecuta una consulta a través del grafo completo, con tracing en
    Langfuse si está configurado."""
    handler = get_langfuse_handler()
    config = {"callbacks": [handler]} if handler else {}

    result = compiled_graph.invoke({"query": query}, config=config)

    # El SDK de Langfuse envía las trazas de forma asíncrona en background.
    # En un script de vida corta (como este CLI) el proceso puede terminar
    # antes de que el batch se envíe, perdiendo la traza. Forzamos el flush
    # para asegurar que quede registrada antes de salir.
    if handler is not None:
        handler.flush()

    return result


def _print_result(result: dict) -> None:
    print("-" * 70)
    print(f"Consulta:  {result.get('query')}")
    print(f"Intención: {result.get('intent')}  ({result.get('reason', '')})")
    print(f"Fuentes:   {', '.join(result.get('sources', [])) or '(ninguna)'}")
    print(f"Respuesta: {result.get('answer')}")
    print("-" * 70)


def validate_against_test_queries() -> None:
    """Corre todas las consultas de test_queries.json contra el grafo y
    reporta si la intención detectada coincide con la esperada."""
    if not TEST_QUERIES_PATH.exists():
        logger.error("No se encontró %s", TEST_QUERIES_PATH)
        sys.exit(1)

    test_cases = json.loads(TEST_QUERIES_PATH.read_text(encoding="utf-8"))
    total = len(test_cases)
    correct = 0

    for case in test_cases:
        result = run_query(case["query"])
        expected = case["expected_intent"]
        got = result.get("intent")
        ok = expected == got
        correct += int(ok)

        status = "OK " if ok else "FAIL"
        print(f"[{status}] esperado={expected:<8} obtenido={got:<8} | {case['query']}")

    print()
    print(f"Resultado: {correct}/{total} consultas correctamente enrutadas "
          f"({correct / total:.0%})")


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
