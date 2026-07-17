"""
stress_test.py — Test de consistencia: corre una misma consulta (el "golden
case" flaky, por defecto la de vacaciones/6 años) muchas veces, intercalada
con otras consultas del golden set (test_queries.json), para medir qué tan
estable es la respuesta del agente ante el mismo input repetido.

Uso:
    python -m src.stress_test
    python -m src.stress_test --repeats 20
    python -m src.stress_test --target "otra consulta" --repeats 10
    python -m src.stress_test --no-interleave   # solo repite el target, sin intercalar
"""

from __future__ import annotations

import argparse
import itertools
import json
import logging
from datetime import datetime, timezone
from pathlib import Path

from src.config import BASE_DIR, TEST_QUERIES_PATH
from src.main import run_query

logging.basicConfig(level=logging.WARNING, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

DEFAULT_TARGET = "¿Cuántos días de vacaciones me corresponden con 6 años de antigüedad?"
RESULTS_DIR = BASE_DIR / "stress_test_results"


def _load_golden_set() -> list[dict]:
    return json.loads(TEST_QUERIES_PATH.read_text(encoding="utf-8"))


def _heuristic_check(query: str, answer: str) -> str | None:
    """Chequeo determinístico auxiliar para el caso conocido de vacaciones,
    independiente del juez LLM (sirve para contrastar contra el evaluador).
    Devuelve 'ok', 'fail' o None si no aplica ningún chequeo conocido."""
    if "6 años de antigüedad" in query and "vacaciones" in query.lower():
        if "21 día" in answer or "21 dias" in answer:
            return "ok"
        if "14 día" in answer or "14 dias" in answer:
            return "fail"
    return None


def run_stress_test(target_query: str, repeats: int, interleave: bool) -> list[dict]:
    golden_set = _load_golden_set()
    distractors = [case["query"] for case in golden_set if case["query"] != target_query]
    distractor_cycle = itertools.cycle(distractors) if distractors else None

    records: list[dict] = []

    for i in range(1, repeats + 1):
        print(f"\n=== Repetición {i}/{repeats} — target ===")
        result = run_query(target_query)
        evaluation = result.get("evaluation") or {}
        heuristic = _heuristic_check(target_query, result.get("answer", ""))

        record = {
            "iteration": i,
            "role": "target",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "query": target_query,
            "intent": result.get("intent"),
            "answer": result.get("answer"),
            "correctness": evaluation.get("correctness"),
            "clarity": evaluation.get("clarity"),
            "grounding": evaluation.get("grounding"),
            "overall_score": evaluation.get("overall_score"),
            "evaluator_comment": evaluation.get("comment"),
            "heuristic_check": heuristic,
        }
        records.append(record)

        status = heuristic or ("eval_ok" if evaluation.get("correctness") == 1.0 else "eval_fail")
        print(f"[{status}] {result.get('answer')}")

        if interleave and distractor_cycle is not None:
            distractor_query = next(distractor_cycle)
            print(f"--- Repetición {i}/{repeats} — distractor ---")
            d_result = run_query(distractor_query)
            d_evaluation = d_result.get("evaluation") or {}
            records.append(
                {
                    "iteration": i,
                    "role": "distractor",
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "query": distractor_query,
                    "intent": d_result.get("intent"),
                    "answer": d_result.get("answer"),
                    "correctness": d_evaluation.get("correctness"),
                    "clarity": d_evaluation.get("clarity"),
                    "grounding": d_evaluation.get("grounding"),
                    "overall_score": d_evaluation.get("overall_score"),
                    "evaluator_comment": d_evaluation.get("comment"),
                    "heuristic_check": None,
                }
            )

    return records


def _summarize(records: list[dict], target_query: str) -> None:
    target_records = [r for r in records if r["role"] == "target"]
    total = len(target_records)

    heuristic_ok = sum(1 for r in target_records if r["heuristic_check"] == "ok")
    heuristic_fail = sum(1 for r in target_records if r["heuristic_check"] == "fail")
    heuristic_na = total - heuristic_ok - heuristic_fail

    eval_scores = [r["correctness"] for r in target_records if r["correctness"] is not None]
    avg_correctness = sum(eval_scores) / len(eval_scores) if eval_scores else None

    print("\n" + "=" * 70)
    print(f"Resumen de consistencia para: {target_query!r}")
    print(f"Repeticiones: {total}")
    if heuristic_ok + heuristic_fail > 0:
        print(
            f"Chequeo heurístico (21 días vs 14 días): "
            f"{heuristic_ok} OK / {heuristic_fail} FAIL / {heuristic_na} sin chequeo aplicable "
            f"({heuristic_ok / total:.0%} de consistencia)"
        )
    if avg_correctness is not None:
        print(f"Promedio de 'correctness' del evaluador: {avg_correctness:.2f}")
    print("=" * 70)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Corre una consulta N veces (intercalada con el golden set) para medir consistencia."
    )
    parser.add_argument("--target", default=DEFAULT_TARGET, help="Consulta a repetir y analizar")
    parser.add_argument("--repeats", type=int, default=20, help="Cantidad de repeticiones del target (default: 20)")
    parser.add_argument(
        "--no-interleave",
        action="store_true",
        help="No intercalar otras consultas del golden set entre cada repetición del target",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Ruta del archivo de resultados (default: stress_test_results/<timestamp>.json)",
    )
    args = parser.parse_args()

    records = run_stress_test(
        target_query=args.target,
        repeats=args.repeats,
        interleave=not args.no_interleave,
    )

    _summarize(records, args.target)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    if args.output:
        output_path = Path(args.output)
    else:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = RESULTS_DIR / f"stress_test_{stamp}.json"
    output_path.write_text(json.dumps(records, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nResultados guardados en: {output_path}")


if __name__ == "__main__":
    main()