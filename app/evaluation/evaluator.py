from collections.abc import Callable, Iterable
from typing import Any


def evaluate_retrieval(
    retrieved_results: list[dict[str, Any]],
    expected_source: str,
) -> dict[str, int]:
    """Calculate top-one, top-k, and source-match retrieval indicators."""
    sources = [str(result.get("source", "")) for result in retrieved_results]
    top_one_hit = int(bool(sources) and sources[0] == expected_source)
    top_k_hit = int(expected_source in sources)
    return {
        "hit": top_one_hit,
        "top_k_hit": top_k_hit,
        "source_match": top_k_hit,
    }


def keyword_coverage(answer: str, expected_keywords: Iterable[str]) -> float:
    """Return the fraction of expected keywords present in an answer."""
    keywords = [keyword.lower() for keyword in expected_keywords]
    if not keywords:
        return 0.0
    answer_lower = answer.lower()
    return sum(keyword in answer_lower for keyword in keywords) / len(keywords)


def answer_has_sources(answer: str, sources: list[dict[str, Any]]) -> bool:
    """Return whether at least one supplied source identifier appears in answer."""
    from app.observability.metrics import citation_count

    return citation_count(answer, sources) > 0


def evaluate_answer(
    answer: str,
    sources: list[dict[str, Any]],
    expected_keywords: Iterable[str],
) -> dict[str, float | bool]:
    """Calculate transparent keyword and source-presence indicators."""
    return {
        "keyword_coverage": keyword_coverage(answer, expected_keywords),
        "answer_has_sources": answer_has_sources(answer, sources),
    }


def _sources_from_result(result: Any) -> list[dict[str, Any]]:
    """Convert a ResearchResult into evaluator source records."""
    return [
        {
            "source_type": "local_document",
            "source": item.get("source", ""),
            "score": item.get("score", 0.0),
        }
        for item in result.local_results
    ] + [
        {
            "source_type": "web",
            "source": item.get("domain", ""),
            "title": item.get("title", ""),
            "url": item.get("url", ""),
        }
        for item in result.web_results
    ]


def run_evaluation(
    dataset: list[dict[str, object]],
    research_function: Callable[[str], Any],
) -> dict[str, object]:
    """Run the dataset through an existing research function and aggregate metrics."""
    cases: list[dict[str, object]] = []
    for case in dataset:
        result = research_function(str(case["question"]))
        retrieval = evaluate_retrieval(
            result.local_results,
            str(case["expected_source"]),
        )
        sources = _sources_from_result(result)
        answer = evaluate_answer(
            result.answer,
            sources,
            [str(keyword) for keyword in case["expected_keywords"]],
        )
        cases.append(
            {
                "question": case["question"],
                **retrieval,
                **answer,
                "latency_ms": float(getattr(result, "total_latency_ms", 0.0)),
                "retrieval_latency_ms": float(
                    getattr(result, "retrieval_latency_ms", 0.0)
                ),
                "generation_latency_ms": float(
                    getattr(result, "generation_latency_ms", 0.0)
                ),
            }
        )

    total = len(cases)
    if not total:
        return {
            "total_questions": 0,
            "retrieval_hit_rate": 0.0,
            "source_match_rate": 0.0,
            "keyword_coverage": 0.0,
            "average_latency_ms": 0.0,
            "average_retrieval_latency_ms": 0.0,
            "average_generation_latency_ms": 0.0,
            "cases": [],
        }

    return {
        "total_questions": total,
        "retrieval_hit_rate": sum(case["hit"] for case in cases) / total,
        "source_match_rate": sum(case["source_match"] for case in cases) / total,
        "keyword_coverage": sum(case["keyword_coverage"] for case in cases) / total,
        "average_latency_ms": sum(case["latency_ms"] for case in cases) / total,
        "average_retrieval_latency_ms": sum(
            case["retrieval_latency_ms"] for case in cases
        ) / total,
        "average_generation_latency_ms": sum(
            case["generation_latency_ms"] for case in cases
        ) / total,
        "cases": cases,
    }
