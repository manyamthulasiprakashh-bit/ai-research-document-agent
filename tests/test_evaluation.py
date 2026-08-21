import unittest
from types import SimpleNamespace

from app.evaluation.dataset import EVALUATION_DATASET
from app.evaluation.evaluator import (
    evaluate_answer,
    evaluate_retrieval,
    keyword_coverage,
    run_evaluation,
)


class EvaluationTests(unittest.TestCase):
    def test_dataset_contains_ten_cases(self) -> None:
        self.assertGreaterEqual(len(EVALUATION_DATASET), 10)
        for case in EVALUATION_DATASET:
            self.assertIn("question", case)
            self.assertIn("expected_source", case)
            self.assertIn("expected_keywords", case)

    def test_retrieval_metrics(self) -> None:
        results = [
            {"source": "research_notes.txt", "score": 0.9},
            {"source": "sample.txt", "score": 0.8},
        ]

        metrics = evaluate_retrieval(results, "research_notes.txt")

        self.assertEqual(metrics, {"hit": 1, "top_k_hit": 1, "source_match": 1})
        self.assertEqual(evaluate_retrieval([], "research_notes.txt")["hit"], 0)

    def test_keyword_coverage(self) -> None:
        self.assertEqual(keyword_coverage("retrieval and generation", ["retrieval", "generation"]), 1.0)
        self.assertEqual(keyword_coverage("retrieval", ["retrieval", "generation"]), 0.5)
        self.assertEqual(keyword_coverage("other", ["retrieval", "generation"]), 0.0)

    def test_answer_evaluation_checks_sources(self) -> None:
        evaluation = evaluate_answer(
            "See research_notes.txt for retrieval.",
            [{"source": "research_notes.txt"}],
            ["retrieval"],
        )

        self.assertEqual(evaluation["keyword_coverage"], 1.0)
        self.assertTrue(evaluation["answer_has_sources"])

    def test_run_evaluation_aggregates_real_case_values(self) -> None:
        def research(question: str):
            return SimpleNamespace(
                local_results=[{"source": "expected.txt", "score": 0.9}],
                web_results=[],
                answer="retrieval generation expected.txt",
                total_latency_ms=10.0,
                retrieval_latency_ms=2.0,
                generation_latency_ms=8.0,
            )

        report = run_evaluation(
            [{
                "question": "Question",
                "expected_source": "expected.txt",
                "expected_keywords": ["retrieval", "generation"],
            }],
            research,
        )

        self.assertEqual(report["total_questions"], 1)
        self.assertEqual(report["retrieval_hit_rate"], 1.0)
        self.assertEqual(report["source_match_rate"], 1.0)
        self.assertEqual(report["keyword_coverage"], 1.0)
        self.assertEqual(report["average_latency_ms"], 10.0)


if __name__ == "__main__":
    unittest.main()
