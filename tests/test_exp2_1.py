import unittest

from src.exp2_1_prompt.data import build_demo_inputs
from src.exp2_1_prompt.metrics import (
    calculate_keyword_coverage,
    calculate_repetition,
    calculate_text_length,
    evaluate_generations,
    remove_thought_chain,
)
from src.exp2_1_prompt.prompts import build_prompt_variants


class PromptExperimentTests(unittest.TestCase):
    def test_remove_thought_chain_keeps_final_answer(self):
        text = "开头[unused16]推理过程[unused17]最终影评[unused10]"
        self.assertEqual(remove_thought_chain(text), "开头最终影评[unused10]")

    def test_metrics_compute_length_repetition_and_keywords(self):
        text = "特效 特效 科幻 视觉"
        self.assertEqual(calculate_text_length("你 好\n"), 2)
        self.assertGreater(calculate_repetition(text, n=1), 0.0)
        self.assertEqual(calculate_keyword_coverage("特效很好，科幻感强", "特效 科幻 视觉"), 2 / 3)

    def test_evaluate_generations_returns_expected_summary(self):
        items = build_demo_inputs()
        generations = ["特效 科幻 视觉 正面 " * 35 for _ in items]
        result = evaluate_generations(items, generations, min_chars=20, max_chars=500)

        self.assertEqual(result["count"], 5)
        self.assertEqual(result["length_compliance_rate"], 1.0)
        self.assertIn("avg_keyword_coverage", result)

    def test_prompt_variants_include_baseline_and_improvements(self):
        variants = build_prompt_variants(build_demo_inputs()[0])
        names = [variant.name for variant in variants]

        self.assertIn("baseline", names)
        self.assertIn("structured_constraints", names)
        self.assertIn("fewshot_style", names)
        self.assertIn("anti_hallucination", names)


if __name__ == "__main__":
    unittest.main()
