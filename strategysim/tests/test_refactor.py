import json
import tempfile
import unittest
from pathlib import Path

from strategysim.agents.prompting import render_stage
from strategysim.orchestrator.aggregation import compute_rollout_weight
from strategysim.orchestrator.reporting import _load_results_from_records


class RefactorTests(unittest.TestCase):
    def test_perception_knowledge_can_be_removed_without_altering_user_input(self):
        context = {
            "sim_time": "12:00", "profile": "用户输入 ${history}",
            "psychological_state": "想吃饭", "coupon_section": "满 20 减 5",
        }
        with_knowledge = render_stage("incentive_perception", **context)
        without = render_stage("incentive_perception", include_knowledge=False, **context)
        self.assertIn("面额和使用门槛", with_knowledge)
        self.assertNotIn("面额和使用门槛", without)
        for value in context.values():
            self.assertIn(value, without)
        self.assertIn('"consumption_plan"', without)

    def test_continuous_weight(self):
        self.assertIsNone(compute_rollout_weight(False, None, 0.1))
        self.assertAlmostEqual(compute_rollout_weight(True, -0.5, 0.1), 0.05)
        self.assertAlmostEqual(compute_rollout_weight(True, None, 0.1), 0.1)

    def test_resume_counts_untagged_wakeups_and_skips_explicit_coupon_events(self):
        with tempfile.TemporaryDirectory() as temp:
            records = Path(temp)
            payload = {"wakeup_records": [
                {"session_id": "1_first", "consumed": True, "rollout_weight": 0.25},
                {"type": "wakeup", "consumed": False}, {"type": "coupon"},
            ]}
            (records / "user_1.json").write_text(json.dumps(payload), encoding="utf-8")
            result, = _load_results_from_records(records, {1})
        self.assertEqual(result["wakeup_count"], 2)
        self.assertEqual(result["order_count"], 0.25)
