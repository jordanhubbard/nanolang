"""I keep my approved shared match policy synchronized across public documents."""

import json
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]


class SharedMatchPolicyDocumentation(unittest.TestCase):
    def test_machine_spec_records_the_complete_policy(self):
        spec = json.loads((ROOT / "spec.json").read_text(encoding="utf-8"))
        match = spec["statements"]["match"]

        self.assertEqual(match["selection"]["order"], "lexical_first_success")
        self.assertEqual(match["selection"]["scrutinee_evaluation"],
                         "exactly_once_before_arms")
        self.assertEqual(match["guards"]["type"], "exact_bool")
        self.assertEqual(match["guards"]["evaluation"],
                         "only_after_pattern_match")
        self.assertEqual(match["wildcards"]["position"], "source_order")
        self.assertEqual(match["wildcards"]["after_unconditional"],
                         "checked_unreachable_error")
        self.assertEqual(match["totality"]["value_matches"], "required")
        self.assertEqual(match["totality"]["statement_matches"], "required")
        self.assertEqual(match["no_success"]["accepted_source"],
                         "statically_unreachable")
        self.assertEqual(match["no_success"]["forbidden_results"],
                         ["void", "zero_value", "fallthrough"])
        self.assertEqual(match["control_flow"]["return"],
                         "exits_enclosing_function")

    def test_public_guidance_states_order_totality_and_migration_boundary(self):
        required = {
            "docs/SPECIFICATION.md": (
                "lexical source order",
                "Every value and statement match is statically total",
                "precise checked capability refusal",
            ),
            "docs/CANONICAL_STYLE.md": (
                "Match Selection And Totality",
                "first arm whose pattern matches",
                "must not silently reorder arms",
            ),
            "userguide/02_control_flow.md": (
                "lexical first-success",
                "both value and statement matches to be statically total",
                "may not silently move a wildcard",
            ),
            "userguide/part1_fundamentals/05_control_flow.md": (
                "Match Order, Guards, and Totality",
                "Every match is total",
                "checked capability refusal",
            ),
        }
        for relative, phrases in required.items():
            text = (ROOT / relative).read_text(encoding="utf-8")
            normalized = " ".join(text.split())
            with self.subTest(document=relative):
                for phrase in phrases:
                    self.assertIn(phrase, normalized)


if __name__ == "__main__":
    unittest.main()
