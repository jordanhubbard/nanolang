"""I reject incomplete or inconsistent linker transport measurements."""

import unittest

from tests.characterize_link_argument_transport import require_consistent


class LinkArgumentAcceptance(unittest.TestCase):
    def test_every_acceptance_field_is_required(self):
        statuses = ("direct_status", "build_status", "returned_status", "later_status")
        answers = ("direct_answer", "cold_answer", "warm_answer", "later_answer")
        invariants = ("generation_reused", "initial_generation_preserved", "failed_replacement_preserved",
                      "metadata_unchanged", "returned_arguments_equal")
        case = {**dict.fromkeys(statuses, 0), **dict.fromkeys(answers, 42), **dict.fromkeys(invariants, True)}
        require_consistent({"cases": [case]})
        for key in statuses + answers + invariants:
            with self.subTest(key=key):
                changed = dict(case)
                changed[key] = 1 if key in statuses else None if key in answers else False
                with self.assertRaises(SystemExit):
                    require_consistent({"cases": [case, changed]})

    def test_empty_measurement_is_not_acceptance(self):
        with self.assertRaises(SystemExit):
            require_consistent({"cases": []})
