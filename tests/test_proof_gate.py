#!/usr/bin/env python3
"""I test the proof gate's failure paths without requiring a prover installation."""
from pathlib import Path
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
CHECKER = ROOT / "formal/check_assumptions.sh"
CONTRACTS = ("preservation", "progress", "determinism", "equivalence", "evaluator")
SOURCE = "".join(f"Print Assumptions {name}_contract.\n" for name in CONTRACTS)
CLOSED = "Closed under the global context\n"


class ProofGateTests(unittest.TestCase):
    def run_gate(self, report, source=SOURCE, required=()):
        with tempfile.TemporaryDirectory(prefix="nanocore-gate-") as directory:
            root = Path(directory)
            (root / "Assumptions.v").write_text(source)
            (root / "report").write_text(report)
            return subprocess.run(
                ["bash", str(CHECKER), str(root / "Assumptions.v"), str(root / "report"), *required],
                capture_output=True, text=True, check=False,
            ).returncode

    def test_closed_reports_pass(self):
        self.assertEqual(self.run_gate(CLOSED * 5), 0)

    def test_blank_lines_pass(self):
        self.assertEqual(self.run_gate((CLOSED + "\n") * 5), 0)

    def test_axiom_report_fails(self):
        self.assertNotEqual(self.run_gate(CLOSED * 4 + "Axioms:\ninjected : False\n"), 0)

    def test_axiom_with_expected_closed_count_still_fails(self):
        self.assertNotEqual(self.run_gate(CLOSED * 5 + "Axioms:\ninjected : False\n"), 0)

    def test_missing_report_fails(self):
        self.assertNotEqual(self.run_gate(CLOSED * 4), 0)

    def test_extra_report_fails(self):
        self.assertNotEqual(self.run_gate(CLOSED * 6), 0)

    def test_empty_report_fails(self):
        self.assertNotEqual(self.run_gate(""), 0)

    def test_missing_required_theorem_fails(self):
        source = SOURCE.replace("Print Assumptions evaluator_contract.\n", "")
        self.assertNotEqual(self.run_gate(CLOSED * 4, source), 0)

    def test_empty_manifest_fails(self):
        self.assertNotEqual(self.run_gate("", ""), 0)

    def test_unexpected_output_fails(self):
        self.assertNotEqual(self.run_gate(CLOSED * 5 + "Something changed\n"), 0)

    def test_missing_report_file_fails(self):
        with tempfile.TemporaryDirectory(prefix="nanocore-gate-") as directory:
            result = subprocess.run(
                ["bash", str(CHECKER), str(ROOT / "formal/Assumptions.v"),
                 str(Path(directory) / "missing")], capture_output=True, check=False,
            )
            self.assertNotEqual(result.returncode, 0)

    def test_sail_inventory_and_reports(self):
        required = ("nop_identity", "push_then_pop", "dup_then_pop", "swap_involution",
                    "dup_underflow", "pop_underflow", "swap_empty_underflow", "swap_singleton_underflow",
                    "execute_frame_extension")
        source = (ROOT / "formal/sail/StackSliceProofs.v").read_text()
        self.assertEqual(self.run_gate(CLOSED * 9, source, required), 0)
        for report in (CLOSED * 8, CLOSED * 10, CLOSED * 9 + "Axioms:\ninjected : False\n"):
            with self.subTest(report=report):
                self.assertNotEqual(self.run_gate(report, source, required), 0)
        missing = source.replace("Print Assumptions swap_involution.\n", "")
        self.assertNotEqual(self.run_gate(CLOSED * 8, missing, required), 0)


if __name__ == "__main__":
    unittest.main()
