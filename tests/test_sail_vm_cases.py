#!/usr/bin/env python3
"""I keep the bounded VM comparison corpus reproducible and non-vacuous."""
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from sail_vm_cases import corpus


class VmCorpusTests(unittest.TestCase):
    def test_deterministic_unique_cases(self):
        cases = corpus()
        self.assertEqual(cases, corpus())
        self.assertEqual(len(cases), len(set(cases)))
        self.assertEqual(len(cases), 1140)

    def test_operand_depth_and_local_boundary(self):
        cases = corpus()
        for locals_ in (0, 2):
            for name, needed in (("DUP", 1), ("POP", 1), ("SWAP", 2)):
                depths = {len(stack) for local, stack, op, _ in cases
                          if local == locals_ and op == name}
                self.assertEqual(depths, set(range(5)))
                self.assertTrue(any(d < needed for d in depths))
                self.assertTrue(any(d >= needed for d in depths))
        underflows = [case for case in cases
                      if len(case[1]) < {"DUP": 1, "POP": 1, "SWAP": 2}.get(case[2], 0)]
        self.assertEqual(len(underflows), 34)

    def test_bit_boundaries_and_ordering(self):
        cases = corpus()
        values = {value for _, _, name, value in cases if name == "PUSH_I64"}
        self.assertTrue({0, 1, (1 << 63) - 1, 1 << 63, (1 << 64) - 1} <= values)
        self.assertTrue(any(len(stack) >= 2 and stack[0] != stack[1]
                            for _, stack, name, _ in cases if name == "SWAP"))


if __name__ == "__main__":
    unittest.main()
