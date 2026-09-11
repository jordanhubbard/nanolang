#!/usr/bin/env python3
"""I reject drift between the bounded Sail slice and normative NanoISA metadata."""
import copy
from pathlib import Path
import sys
import unittest

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from sail_decode_cases import validate_schema, corpus


class SailSchemaTests(unittest.TestCase):
    def setUp(self):
        self.schema = yaml.safe_load((ROOT / "spec/nanoisa.yaml").read_text())
        self.model = (ROOT / "formal/sail/stack_slice.sail").read_text()

    def test_current_contract(self):
        self.assertIn("PUSH_I64", validate_schema(self.schema, self.model))

    def test_changed_opcode(self):
        with self.assertRaises(ValueError):
            validate_schema(self.schema, self.model.replace("0x01 =>", "0x02 =>"))

    def test_missing_instruction(self):
        with self.assertRaises(ValueError):
            validate_schema(self.schema, self.model.replace("    0x07 => Some((Dup(), rest)),\n", ""))

    def test_truncated_operand_pattern(self):
        with self.assertRaises(ValueError):
            validate_schema(self.schema, self.model.replace("b6 :: b7 :: tail", "b6 :: tail"))

    def test_changed_byte_order(self):
        self.schema["encoding"]["byte_order"] = "big"
        with self.assertRaises(ValueError):
            validate_schema(self.schema, self.model)

    def test_changed_stack_effect(self):
        for entry in self.schema["legacy_opcodes"]:
            if entry["name"] == "DUP":
                entry["pushes"] = 1
        with self.assertRaises(ValueError):
            validate_schema(self.schema, self.model)

    def test_changed_operand_kind(self):
        for entry in self.schema["legacy_opcodes"]:
            if entry["name"] == "PUSH_I64":
                entry["operands"] = ["F64"]
        with self.assertRaises(ValueError):
            validate_schema(self.schema, self.model)

    def test_changed_extension_prefix(self):
        self.schema["encoding"]["extension_prefix"] = 254
        with self.assertRaises(ValueError):
            validate_schema(self.schema, self.model)

    def test_corpus_is_reproducible_and_covers_boundaries(self):
        entries = validate_schema(self.schema, self.model)
        cases = corpus(entries)
        self.assertEqual(cases, corpus(copy.deepcopy(entries)))
        for data in (b"", b"\xff", b"\x01", b"\x01" + b"\xff" * 8,
                     b"\x01" + (1 << 63).to_bytes(8, "little")):
            self.assertIn(data, cases)


if __name__ == "__main__":
    unittest.main()
