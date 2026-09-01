import copy
import importlib.util
import unittest
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parent.parent
SPEC = importlib.util.spec_from_file_location(
    "gen_nanoisa_schema", ROOT / "scripts/gen_nanoisa_schema.py"
)
generator = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(generator)


class NanoisaSchemaTests(unittest.TestCase):
    def setUp(self):
        self.schema = yaml.safe_load((ROOT / "spec/nanoisa.yaml").read_text())

    def _v2_instructions(self):
        for family in self.schema["instruction_families"].values():
            for instruction in family:
                yield instruction

    def test_codes_and_names_are_unique(self):
        entries = self.schema["legacy_opcodes"]
        self.assertEqual(len({entry["code"] for entry in entries}), len(entries))
        self.assertEqual(len({entry["name"] for entry in entries}), len(entries))

    def test_generated_header_is_current(self):
        expected = generator.generate(self.schema)
        actual = (ROOT / "src/nanoisa/generated_schema.h").read_text()
        self.assertEqual(actual, expected)

    def test_v2_families_declare_stack_and_ownership(self):
        for instruction in self._v2_instructions():
            self.assertIn("pops", instruction)
            self.assertIn("pushes", instruction)
            self.assertIn("ownership", instruction)

    def test_every_instruction_has_a_unique_meaning(self):
        meanings = []
        for instruction in self._v2_instructions():
            meaning = instruction.get("meaning")
            self.assertTrue(
                meaning and meaning.strip(),
                f"{instruction['name']} is missing a meaning",
            )
            meanings.append(meaning)
        self.assertEqual(
            len(meanings), len(set(meanings)), "v2 instruction meanings must be unique"
        )

    def test_operands_reference_declared_kinds(self):
        declared = {kind["name"] for kind in self.schema["operand_kinds"]}
        for instruction in self._v2_instructions():
            for operand in instruction.get("operands", []):
                self.assertIn(
                    operand,
                    declared,
                    f"{instruction['name']} references undeclared operand kind {operand}",
                )

    def test_paired_instructions_are_symmetric(self):
        by_name = {i["name"]: i for i in self._v2_instructions()}
        for left, right in generator.SYMMETRIC_PAIRS:
            self.assertIn(left, by_name)
            self.assertIn(right, by_name)
            self.assertEqual(
                by_name[left].get("operands", []),
                by_name[right].get("operands", []),
                f"{left} and {right} must share operand forms",
            )

    def test_memory_instructions_use_offset_align(self):
        for instruction in self._v2_instructions():
            if instruction["name"].startswith("mem."):
                self.assertEqual(
                    instruction.get("operands", []),
                    ["offset", "align"],
                    f"{instruction['name']} must use the [offset, align] form",
                )

    def test_validate_accepts_current_schema(self):
        generator.validate(self.schema)

    def test_validate_rejects_missing_meaning(self):
        broken = copy.deepcopy(self.schema)
        broken["instruction_families"]["constants"][0].pop("meaning")
        with self.assertRaises(ValueError):
            generator.validate(broken)

    def test_validate_rejects_duplicate_meaning(self):
        broken = copy.deepcopy(self.schema)
        constants = broken["instruction_families"]["constants"]
        constants[1]["meaning"] = constants[0]["meaning"]
        with self.assertRaises(ValueError):
            generator.validate(broken)

    def test_validate_rejects_undeclared_operand(self):
        broken = copy.deepcopy(self.schema)
        broken["instruction_families"]["constants"][0]["operands"] = ["not-a-kind"]
        with self.assertRaises(ValueError):
            generator.validate(broken)

    def test_validate_rejects_asymmetric_pair(self):
        broken = copy.deepcopy(self.schema)
        for instruction in broken["instruction_families"]["variables"]:
            if instruction["name"] == "local.set":
                instruction["operands"] = ["local", "count"]
        with self.assertRaises(ValueError):
            generator.validate(broken)

    def test_c_string_escapes_control_and_quotes(self):
        escaped = generator.c_string('a "quote"\nand\ttab\\end')
        self.assertNotIn("\n", escaped)
        self.assertNotIn("\t", escaped)
        self.assertIn('\\"', escaped)
        self.assertIn("\\n", escaped)
        self.assertIn("\\t", escaped)
        self.assertIn("\\\\", escaped)


if __name__ == "__main__":
    unittest.main()
