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

    def test_codes_and_names_are_unique(self):
        entries = self.schema["legacy_opcodes"]
        self.assertEqual(len({entry["code"] for entry in entries}), len(entries))
        self.assertEqual(len({entry["name"] for entry in entries}), len(entries))

    def test_generated_header_is_current(self):
        expected = generator.generate(self.schema)
        actual = (ROOT / "src/nanoisa/generated_schema.h").read_text()
        self.assertEqual(actual, expected)

    def test_v2_families_declare_stack_and_ownership(self):
        for family in self.schema["instruction_families"].values():
            for instruction in family:
                self.assertTrue(instruction["meaning"])
                self.assertIn("pops", instruction)
                self.assertIn("pushes", instruction)
                self.assertIn("ownership", instruction)
                self.assertEqual(
                    len(instruction["operands"]), len(instruction["operand_roles"])
                )

    def test_symmetric_v2_forms_use_the_same_immediates(self):
        instructions = {
            instruction["name"]: instruction
            for family in self.schema["instruction_families"].values()
            for instruction in family
        }
        pairs = (
            ("local.get", "local.set"),
            ("global.get", "global.set"),
            ("upvalue.get", "upvalue.set"),
            ("mem.load8", "mem.store8"),
            ("mem.load16", "mem.store16"),
            ("mem.load32", "mem.store32"),
            ("mem.load64", "mem.store64"),
            ("aggregate.get", "aggregate.set"),
            ("branch", "branch.zero"),
            ("branch", "branch.nonzero"),
            ("call", "tail.call"),
        )
        for left, right in pairs:
            self.assertEqual(
                instructions[left]["operands"], instructions[right]["operands"]
            )
            self.assertEqual(
                instructions[left]["operand_roles"], instructions[right]["operand_roles"]
            )


if __name__ == "__main__":
    unittest.main()
