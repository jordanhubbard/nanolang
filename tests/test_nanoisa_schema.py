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
                self.assertIn("pops", instruction)
                self.assertIn("pushes", instruction)
                self.assertIn("ownership", instruction)

    def test_fixed_legacy_stack_effects_are_paired(self):
        for instruction in self.schema["legacy_opcodes"]:
            self.assertEqual("pops" in instruction, "pushes" in instruction)
        add = next(op for op in self.schema["legacy_opcodes"] if op["name"] == "ADD")
        self.assertEqual((add["pops"], add["pushes"]), (2, 1))


if __name__ == "__main__":
    unittest.main()
