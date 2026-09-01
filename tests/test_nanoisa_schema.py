import importlib.util
import unittest
from collections import Counter
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
            yield from family

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

    def test_every_v2_instruction_has_a_unique_meaning(self):
        meanings = []
        for instruction in self._v2_instructions():
            meaning = instruction.get("meaning")
            self.assertTrue(
                meaning and str(meaning).strip(),
                f"{instruction['name']} is missing a meaning",
            )
            meanings.append(str(meaning))
        duplicates = [m for m, c in Counter(meanings).items() if c > 1]
        self.assertEqual(duplicates, [], f"duplicate meanings: {duplicates}")

    def test_operands_reference_declared_operand_kinds(self):
        kinds = {kind["name"] for kind in self.schema["operand_kinds"]}
        for instruction in self._v2_instructions():
            for operand in instruction.get("operands", []):
                self.assertIn(
                    operand,
                    kinds,
                    f"{instruction['name']} uses undeclared operand {operand!r}",
                )

    def test_operand_forms_are_symmetric_for_paired_instructions(self):
        forms = {
            instruction["name"]: instruction.get("operands", [])
            for instruction in self._v2_instructions()
        }
        pairs = [
            ("local.get", "local.set"),
            ("global.get", "global.set"),
            ("upvalue.get", "upvalue.set"),
        ]
        for getter, setter in pairs:
            self.assertEqual(forms[getter], forms[setter], f"{getter}/{setter}")

        load_forms = {
            name: form for name, form in forms.items() if name.startswith("mem.load")
        }
        store_forms = {
            name: form for name, form in forms.items() if name.startswith("mem.store")
        }
        self.assertEqual(len(set(map(tuple, load_forms.values()))), 1)
        self.assertEqual(len(set(map(tuple, store_forms.values()))), 1)
        self.assertEqual(
            next(iter(load_forms.values())),
            next(iter(store_forms.values())),
            "mem.load*/mem.store* operand forms must match",
        )

    def test_validator_rejects_missing_meaning(self):
        broken = yaml.safe_load((ROOT / "spec/nanoisa.yaml").read_text())
        broken["instruction_families"]["stack"][0].pop("meaning")
        with self.assertRaises(ValueError):
            generator.validate(broken)

    def test_validator_rejects_undeclared_operand(self):
        broken = yaml.safe_load((ROOT / "spec/nanoisa.yaml").read_text())
        broken["instruction_families"]["memory"][0]["operands"] = ["sleb"]
        with self.assertRaises(ValueError):
            generator.validate(broken)

    def test_validator_rejects_duplicate_meaning(self):
        broken = yaml.safe_load((ROOT / "spec/nanoisa.yaml").read_text())
        first = broken["instruction_families"]["integer"][0]["meaning"]
        broken["instruction_families"]["integer"][1]["meaning"] = first
        with self.assertRaises(ValueError):
            generator.validate(broken)


if __name__ == "__main__":
    unittest.main()
