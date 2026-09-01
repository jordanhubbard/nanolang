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

    def test_every_v2_instruction_has_one_comprehensible_meaning(self):
        meanings = {}
        for instruction in self._v2_instructions():
            meaning = instruction.get("meaning")
            self.assertTrue(
                meaning and meaning.strip(),
                f"{instruction['name']} is missing a meaning",
            )
            self.assertNotIn(
                meaning,
                meanings,
                f"{instruction['name']} and {meanings.get(meaning)} share a meaning",
            )
            meanings[meaning] = instruction["name"]

    def test_operand_forms_reference_declared_kinds(self):
        declared = {kind["name"] for kind in self.schema["operand_kinds"]}
        for instruction in self._v2_instructions():
            for operand in instruction.get("operands", []):
                self.assertIn(
                    operand,
                    declared,
                    f"{instruction['name']} uses undeclared operand kind {operand}",
                )

    def test_paired_instructions_use_symmetric_operands(self):
        by_name = {
            instruction["name"]: instruction for instruction in self._v2_instructions()
        }
        get_set_pairs = [
            ("local.get", "local.set"),
            ("global.get", "global.set"),
            ("upvalue.get", "upvalue.set"),
            ("aggregate.get", "aggregate.set"),
        ]
        for getter, setter in get_set_pairs:
            self.assertEqual(
                by_name[getter]["operands"],
                by_name[setter]["operands"],
                f"{getter}/{setter} operand forms are asymmetric",
            )
        for width in (8, 16, 32, 64):
            load = by_name[f"mem.load{width}"]["operands"]
            store = by_name[f"mem.store{width}"]["operands"]
            self.assertEqual(
                load,
                store,
                f"mem.load{width}/mem.store{width} operand forms are asymmetric",
            )
            self.assertEqual(load, ["offset", "align"])

    def test_validate_rejects_missing_meaning(self):
        broken = yaml.safe_load((ROOT / "spec/nanoisa.yaml").read_text())
        broken["instruction_families"]["stack"][0].pop("meaning")
        with self.assertRaises(ValueError):
            generator.validate(broken)

    def test_validate_rejects_undeclared_operand(self):
        broken = yaml.safe_load((ROOT / "spec/nanoisa.yaml").read_text())
        broken["instruction_families"]["stack"][0]["operands"] = ["not-a-kind"]
        with self.assertRaises(ValueError):
            generator.validate(broken)


if __name__ == "__main__":
    unittest.main()
