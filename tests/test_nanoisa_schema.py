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

    def test_operand_kinds_are_well_formed(self):
        kinds = self.schema["operand_kinds"]
        names = [kind["name"] for kind in kinds]
        self.assertEqual(len(set(names)), len(names))
        for kind in kinds:
            self.assertTrue(kind.get("encoding"), kind)
            self.assertTrue(kind.get("meaning"), kind)

    def test_every_v2_instruction_has_one_comprehensible_meaning(self):
        """Each portable instruction carries exactly one non-empty meaning, and
        no two instructions share the same meaning."""
        meanings = []
        for instruction in self._v2_instructions():
            meaning = instruction.get("meaning")
            self.assertIsInstance(meaning, str, instruction["name"])
            self.assertTrue(meaning.strip(), instruction["name"])
            meanings.append(meaning.strip())
        self.assertEqual(
            len(set(meanings)), len(meanings), "instruction meanings must be unique"
        )

    def test_v2_operand_forms_reference_declared_kinds(self):
        """Operand forms stay symmetric: every operand names a declared operand
        kind, so a given operand form always means the same thing and no raw
        wire encoding leaks into an instruction operand list."""
        declared = {kind["name"] for kind in self.schema["operand_kinds"]}
        encodings = {kind["encoding"] for kind in self.schema["operand_kinds"]}
        for instruction in self._v2_instructions():
            for operand in instruction.get("operands", []):
                self.assertIn(operand, declared, instruction["name"])
                self.assertNotIn(
                    operand, encodings - declared, instruction["name"]
                )

    def test_load_store_variable_families_are_symmetric(self):
        """Paired get/set portable forms share identical operand forms so the
        access instructions stay symmetric."""
        by_name = {i["name"]: i for i in self._v2_instructions()}
        pairs = [
            ("local.get", "local.set"),
            ("global.get", "global.set"),
            ("upvalue.get", "upvalue.set"),
        ]
        for getter, setter in pairs:
            self.assertIn(getter, by_name)
            self.assertIn(setter, by_name)
            self.assertEqual(
                by_name[getter]["operands"],
                by_name[setter]["operands"],
                f"{getter}/{setter} operand forms must match",
            )

    def test_memory_family_operand_forms_are_uniform(self):
        """Every memory access shares the same [offset, align] operand form."""
        mem = self.schema["instruction_families"]["memory"]
        for instruction in mem:
            self.assertEqual(
                instruction["operands"],
                ["offset", "align"],
                instruction["name"],
            )


if __name__ == "__main__":
    unittest.main()
