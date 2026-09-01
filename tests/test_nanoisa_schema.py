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


def families(schema):
    return [item for group in schema["instruction_families"].values() for item in group]


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
        for family in families(self.schema):
            self.assertIn("pops", family)
            self.assertIn("pushes", family)
            self.assertIn("ownership", family)

    def test_every_family_has_one_unique_meaning(self):
        meanings = [family.get("meaning") for family in families(self.schema)]
        self.assertTrue(all(isinstance(m, str) and m.strip() for m in meanings))
        self.assertEqual(len(set(meanings)), len(meanings))

    def test_operands_reference_declared_kinds(self):
        kinds = {kind["name"] for kind in self.schema["operand_kinds"]}
        for family in families(self.schema):
            for operand in family.get("operands", []):
                self.assertIn(operand, kinds, family["name"])

    def test_operand_kinds_pin_one_encoding(self):
        for kind in self.schema["operand_kinds"]:
            self.assertIn("encoding", kind)
            self.assertTrue(kind["encoding"])

    def test_memory_access_operands_are_symmetric(self):
        by_name = {family["name"]: family for family in families(self.schema)}
        form = ["mem-offset", "mem-align"]
        for width in (8, 16, 32, 64):
            load = by_name[f"mem.load{width}"]["operands"]
            store = by_name[f"mem.store{width}"]["operands"]
            self.assertEqual(load, form)
            self.assertEqual(store, form)

    def test_get_set_pairs_share_operand_forms(self):
        by_name = {family["name"]: family for family in families(self.schema)}
        for base in ("local", "global", "upvalue", "aggregate"):
            self.assertEqual(
                by_name[f"{base}.get"]["operands"],
                by_name[f"{base}.set"]["operands"],
            )

    def test_validate_accepts_current_schema(self):
        generator.validate(self.schema)

    def test_validate_rejects_missing_meaning(self):
        broken = copy.deepcopy(self.schema)
        del broken["instruction_families"]["stack"][0]["meaning"]
        with self.assertRaises(ValueError):
            generator.validate(broken)

    def test_validate_rejects_duplicate_meaning(self):
        broken = copy.deepcopy(self.schema)
        stack = broken["instruction_families"]["stack"]
        stack[1]["meaning"] = stack[0]["meaning"]
        with self.assertRaises(ValueError):
            generator.validate(broken)

    def test_validate_rejects_undeclared_operand(self):
        broken = copy.deepcopy(self.schema)
        broken["instruction_families"]["constants"][0]["operands"] = ["sleb"]
        with self.assertRaises(ValueError):
            generator.validate(broken)

    def test_validate_rejects_asymmetric_pair(self):
        broken = copy.deepcopy(self.schema)
        broken["instruction_families"]["memory"][0]["operands"] = ["mem-offset"]
        with self.assertRaises(ValueError):
            generator.validate(broken)


if __name__ == "__main__":
    unittest.main()
