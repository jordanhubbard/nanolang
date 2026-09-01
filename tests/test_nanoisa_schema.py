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

    def families(self):
        return [
            instruction
            for family in self.schema["instruction_families"].values()
            for instruction in family
        ]

    def test_codes_and_names_are_unique(self):
        entries = self.schema["legacy_opcodes"]
        self.assertEqual(len({entry["code"] for entry in entries}), len(entries))
        self.assertEqual(len({entry["name"] for entry in entries}), len(entries))

    def test_generated_header_is_current(self):
        expected = generator.generate(self.schema)
        actual = (ROOT / "src/nanoisa/generated_schema.h").read_text()
        self.assertEqual(actual, expected)

    def test_v2_families_declare_stack_and_ownership(self):
        for instruction in self.families():
            self.assertIn("pops", instruction)
            self.assertIn("pushes", instruction)
            self.assertIn("ownership", instruction)

    def test_every_instruction_has_a_comprehensible_meaning(self):
        for instruction in self.families():
            meaning = instruction.get("meaning")
            self.assertIsInstance(meaning, str, instruction["name"])
            self.assertTrue(meaning.strip(), instruction["name"])

    def test_meanings_are_unique(self):
        meanings = [instruction["meaning"] for instruction in self.families()]
        self.assertEqual(len(set(meanings)), len(meanings))

    def test_operands_reference_declared_kinds(self):
        declared = {kind["name"] for kind in self.schema["operand_kinds"]}
        for instruction in self.families():
            for operand in instruction.get("operands", []):
                self.assertIn(operand, declared, instruction["name"])

    def test_get_set_pairs_are_symmetric(self):
        by_name = {inst["name"]: inst for inst in self.families()}
        for base in ("local", "global", "upvalue"):
            getter = by_name[f"{base}.get"]
            setter = by_name[f"{base}.set"]
            self.assertEqual(getter["operands"], setter["operands"], base)

    def test_memory_operands_are_uniform(self):
        mem = [inst for inst in self.families() if inst["name"].startswith("mem.")]
        self.assertTrue(mem)
        for instruction in mem:
            self.assertEqual(instruction["operands"], ["offset", "align"])

    def test_validate_rejects_missing_meaning(self):
        broken = copy.deepcopy(self.schema)
        broken["instruction_families"]["stack"][0].pop("meaning")
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

    def test_c_string_escapes_dangerous_characters(self):
        self.assertEqual(generator.c_string('a"b'), '"a\\"b"')
        self.assertEqual(generator.c_string("a\\b"), '"a\\\\b"')
        self.assertEqual(generator.c_string("a\nb"), '"a\\nb"')
        self.assertEqual(generator.c_string("café"), '"caf\\xc3\\xa9"')

    def test_fixed_legacy_stack_effects_are_paired(self):
        for instruction in self.schema["legacy_opcodes"]:
            self.assertEqual("pops" in instruction, "pushes" in instruction)
        add = next(op for op in self.schema["legacy_opcodes"] if op["name"] == "ADD")
        self.assertEqual((add["pops"], add["pushes"]), (2, 1))

    def _all_family_instructions(self):
        by_name = {}
        for family in self.schema["instruction_families"].values():
            for instruction in family:
                by_name[instruction["name"]] = instruction
        return by_name

    def test_compact_operand_kinds_alias_canonical_encodings(self):
        kinds = list(self.schema["operand_kinds"])
        canonical_encodings = {
            kind["encoding"] for kind in kinds if "canonical" not in kind
        }
        compact_kinds = [k for k in kinds if "canonical" in k]
        self.assertTrue(compact_kinds, "expected compact operand kinds")
        for kind in compact_kinds:
            # Each compact operand decodes to a canonical variable-length
            # encoding already used by a non-compact operand kind.
            self.assertIn(kind["canonical"], canonical_encodings)
            self.assertIn("range", kind)
            low, high = kind["range"]
            self.assertLessEqual(low, high)

    def test_compact_forms_alias_canonical_instructions(self):
        instructions = self._all_family_instructions()
        compact_family = self.schema["instruction_families"].get("compact", [])
        self.assertTrue(compact_family, "expected a compact instruction family")
        for compact in compact_family:
            self.assertIn(
                "canonical",
                compact,
                f"{compact['name']} must name its canonical instruction",
            )
            canonical = instructions.get(compact["canonical"])
            self.assertIsNotNone(
                canonical,
                f"{compact['name']} aliases unknown {compact['canonical']}",
            )
            # A compact form is an encoding-only variant: identical stack effect
            # and ownership keep assembly regular.
            self.assertEqual(compact["pops"], canonical["pops"])
            self.assertEqual(compact["pushes"], canonical["pushes"])
            self.assertEqual(compact["ownership"], canonical["ownership"])
            self.assertNotIn(
                "canonical",
                canonical,
                "a compact form must alias a canonical (non-compact) instruction",
            )

    def test_compact_forms_use_compact_operands(self):
        compact_kinds = {
            kind["name"]
            for kind in self.schema["operand_kinds"]
            if "canonical" in kind
        }
        for compact in self.schema["instruction_families"].get("compact", []):
            self.assertTrue(
                any(op in compact_kinds for op in compact.get("operands", [])),
                f"{compact['name']} must carry a compact operand",
            )

    def test_extension_prefix_is_not_a_primary_opcode(self):
        encoding = self.schema["encoding"]
        prefix = generator._as_int(encoding["extension_prefix"])
        limit = generator._as_int(encoding["primary_opcode_limit"])
        # The prefix is the exclusive upper bound of the primary plane: no
        # legacy opcode may use it, and it must equal the plane limit.
        self.assertEqual(prefix, limit)
        codes = {generator._as_int(entry["code"]) for entry in self.schema["legacy_opcodes"]}
        self.assertNotIn(prefix, codes)
        self.assertTrue(all(code < limit for code in codes))

    def test_extended_plane_is_defined_and_disjoint(self):
        self.assertIn("extended_opcodes", self.schema)
        extended = self.schema["extended_opcodes"] or []
        ext_codes = [generator._as_int(entry["code"]) for entry in extended]
        # Extended codes are a separate one-byte plane; they may reuse values
        # from the primary plane because the prefix disambiguates them.
        self.assertEqual(len(ext_codes), len(set(ext_codes)))
        for code in ext_codes:
            self.assertTrue(0 <= code <= 0xFF)

    def test_generated_header_exposes_encoding_constants(self):
        rendered = generator.generate(self.schema)
        self.assertIn("#define NANOISA_PRIMARY_OPCODE_LIMIT", rendered)
        self.assertIn("#define NANOISA_EXTENSION_PREFIX", rendered)
        self.assertIn("#define NANOISA_EXTENDED_OPCODE_COUNT", rendered)
        self.assertIn("nanoisa_extended_opcodes", rendered)

    def test_v2_families_exclude_string_and_collection_algorithms(self):
        """Trimming, case, splitting, replacement, formatting, parsing, and
        collection algorithms live in runtime libraries, not the portable ISA."""
        forbidden = (
            "trim", "lower", "upper", "split", "join", "replace",
            "starts_with", "ends_with", "index_of", "format", "parse",
            "to_int", "to_float",
        )
        for family_name, family in self.schema["instruction_families"].items():
            for instruction in family:
                name = instruction["name"].lower()
                for token in forbidden:
                    self.assertNotIn(
                        token,
                        name,
                        msg=(
                            f"v2 family '{family_name}' instruction "
                            f"'{instruction['name']}' looks like a runtime-library "
                            f"algorithm ('{token}') that must stay out of the ISA"
                        ),
                    )

    def test_runtime_library_algorithms_are_documented(self):
        algorithms = self.schema["runtime_library_algorithms"]
        self.assertTrue(algorithms)
        expected_categories = {
            "trimming",
            "case-conversion",
            "splitting",
            "replacement",
            "formatting",
            "parsing",
            "collection",
        }
        seen_categories = set()
        for entry in algorithms:
            self.assertIn("name", entry)
            self.assertIn("category", entry)
            self.assertIn("primitives", entry)
            self.assertTrue(entry["primitives"])
            seen_categories.add(entry["category"])
        self.assertLessEqual(expected_categories, seen_categories)

    def test_runtime_library_algorithms_are_not_v2_instructions(self):
        family_names = {
            instruction["name"]
            for family in self.schema["instruction_families"].values()
            for instruction in family
        }
        for entry in self.schema["runtime_library_algorithms"]:
            self.assertNotIn(entry["name"], family_names)


if __name__ == "__main__":
    unittest.main()
