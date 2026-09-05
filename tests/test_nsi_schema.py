import copy
import json
import re
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SCHEMA = json.loads((ROOT / "schema/nsi/v0/schema.json").read_text())
EXAMPLE = json.loads((ROOT / "schema/nsi/examples/types.nsi.json").read_text())


class ValidationError(ValueError):
    pass


SCALARS = {"bool", "i64", "u64", "f64", "string", "binary"}
ID = re.compile(r"^[a-z][a-z0-9]*(\.[a-z][a-z0-9]*)*$")
DECLARATION_KEYS = {
    "record": {"id", "name", "kind", "fields"},
    "variant": {"id", "name", "kind", "cases"},
    "resource": {"id", "name", "kind"},
    "callback": {"id", "name", "kind", "parameters", "result", "async"},
    "error": {"id", "name", "kind", "version", "fields"},
}


def validate_type_ref(type_ref):
    if isinstance(type_ref, str):
        if type_ref in SCALARS or ID.fullmatch(type_ref):
            return
    if isinstance(type_ref, dict) and set(type_ref) == {"kind", "element"}:
        if type_ref["kind"] != "array":
            raise ValidationError("unknown type-reference kind")
        validate_type_ref(type_ref["element"])
        return
    raise ValidationError("invalid type reference")


def validate_field(field):
    if not {"id", "name", "type"} <= set(field):
        raise ValidationError("incomplete field")
    if not set(field) <= {"id", "name", "type", "optional"}:
        raise ValidationError("unknown field property")
    validate_identity(field)
    validate_type_ref(field["type"])


def validate_identity(value):
    if not isinstance(value["id"], int) or value["id"] < 1:
        raise ValidationError("invalid numeric identifier")
    if not isinstance(value["name"], str) or not ID.fullmatch(value["name"]):
        raise ValidationError("invalid name")


def validate_version(version):
    if set(version) != {"major", "minor"}:
        raise ValidationError("invalid version")
    if any(not isinstance(version[key], int) or version[key] < 0 for key in version):
        raise ValidationError("invalid version number")


def validate_document(document):
    if set(document) != {"nsi", "interface", "version", "types"}:
        raise ValidationError("unknown or missing document property")
    if document["nsi"] != 0:
        raise ValidationError("unsupported NSI version")
    if not isinstance(document["interface"], str) or not ID.fullmatch(document["interface"]):
        raise ValidationError("invalid interface name")
    validate_version(document["version"])
    for declaration in document["types"]:
        kind = declaration.get("kind")
        if kind not in DECLARATION_KEYS:
            raise ValidationError("unknown declaration kind")
        if set(declaration) != DECLARATION_KEYS[kind]:
            if not (kind == "callback" and set(declaration) == DECLARATION_KEYS[kind] - {"async"}):
                raise ValidationError("unknown or missing declaration property")
        validate_identity(declaration)
        if kind in {"record", "error"}:
            if kind == "error":
                validate_version(declaration["version"])
            for field in declaration["fields"]:
                validate_field(field)
        elif kind == "variant":
            for case in declaration["cases"]:
                if not {"id", "name"} <= set(case) <= {"id", "name", "type"}:
                    raise ValidationError("invalid variant case")
                validate_identity(case)
                if "type" in case:
                    validate_type_ref(case["type"])
        elif kind == "callback":
            for parameter in declaration["parameters"]:
                validate_field(parameter)
            validate_type_ref(declaration["result"])


class NsiSchemaTests(unittest.TestCase):
    def test_typed_payload_example(self):
        self.assertEqual(SCHEMA["$schema"], "https://json-schema.org/draft/2020-12/schema")
        validate_document(EXAMPLE)

    def test_unknown_declaration_kind_fails_closed(self):
        document = copy.deepcopy(EXAMPLE)
        document["types"][0]["kind"] = "future-record"
        with self.assertRaises(ValidationError):
            validate_document(document)

    def test_unknown_type_reference_kind_fails_closed(self):
        document = copy.deepcopy(EXAMPLE)
        document["types"][0]["fields"][2]["type"]["kind"] = "set"
        with self.assertRaises(ValidationError):
            validate_document(document)

    def test_unknown_properties_fail_closed(self):
        document = copy.deepcopy(EXAMPLE)
        document["types"][0]["wire_hint"] = "native"
        with self.assertRaises(ValidationError):
            validate_document(document)


if __name__ == "__main__":
    unittest.main()
