#!/usr/bin/env python3
"""Generate shared compiler schema artifacts for NanoLang."""

from __future__ import annotations

import json
from pathlib import Path
from textwrap import dedent

ROOT = Path(__file__).resolve().parents[1]
SCHEMA_PATH = ROOT / "schema" / "compiler_schema.json"
NANO_SCHEMA_OUT = ROOT / "src_nano" / "generated" / "compiler_schema.nano"
NANO_AST_OUT = ROOT / "src_nano" / "generated" / "compiler_ast.nano"
NANO_CONTRACTS_OUT = ROOT / "src_nano" / "generated" / "compiler_contracts.nano"
C_OUT = ROOT / "src" / "generated" / "compiler_schema.h"


HEADER_COMMENT = """/* AUTO-GENERATED FILE - DO NOT EDIT DIRECTLY. */
"""


def load_schema() -> dict:
    data = json.loads(SCHEMA_PATH.read_text())
    return data


def prepare_tokens(schema: dict) -> list[dict]:
    prepared: list[dict] = []
    for idx, entry in enumerate(schema["tokens"]):
        if isinstance(entry, str):
            prepared.append({"name": entry, "value": idx})
        else:
            token = dict(entry)
            token.setdefault("value", idx)
            prepared.append(token)
    return prepared


def gen_nano_schema(schema: dict) -> str:
    tokens = prepare_tokens(schema)
    lines: list[str] = [HEADER_COMMENT.strip(), ""]

    lines.append("/* Shared lexer token definition */")
    lines.append("struct LexerToken {")
    for idx, field in enumerate(schema["token"]["fields"]):
        ftype = field["type"]
        if ftype == "enum":
            nano_type = "int"
        else:
            nano_type = ftype
        comma = "," if idx < len(schema["token"]["fields"]) - 1 else ""
        lines.append(f"    {field['name']}: {nano_type}{comma}")
    lines.append("}")
    lines.append("")

    lines.append("enum LexerTokenType {")
    for idx, token in enumerate(tokens):
        comment = token.get("comment")
        suffix = "," if idx < len(tokens) - 1 else ""
        entry = f"    {token['name']} = {token['value']}{suffix}"
        if comment:
            entry += f",  /* {comment} */"
        lines.append(entry)
    lines.append("}")
    lines.append("")

    lines.append("enum ParseNodeType {")
    for idx, name in enumerate(schema["parse_nodes"]):
        suffix = "," if idx < len(schema["parse_nodes"]) - 1 else ""
        lines.append(f"    {name} = {idx}{suffix}")
    lines.append("}")
    lines.append("")

    lines.append("struct TypeEnvironment {")
    for idx, field in enumerate(schema["type_environment"]):
        comma = "," if idx < len(schema["type_environment"]) - 1 else ""
        lines.append(f"    {field['name']}: {field['type']}{comma}")
    lines.append("}")
    lines.append("")

    return "\n".join(lines)


def gen_nano_ast(schema: dict) -> str:
    lines: list[str] = [HEADER_COMMENT.strip(), "", 'import "src_nano/generated/compiler_schema.nano"', ""]

    for enum in schema.get("nano_enums", []):
        lines.append(f"enum {enum['name']} {{")
        for idx, value in enumerate(enum["values"]):
            suffix = "," if idx < len(enum["values"]) - 1 else ""
            lines.append(f"    {value}{suffix}")
        lines.append("}")
        lines.append("")

    for struct in schema.get("nano_structs", []):
        lines.append(f"struct {struct['name']} {{")
        for idx, (field_name, field_type) in enumerate(struct["fields"]):
            comma = "," if idx < len(struct["fields"]) - 1 else ""
            lines.append(f"    {field_name}: {field_type}{comma}")
        lines.append("}")
        lines.append("")

    return "\n".join(lines)


def gen_contracts(schema: dict) -> str:
    lines: list[str] = [HEADER_COMMENT.strip(), "", 'import "src_nano/generated/compiler_schema.nano"', ""]

    for enum in schema.get("nano_enums", []):
        if not enum.get("contracts"):
            continue
        lines.append(f"enum {enum['name']} {{")
        for idx, value in enumerate(enum["values"]):
            suffix = "," if idx < len(enum["values"]) - 1 else ""
            lines.append(f"    {value}{suffix}")
        lines.append("}")
        lines.append("")

    for struct in schema.get("nano_structs", []):
        if not struct.get("contracts"):
            continue
        lines.append(f"struct {struct['name']} {{")
        for idx, (field_name, field_type) in enumerate(struct["fields"]):
            comma = "," if idx < len(struct["fields"]) - 1 else ""
            lines.append(f"    {field_name}: {field_type}{comma}")
        lines.append("}")
        lines.append("")

    return "\n".join(lines)


C_GUARD = "NANOLANG_GENERATED_COMPILER_SCHEMA_H"


def gen_c(schema: dict) -> str:
    lines: list[str] = [HEADER_COMMENT.strip(), "", f"#ifndef {C_GUARD}", f"#define {C_GUARD}", ""]

    tokens = prepare_tokens(schema)
    lines.append("typedef enum {")
    for idx, token in enumerate(tokens):
        comment = token.get("comment")
        suffix = "," if idx < len(tokens) - 1 else ""
        entry = f"    {token['name']} = {token['value']}{suffix}"
        if comment:
            entry += f",  /* {comment} */"
        lines.append(entry)
    lines.append("} TokenType;")
    lines.append("")

    lines.append("typedef struct {")
    for field in schema["token"]["fields"]:
        c_type = "TokenType" if field["type"] == "enum" else field["type"]
        if c_type == "string":
            c_type = "char *"
        lines.append(f"    {c_type} {field['name']};")
    lines.append("} Token;")
    lines.append("")

    lines.append("typedef enum {")
    for idx, name in enumerate(schema["parse_nodes"]):
        suffix = "," if idx < len(schema["parse_nodes"]) - 1 else ""
        lines.append(f"    {name} = {idx}{suffix}")
    lines.append("} ParseNodeType;")
    lines.append("")

    lines.append("typedef struct {")
    for field in schema["type_environment"]:
        c_type = "bool" if field["type"] == "bool" else field["type"]
        lines.append(f"    {c_type} {field['name']};")
    lines.append("} TypeEnvironment;")
    lines.append("")

    for struct in schema.get("nano_structs", []):
        if not struct.get("emit_c"):
            continue
        lines.append(f"typedef struct nl_{struct['name']} {{")
        for field_name, field_type in struct["fields"]:
            c_type = field_type
            if c_type == "string":
                c_type = "char *"
            elif c_type == "bool":
                c_type = "bool"
            lines.append(f"    {c_type} {field_name};")
        lines.append(f"}} {struct['name']};")
        lines.append("")

    lines.append(f"#endif /* {C_GUARD} */")
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    schema = load_schema()
    NANO_SCHEMA_OUT.parent.mkdir(parents=True, exist_ok=True)
    C_OUT.parent.mkdir(parents=True, exist_ok=True)
    NANO_SCHEMA_OUT.write_text(gen_nano_schema(schema))
    NANO_AST_OUT.write_text(gen_nano_ast(schema))
    NANO_CONTRACTS_OUT.write_text(gen_contracts(schema))
    C_OUT.write_text(gen_c(schema))


if __name__ == "__main__":
    main()
