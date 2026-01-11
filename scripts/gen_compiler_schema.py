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

    return "\n".join(lines)


def gen_nano_ast(schema: dict) -> str:
    # compiler_ast.nano is intended to hold AST/IR struct definitions that are NOT part of
    # the higher-level "contracts" layer. Contract types (diagnostics, phase outputs, etc.)
    # live in compiler_contracts.nano.
    lines: list[str] = [
        HEADER_COMMENT.strip(),
        "",
        'import "src_nano/generated/compiler_schema.nano"',
        'import "src_nano/generated/compiler_contracts.nano"',
        "",
    ]

    for enum in schema.get("nano_enums", []):
        # Enums marked as contracts are emitted to compiler_contracts.nano.
        if enum.get("contracts"):
            continue
        lines.append(f"enum {enum['name']} {{")
        for idx, value in enumerate(enum["values"]):
            suffix = "," if idx < len(enum["values"]) - 1 else ""
            lines.append(f"    {value}{suffix}")
        lines.append("}")
        lines.append("")

    for struct in schema.get("nano_structs", []):
        # Structs marked as contracts are emitted to compiler_contracts.nano.
        if struct.get("contracts"):
            continue
        # Use extern struct for types that are emitted to C - they're available via C header
        prefix = "extern " if struct.get("emit_c") else ""
            
        struct_name = struct['name']
            
        lines.append(f"{prefix}struct {struct_name} {{")
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
        prefix = "extern " if struct.get("emit_c") else ""
        lines.append(f"{prefix}struct {struct['name']} {{")
        for idx, (field_name, field_type) in enumerate(struct["fields"]):
            comma = "," if idx < len(struct["fields"]) - 1 else ""
            lines.append(f"    {field_name}: {field_type}{comma}")
        lines.append("}")
        lines.append("")

    return "\n".join(lines)


C_GUARD = "NANOLANG_GENERATED_COMPILER_SCHEMA_H"


def gen_c(schema: dict) -> str:
    lines: list[str] = [HEADER_COMMENT.strip(), "", f"#ifndef {C_GUARD}", f"#define {C_GUARD}", ""]
    
    lines.append("#include <stdint.h>")
    lines.append("#include <stdbool.h>")
    lines.append("#include \"runtime/dyn_array.h\"")
    lines.append("")
    
    # Forward declare List types
    detected_lists = set()
    for struct in schema.get("nano_structs", []):
        for _, ftype in struct["fields"]:
            if ftype.startswith("List<"):
                detected_lists.add(ftype[5:-1])
    
    if detected_lists:
        lines.append("/* Forward declare List types */")
        for inner in sorted(detected_lists):
            lines.append(f"#ifndef FORWARD_DEFINED_List_{inner}")
            lines.append(f"#define FORWARD_DEFINED_List_{inner}")
            lines.append(f"typedef struct List_{inner} List_{inner};")
            lines.append("#endif")
        lines.append("")

    # Add enum declarations for C-emitted enums
    for enum in schema.get("nano_enums", []):
        if not enum.get("emit_c"):
            continue
            
        enum_name = enum['name']
        lines.append(f"#ifndef DEFINED_{enum_name}")
        lines.append(f"#define DEFINED_{enum_name}")
        lines.append(f"typedef enum {enum_name} {{")
        for idx, val in enumerate(enum["values"]):
            suffix = "," if idx < len(enum["values"]) - 1 else ""
            lines.append(f"    {enum_name}_{val}{suffix}")
        lines.append(f"}} {enum_name};")
        lines.append("#endif")
        lines.append("")

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

    lines.append("typedef enum {")
    for idx, name in enumerate(schema["parse_nodes"]):
        suffix = "," if idx < len(schema["parse_nodes"]) - 1 else ""
        lines.append(f"    {name} = {idx}{suffix}")
    lines.append("} ParseNodeType;")
    lines.append("")

    for struct in schema.get("nano_structs", []):
        if not struct.get("emit_c"):
            continue
        
        # Schema types should match their NanoLang names in C to avoid redefinition errors
        struct_name = struct['name']
        c_struct_name = f"nl_{struct_name}"
            
        lines.append(f"#ifndef DEFINED_{c_struct_name}")
        lines.append(f"#define DEFINED_{c_struct_name}")
        lines.append(f"typedef struct {c_struct_name} {{")
        for field_name, field_type in struct["fields"]:
            c_type = field_type
            if c_type == "string":
                c_type = "const char *"
            elif c_type == "bool":
                c_type = "bool"
            elif c_type.startswith("array<"):
                c_type = "DynArray *"
            elif c_type.startswith("List<"):
                # Extract T from List<T>
                inner = c_type[5:-1]
                c_type = f"List_{inner} *"
            elif c_type == "Parser" or c_type == "TypeEnvironment":
                c_type = f"{c_type}" # Use value instead of pointer to match NanoLang semantics
            elif c_type == "Type":
                c_type = "NSType" # Special case for 'Type' conflict
            lines.append(f"    {c_type} {field_name};")
        lines.append(f"}} {c_struct_name};")
        lines.append(f"typedef {c_struct_name} {struct_name};")
        if struct_name == "LexerToken":
            lines.append("typedef nl_LexerToken Token;")
        if struct_name == "Type":
            lines.append("typedef Type NSType;") # Backwards compatibility if needed
        lines.append("#endif")
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
