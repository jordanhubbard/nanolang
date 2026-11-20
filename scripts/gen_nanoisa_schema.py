#!/usr/bin/env python3
"""Generate active NanoISA metadata from the normative v2 design schema."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parent.parent
SCHEMA = ROOT / "spec/nanoisa.yaml"
OUTPUT = ROOT / "src/nanoisa/generated_schema.h"
NANOLANG_OUTPUT = ROOT / "src_nano/nvm2c/isa_metadata.nano"


# Every portable ("public") v2 instruction records why it belongs in the
# instruction set rather than a runtime library reached through call.import.
# A frontend or runtime can compute any richer behavior by composing these; the
# ISA only exposes what a portable library provably cannot express for itself.
#
#   representation       reads or builds a value's internal layout (constant
#                        pool loads, linear-memory access, aggregate pack/get/
#                        set/tag). A library cannot reach the physical layout a
#                        portable ISA exists to hide.
#   core-semantics       a primitive whose bit-exact result the ISA must fix
#                        once (typed literals, integer and float arithmetic).
#   execution-substrate  operates on the VM's own operand stack or call frame
#                        (stack shuffles, local/global/upvalue access) that no
#                        library has portable access to.
#   control-flow         manipulates the instruction pointer, call frames, or
#                        signatures (branches, calls, tail calls, return).
#   host-boundary        the effect boundary itself (traps and host imports); a
#                        library cannot cross it without one of these.
#   encoding             an encoding-only alias of a canonical instruction; it
#                        belongs for compact size, not for any new meaning.
JUSTIFICATIONS = frozenset(
    {
        "representation",
        "core-semantics",
        "execution-substrate",
        "control-flow",
        "host-boundary",
        "encoding",
    }
)


def c_name(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "_", name).upper().strip("_")


def _as_int(value) -> int:
    if isinstance(value, str):
        return int(value, 0)
    return int(value)


def c_string(text: str) -> str:
    """Escape a Python string so it is a safe C string literal."""
    out = []
    for ch in text:
        code = ord(ch)
        if ch == "\\":
            out.append("\\\\")
        elif ch == '"':
            out.append('\\"')
        elif ch == "\n":
            out.append("\\n")
        elif ch == "\t":
            out.append("\\t")
        elif ch == "\r":
            out.append("\\r")
        elif 0x20 <= code < 0x7F:
            out.append(ch)
        else:
            for byte in ch.encode("utf-8"):
                out.append(f"\\x{byte:02x}")
    return '"' + "".join(out) + '"'


def validate(schema: dict) -> None:
    """Enforce the portable-instruction design contract.

    Every v2 family instruction must carry one comprehensible ``meaning`` and
    reference only declared operand kinds so operand forms stay symmetric.
    """
    primary = [_as_int(item["code"]) for item in schema["legacy_opcodes"]]
    if len(set(primary)) != len(primary) or any(code < 0 or code > 255 for code in primary):
        raise ValueError("I require unique one-byte allocations for primary opcodes")
    declared_kinds = {kind["name"] for kind in schema["operand_kinds"]}
    families = [
        item
        for group in schema["instruction_families"].values()
        for item in group
    ]

    names = [item["name"] for item in families]
    if len(set(names)) != len(names):
        raise ValueError("v2 family names are not unique")

    meanings: dict[str, str] = {}
    for item in families:
        name = item["name"]
        meaning = item.get("meaning")
        if not isinstance(meaning, str) or not meaning.strip():
            raise ValueError(f"{name} is missing a comprehensible meaning")
        if meaning in meanings:
            raise ValueError(
                f"{name} shares its meaning with {meanings[meaning]}; "
                "each instruction needs one distinct meaning"
            )
        meanings[meaning] = name

        justification = item.get("justification")
        if justification not in JUSTIFICATIONS:
            raise ValueError(
                f"{name} needs a justification for belonging in the ISA; "
                f"'{justification}' is not one of {sorted(JUSTIFICATIONS)}"
            )
        if justification == "encoding" and "canonical" not in item:
            raise ValueError(
                f"{name} claims an encoding justification but names no "
                "canonical instruction it aliases"
            )

        for operand in item.get("operands", []):
            if operand not in declared_kinds:
                raise ValueError(
                    f"{name} operand '{operand}' is not a declared operand kind"
                )


def generate(schema: dict) -> str:
    validate(schema)
    tags = schema["value_tags"]
    opcodes = schema["legacy_opcodes"]
    extended = schema.get("extended_opcodes", []) or []
    families = [item for group in schema["instruction_families"].values() for item in group]
    encoding = schema.get("encoding", {})
    primary_limit = _as_int(encoding.get("primary_opcode_limit", 0xFF))
    extension_prefix = _as_int(encoding.get("extension_prefix", 0xFF))
    lines = [
        "/* Generated by scripts/gen_nanoisa_schema.py. Do not edit. */",
        "#ifndef NANOISA_GENERATED_SCHEMA_H",
        "#define NANOISA_GENERATED_SCHEMA_H",
        "",
        f"#define NANOISA_SCHEMA_VERSION {schema['version']}",
        f"#define NANOISA_LEGACY_OPCODE_COUNT {len(opcodes)}",
        f"#define NANOISA_V2_FAMILY_COUNT {len(families)}",
        "",
        "/* Encoding of the opcode space. The primary plane holds one-byte",
        " * opcode identifiers below NANOISA_PRIMARY_OPCODE_LIMIT; that limit is",
        " * the exclusive upper bound of the range, not a count of opcodes. The",
        " * value NANOISA_EXTENSION_PREFIX is reserved as an escape byte and is",
        " * never an instruction on its own: it introduces a second byte that",
        " * selects an extended opcode from nanoisa_extended_opcodes[]. */",
        f"#define NANOISA_PRIMARY_OPCODE_LIMIT 0x{primary_limit:02x}",
        f"#define NANOISA_EXTENSION_PREFIX 0x{extension_prefix:02x}",
        f"#define NANOISA_EXTENDED_OPCODE_COUNT {len(extended)}",
        "",
        "static const NanoisaSchemaTag nanoisa_schema_tags[] = {",
    ]
    lines.extend(f'    {{"{tag["name"]}", 0x{tag["code"]:02x}}},' for tag in tags)
    lines.extend(["};", "", "static const NanoisaSchemaOpcode nanoisa_schema_opcodes[] = {"])
    for opcode in opcodes:
        operands = opcode.get("operands", [])
        padded = operands + ["NONE"] * (4 - len(operands))
        rendered = ", ".join(f"OPERAND_{operand}" for operand in padded)
        lines.append(
            f'    {{"{opcode["name"]}", 0x{opcode["code"]:02x}, '
            f'{len(operands)}, {{{rendered}}}, '
            f'{opcode.get("pops", -1)}, {opcode.get("pushes", -1)}}},'
        )
    lines.extend(["};", ""])
    lines.append("/* Extended opcode plane, reached through NANOISA_EXTENSION_PREFIX. */")
    if extended:
        lines.append("static const NanoisaSchemaOpcode nanoisa_extended_opcodes[] = {")
        for opcode in extended:
            operands = opcode.get("operands", [])
            padded = operands + ["NONE"] * (4 - len(operands))
            rendered = ", ".join(f"OPERAND_{operand}" for operand in padded)
            lines.append(
                f'    {{"{opcode["name"]}", 0x{_as_int(opcode["code"]):02x}, '
                f'{len(operands)}, {{{rendered}}}, 0, 0}},'
            )
        lines.append("};")
    else:
        lines.append(
            "static const NanoisaSchemaOpcode "
            "nanoisa_extended_opcodes[1] = { "
            "{0, 0, 0, {OPERAND_NONE, OPERAND_NONE, OPERAND_NONE, OPERAND_NONE}, 0, 0} };"
        )
    lines.extend(["", "static const NanoisaV2Family nanoisa_v2_families[] = {"])
    for item in families:
        lines.append(
            f'    {{"{item["name"]}", {c_string(item["meaning"])}, '
            f'"{item["ownership"]}", "{item["justification"]}", '
            f'{len(item.get("operands", []))}, {len(item.get("pops", []))}, '
            f'{len(item.get("pushes", []))}}},'
        )
    lines.extend(["};", "", "#endif", ""])
    return "\n".join(lines)


def generate_nanolang(schema: dict) -> str:
    """I share primary decode widths with the C reader without calling it."""
    validate(schema)
    sizes = {"NONE": 0, "U8": 1, "U16": 2, "U32": 4, "I32": 4, "I64": 8, "F64": 8}
    widths = [0] * 256
    for opcode in schema["legacy_opcodes"]:
        widths[_as_int(opcode["code"])] = 1 + sum(sizes[o] for o in opcode.get("operands", []))
    lines = ["# I am generated from spec/nanoisa.yaml by scripts/gen_nanoisa_schema.py.",
             "module nvm2c_isa_metadata", "", "let widths: array<int> = ["]
    for index in range(0, 256, 16):
        lines.append("    " + ", ".join(map(str, widths[index:index+16])) + ("," if index < 240 else ""))
    lines += ["]", "", "pub fn primary_width(opcode: int) -> int {",
              "    if (or (< opcode 0) (> opcode 255)) { return 0 }",
              "    return (at widths opcode)", "}", "", "shadow primary_width {",
              "    assert (== (primary_width 0) 1)",
              "    assert (== (primary_width 1) 9)",
              "    assert (== (primary_width 255) 0)",
              "    assert (== (primary_width -1) 0)", "}", ""]
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    schema = yaml.safe_load(SCHEMA.read_text())
    outputs = {OUTPUT: generate(schema), NANOLANG_OUTPUT: generate_nanolang(schema)}
    if args.check:
        stale = [p for p, rendered in outputs.items() if not p.exists() or p.read_text() != rendered]
        for path in stale:
            print(f"{path.relative_to(ROOT)} is out of date")
        return int(bool(stale))
    for path, rendered in outputs.items():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(rendered)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
