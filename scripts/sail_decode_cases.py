#!/usr/bin/env python3
"""I validate the Sail slice against the schema and build a decoder comparison."""
import argparse
from pathlib import Path
import random
import re
import subprocess

import yaml

ROOT = Path(__file__).resolve().parents[1]
SLICE = {
    "NOP": ("Nop", [], 0, 0),
    "PUSH_I64": ("Push", ["I64"], 0, 1),
    "DUP": ("Dup", [], 1, 2),
    "POP": ("Pop", [], 1, 0),
    "SWAP": ("Swap", [], 2, 2),
}


def validate_schema(schema, model):
    if schema["encoding"]["byte_order"] != "little":
        raise ValueError("I require the reviewed little-endian encoding.")
    if schema["encoding"]["extension_prefix"] != 255:
        raise ValueError("I require the reviewed extension prefix.")
    entries = {entry["name"]: entry for entry in schema["legacy_opcodes"]}
    body = model.split("function decode(code) =", 1)[1].split("val execute", 1)[0]
    clauses = re.findall(
        r"^\s*0x([0-9a-f]{2}) :: (.*?)(?=^\s*(?:0x[0-9a-f]{2} ::|_ =>))",
        body, re.M | re.S,
    )
    decoded = {}
    for code, clause in clauses:
        constructor = re.search(r"Some\(\((\w+)\(", clause).group(1)
        if constructor in decoded:
            raise ValueError("I reject duplicate model constructors.")
        decoded[constructor] = (int(code, 16), clause.split("=>", 1)[0].count("::"))
    if set(decoded) != {entry[0] for entry in SLICE.values()}:
        raise ValueError("My model instruction inventory changed without review.")
    for name, (constructor, operands, pops, pushes) in SLICE.items():
        entry = entries[name]
        if (entry["operands"], entry["pops"], entry["pushes"]) != (operands, pops, pushes):
            raise ValueError(f"My schema contract changed for {name}.")
        if decoded[constructor] != (entry["code"], 8 if operands else 0):
            raise ValueError(f"My Sail encoding disagrees with the schema for {name}.")
    return entries


def corpus(entries):
    rng = random.Random(211)
    cases = {b""}
    push = entries["PUSH_I64"]["code"]
    values = {0, 1, (1 << 63) - 1, 1 << 63, (1 << 64) - 1, 0x0102030405060708}
    values.update(1 << bit for bit in range(64))
    values.update(((1 << 64) - 1) ^ (1 << bit) for bit in range(64))
    values.update(rng.getrandbits(64) for _ in range(64))
    for value in values:
        encoded = bytes([push]) + value.to_bytes(8, "little")
        cases.update(encoded[:length] for length in range(9))
        cases.add(encoded)
        cases.add(encoded + b"\x00\xff\x01")
    for name in SLICE:
        if name != "PUSH_I64":
            for length in range(9):
                cases.add(bytes([entries[name]["code"]]) + rng.randbytes(length))
    known = {entry["code"] for entry in entries.values()}
    for opcode in range(256):
        if opcode not in known:
            cases.add(bytes([opcode]))
            cases.add(bytes([opcode]) + bytes(range(12)))
    return sorted(cases)


def sail_bytes(data):
    return "[|" + ", ".join(f"0x{byte:02x}" for byte in data) + "|]"


def build_comparison(work):
    schema = yaml.safe_load((ROOT / "spec/nanoisa.yaml").read_text())
    entries = validate_schema(schema, (ROOT / "formal/sail/stack_slice.sail").read_text())
    cases = corpus(entries)
    oracle = work / "decode_oracle"
    subprocess.run([
        "cc", "-std=c11", "-Wall", "-Wextra", "-Werror", "-I" + str(ROOT / "src/nanoisa"),
        str(ROOT / "tests/nanoisa/sail_decode_oracle.c"), str(ROOT / "src/nanoisa/isa.c"),
        "-o", str(oracle),
    ], check=True)
    output = subprocess.run([str(oracle)], input="".join(c.hex() + "\n" for c in cases),
                            capture_output=True, text=True, check=True).stdout.splitlines()
    if len(output) != len(cases):
        raise ValueError("My production decoder returned an incomplete corpus.")
    constructors = {entries[name]["code"]: value[0] for name, value in SLICE.items()}
    lines = ["val main : unit -> unit", "function main() = {"]
    for index, (data, row) in enumerate(zip(cases, output)):
        if row == "0":
            expected = "None()"
        else:
            consumed, opcode, payload = row.split()
            consumed, opcode = int(consumed), int(opcode)
            if not 0 < consumed <= len(data):
                raise ValueError("My production decoder reported an invalid byte count.")
            name = constructors[opcode]  # No silently accepted out-of-slice result.
            value = f"0x{payload}" if name == "Push" else ""
            expected = f"Some(({name}({value}), {sail_bytes(data[consumed:])}))"
        lines += [f"  // Case {index}: {data.hex()}",
                  f"  assert(match decode({sail_bytes(data)}) {{ {expected} => true, _ => false }});"]
    lines += [f'  print_endline("I matched isa_decode on {len(cases)} byte sequences.")', "}"]
    (work / "decode_cases.sail").write_text("\n".join(lines) + "\n")
    print(f"I checked schema agreement and generated {len(cases)} production-decoder comparisons.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("work", type=Path, help="existing temporary build directory")
    args = parser.parse_args()
    build_comparison(args.work.resolve())
