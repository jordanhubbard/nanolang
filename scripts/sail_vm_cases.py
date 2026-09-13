#!/usr/bin/env python3
"""I compare the independent Sail stack semantics with actual NanoVM execution."""
import argparse
from pathlib import Path
import random
import subprocess

import yaml
from sail_decode_cases import ROOT, SLICE, validate_schema


def corpus():
    rng = random.Random(211)
    boundaries = [0, 1, (1 << 63) - 1, 1 << 63, (1 << 64) - 1,
                  0x0102030405060708]
    stacks = {()}
    for depth in range(1, 5):
        for value in boundaries:
            stacks.add(tuple((value + i) % (1 << 64) for i in range(depth)))
        for _ in range(8):
            stacks.add(tuple(rng.getrandbits(64) for _ in range(depth)))
    return [(locals_, stack, name, value)
            for locals_ in (0, 2) for stack in sorted(stacks) for name in SLICE
            for value in (boundaries if name == "PUSH_I64" else [0])]


def sail_stack(values):
    return "[|" + ", ".join(f"0x{v:016x}" for v in values) + "|]"


def build_comparison(work):
    schema = yaml.safe_load((ROOT / "spec/nanoisa.yaml").read_text())
    entries = validate_schema(schema, (ROOT / "formal/sail/stack_slice.sail").read_text())
    oracle = work / "vm_oracle"
    subprocess.run(["make", "-s", "sail-vm-oracle", f"SAIL_VM_ORACLE={oracle}"],
                   cwd=ROOT, check=True)
    cases = corpus()
    def push(value):
        return bytes([entries["PUSH_I64"]["code"]]) + value.to_bytes(8, "little")
    inputs = []
    for locals_, stack, name, value in cases:
        code = b"".join(push(v) for v in reversed(stack))
        code += push(value) if name == "PUSH_I64" else bytes([entries[name]["code"]])
        inputs.append(f"{locals_} {code.hex()}\n")
    output = subprocess.run([str(oracle)], input="".join(inputs), capture_output=True,
                            text=True, check=True).stdout.splitlines()
    if len(output) != len(cases):
        raise ValueError("My VM returned an incomplete corpus.")
    lines = ["val main : unit -> unit", "function main() = {"]
    underflows = 0
    for index, ((locals_, stack, name, value), row) in enumerate(zip(cases, output)):
        if row == "underflow":
            expected = "None()"
            underflows += 1
        else:
            fields = row.split()
            if not fields or fields[0] != "ok":
                raise ValueError(f"I reject an unknown oracle result: {row}")
            values = [int(v, 16) for v in fields[1:]]
            if any(len(v) != 16 for v in fields[1:]):
                raise ValueError("I require exact 64-bit oracle values.")
            expected = f"Some({sail_stack(values)})"
        constructor = SLICE[name][0]
        arg = f"0x{value:016x}" if name == "PUSH_I64" else ""
        lines += [f"  // Case {index}: {name}, {locals_} locals, depth {len(stack)}",
                  f"  assert(match execute({constructor}({arg}), {sail_stack(stack)}) "
                  f"{{ {expected} => true, _ => false }});"]
    lines += [f'  print_endline("I matched NanoVM on {len(cases)} stack cases '
              f'({underflows} underflows).")', "}"]
    (work / "vm_cases.sail").write_text("\n".join(lines) + "\n")
    print(f"I generated {len(cases)} production-VM comparisons ({underflows} underflows).")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("work", type=Path, help="existing temporary build directory")
    build_comparison(parser.parse_args().work.resolve())
