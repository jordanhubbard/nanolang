#!/usr/bin/env python3
"""I share the closed intrinsic set between my two frontend analyses."""
import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
spec = json.loads((ROOT / "spec/purity_intrinsics.json").read_text())
names = spec["names"]
io_names = spec["io_names"]
assert spec["schema"] == "nanolang.purity_intrinsics.v1"
assert names and len(names) == len(set(names))
assert io_names and len(io_names) == len(set(io_names))
assert not set(names).intersection(io_names)
outputs = {
    ROOT / "src/generated/purity_intrinsics.h":
        "/* I generate this file from spec/purity_intrinsics.json. */\n"
        "static const char *const purity_intrinsic_names[] = {\n"
        + "".join(f'    "{name}",\n' for name in names) + "    NULL\n};\n"
        + "static const char *const purity_io_names[] = {\n"
        + "".join(f'    "{name}",\n' for name in io_names) + "    NULL\n};\n",
    ROOT / "src_nano/generated/purity_intrinsics.nano":
        "# I generate this closed intrinsic set from spec/purity_intrinsics.json.\n"
        "pub fn purity_intrinsic(name: string) -> bool {\n"
        + "".join(f'    if (== name "{name}") {{ return true }}\n' for name in names)
        + "    return false\n}\nshadow purity_intrinsic {\n"
        '    assert (purity_intrinsic "abs")\n'
        '    assert (not (purity_intrinsic "println"))\n}\n'
        + 'pub fn purity_observable(name: string) -> bool {\n'
        + ''.join(f'    if (== name "{name}") {{ return true }}\n' for name in io_names)
        + '    return false\n}\nshadow purity_observable {\n'
        + '    assert (purity_observable "getenv")\n'
        + '    assert (not (purity_observable "abs"))\n}\n',
}
check = argparse.ArgumentParser()
check.add_argument("--check", action="store_true")
args = check.parse_args()
for path, text in outputs.items():
    if args.check:
        if not path.exists() or path.read_text() != text:
            raise SystemExit(f"I need to regenerate {path.relative_to(ROOT)}")
    else:
        path.write_text(text)
