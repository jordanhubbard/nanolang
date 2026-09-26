#!/usr/bin/env python3
"""I retain bounded local slots across successive ordered record literals."""

from pathlib import Path
import subprocess
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]


class RecordLiteralSlotReuse(unittest.TestCase):
    def test_successive_ordered_literals_do_not_exhaust_locals(self):
        literals = "\n".join(
            f"    set rows (array_push rows Five {{ a: {i}, b: {i + 1}, c: {i + 2}, d: {i + 3}, e: {i + 4} }})"
            for i in range(210)
        )
        source = (
            "struct Five { a: int, b: int, c: int, d: int, e: int }\n"
            "fn build() -> array<Five> {\n"
            "    let mut rows: array<Five> = []\n"
            f"{literals}\n"
            "    return rows\n"
            "}\n"
            "shadow build {\n"
            "    let rows: array<Five> = (build)\n"
            "    assert (== (array_length rows) 210)\n"
            "    assert (== (at rows 209).e 213)\n"
            "}\n"
        )
        with tempfile.TemporaryDirectory(prefix="nanolang-record-slots-") as temp:
            temp_path = Path(temp)
            program = temp_path / "records.nano"
            module = temp_path / "records.nvm"
            program.write_text(source)
            compiled = subprocess.run(
                [ROOT / "bin/nano_virt", program, "--emit-nvm", "-o", module],
                cwd=ROOT, text=True, capture_output=True,
            )
            self.assertEqual(compiled.returncode, 0, compiled.stdout + compiled.stderr)
            executed = subprocess.run(
                [ROOT / "bin/nano_vm", module], cwd=ROOT, text=True, capture_output=True,
            )
            self.assertEqual(executed.returncode, 0, executed.stdout + executed.stderr)


if __name__ == "__main__":
    unittest.main(verbosity=2)
