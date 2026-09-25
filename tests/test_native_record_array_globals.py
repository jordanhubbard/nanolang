"""I retain exact record-array globals and every owner reachable through them."""

import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from tests.native_toolchain import native_cc


ROOT = Path(__file__).resolve().parents[1]


class NativeRecordArrayGlobals(unittest.TestCase):
    def command(self, args, timeout=90):
        leak_detection = "0" if sys.platform == "darwin" else "1"
        return subprocess.run(
            [str(arg) for arg in args], cwd=ROOT, capture_output=True, text=True,
            timeout=timeout,
            env={**os.environ, "ASAN_OPTIONS": f"detect_leaks={leak_detection}"},
        )

    def assemble(self, work, text):
        assembly = work / "input.nasm"
        module = work / "input.nvm"
        assembly.write_text(text)
        result = self.command([ROOT / "bin/nanoisa", "asm", assembly, "-o", module])
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        return module

    def test_empty_alias_nested_fields_overwrite_and_collection_match_vm(self):
        text = """.string before "before"
.string after "after"
.entry main
.function main 0 0 0 int 1
LOAD_GLOBAL 0
ARR_LEN
PUSH_I64 0
I64_EQ
ASSERT
CALL make_before
STORE_GLOBAL 0
LOAD_GLOBAL 0
CALL verify_alias
RET
.end
.function __init__ 0 0 0 void 0
ARR_NEW 8
STORE_GLOBAL 0
RET
.end
.function make_before 0 0 0 array 1
PUSH_STR before
PUSH_STR before
ARR_LITERAL 5 1
AGG_PACK 0 0 0 2
ARR_LITERAL 8 1
RET
.end
.function make_after 0 0 0 array 1
PUSH_STR after
PUSH_STR after
ARR_LITERAL 5 1
AGG_PACK 0 0 0 2
ARR_LITERAL 8 1
RET
.end
.function churn_many 0 1 0 void 0
PUSH_I64 0
STORE_LOCAL 0
loop:
LOAD_LOCAL 0
PUSH_I64 2048
I64_LT_S
JMP_FALSE done
CALL make_after
POP
LOAD_LOCAL 0
PUSH_I64 1
I64_ADD
STORE_LOCAL 0
JMP loop
done:
RET
.end
.function verify_alias 1 1 0 int 1
CALL make_after
STORE_GLOBAL 0
CALL churn_many
LOAD_LOCAL 0
PUSH_I64 0
ARR_GET
AGG_GET 0
PUSH_STR before
EQ
ASSERT
LOAD_LOCAL 0
PUSH_I64 0
ARR_GET
AGG_GET 1
PUSH_I64 0
ARR_GET
PUSH_STR before
EQ
ASSERT
LOAD_GLOBAL 0
PUSH_I64 0
ARR_GET
AGG_GET 0
PUSH_STR after
EQ
ASSERT
LOAD_LOCAL 0
PUSH_I64 0
ARR_GET
AGG_GET 0
PRINTLN
LOAD_GLOBAL 0
PUSH_I64 0
ARR_GET
AGG_GET 0
PRINTLN
PUSH_I64 0
RET
.end
.parameters 5 array
"""
        with tempfile.TemporaryDirectory(prefix="nano-record-array-global-") as tmp:
            work = Path(tmp)
            module = self.assemble(work, text)
            vm = self.command([ROOT / "bin/nano_vm", module])
            self.assertEqual(vm.returncode, 0, vm.stdout + vm.stderr)

            source, binary = work / "input.c", work / "program"
            translated = self.command([ROOT / "bin/nvm2c", module, "-o", source])
            if translated.returncode:
                disassembly = self.command([ROOT / "bin/nanoisa", "dump", module])
                translated.stderr += "\n" + disassembly.stdout + disassembly.stderr
            self.assertEqual(translated.returncode, 0, translated.stdout + translated.stderr)
            generated = source.read_text()
            self.assertIn("nglobal[0].integer != 6", generated)
            self.assertIn("nroot_value(&work, nglobal[i])", generated)
            compiled = self.command([
                *native_cc(), "-std=c11", "-O1", "-g", "-Wall", "-Wextra", "-Werror",
                "-fsanitize=address,undefined", "-fno-sanitize-recover=all",
                source, "-o", binary,
            ])
            self.assertEqual(compiled.returncode, 0, compiled.stdout + compiled.stderr)
            native = self.command([binary])
            self.assertEqual(native.returncode, 0, native.stdout + native.stderr)
            self.assertEqual(native.stdout, vm.stdout)
            self.assertEqual(native.stdout, "before\nafter\n")

    def test_dynamic_or_conflicting_record_array_globals_are_refused(self):
        fixtures = {
            "dynamic_slot": """.entry main
.function main 0 0 0 int 1
PUSH_I64 7
STORE_GLOBAL 0
PUSH_I64 0
RET
.end
.function __init__ 0 0 0 void 0
ARR_NEW 8
STORE_GLOBAL 0
RET
.end
""",
            "conflicting_element_shape": """.string text "text"
.entry main
.function main 0 0 0 int 1
PUSH_I64 7
AGG_PACK 0 0 0 1
ARR_LITERAL 8 1
STORE_GLOBAL 0
PUSH_I64 0
RET
.end
.function __init__ 0 0 0 void 0
PUSH_STR text
AGG_PACK 0 0 0 1
ARR_LITERAL 8 1
STORE_GLOBAL 0
RET
.end
""",
        }
        with tempfile.TemporaryDirectory(prefix="nano-record-array-global-bad-") as tmp:
            work = Path(tmp)
            for name, text in fixtures.items():
                with self.subTest(name=name):
                    module = self.assemble(work, text)
                    output = work / "retained.c"
                    output.write_text("retained")
                    translated = self.command([ROOT / "bin/nvm2c", module, "-o", output])
                    self.assertNotEqual(translated.returncode, 0)
                    self.assertIn("record-array global", translated.stderr)
                    self.assertEqual(output.read_text(), "retained")


if __name__ == "__main__":
    unittest.main()
