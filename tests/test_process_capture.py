"""I preserve long commands, both output streams and the same child's status."""
import json
from pathlib import Path
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]


class ProcessCapture(unittest.TestCase):
    def test_builtin_and_module_backends(self):
        command = 'i=0; while [ "$i" -lt 8192 ]; do printf 0123456789abcdef; printf FEDCBA9876543210 >&2; i=$((i+1)); done; exit 7'
        for module in (False, True):
            for compiler, vm in (("nanoc_c", False), ("nano_virt", True)):
                with self.subTest(module=module, compiler=compiler), tempfile.TemporaryDirectory(prefix="nano-process-capture-") as tmp:
                    root = Path(tmp)
                    declaration = 'from "modules/std/process.nano" import run, Output\n' if module else ''
                    invoke = ('let r: Output = (run command)\nassert (== r.code 7)\n'
                              'assert (== r.stdout expected_out)\nassert (== r.stderr expected_err)') if module else (
                              'let r: array<string> = (process_run command)\nassert (== (at r 0) "7")\n'
                              'assert (== (at r 1) expected_out)\nassert (== (at r 2) expected_err)')
                    source = declaration + 'fn check(command: string, expected_out: string, expected_err: string) -> void {\n' + invoke + '\n}\n'
                    source += 'shadow check { (check "printf out; printf err >&2; exit 7" "out" "err") }\n'
                    source += 'fn main() -> int {\nlet mut padding: string = " "\nwhile (< (str_length padding) 16384) { set padding (+ padding padding) }\n'
                    source += '(check (+ ": " (+ padding "; printf out; printf err >&2; exit 7")) "out" "err")\n'
                    source += 'let mut out: string = "0123456789abcdef"\nlet mut err: string = "FEDCBA9876543210"\nwhile (< (str_length out) 131072) { set out (+ out out) set err (+ err err) }\n'
                    source += '(check ' + json.dumps(command) + ' out err)\nreturn 0\n}\nshadow main { assert (== (main) 0) }\n'
                    path = root / "main.nano"
                    path.write_text(source)
                    output = root / ("main.nvm" if vm else "main")
                    build = [str(ROOT / "bin" / compiler), str(path), "-o", str(output)]
                    if vm: build.append("--emit-nvm")
                    result = subprocess.run(build, cwd=ROOT, capture_output=True, timeout=60)
                    self.assertEqual(result.returncode, 0, result.stderr.decode(errors="replace"))
                    run = [str(ROOT / "bin/nano_vm"), str(output)] if vm else [str(output)]
                    result = subprocess.run(run, cwd=ROOT, capture_output=True, timeout=20)
                    self.assertEqual(result.returncode, 0, result.stderr.decode(errors="replace"))


if __name__ == "__main__":
    unittest.main()
