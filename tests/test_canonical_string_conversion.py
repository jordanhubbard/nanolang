"""I preserve scalar interpolation values and evaluate operands once."""
from pathlib import Path
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
SOURCE = r'''enum Mode { Quiet = 3, Loud = 7 }
let mut calls: int = 0
fn next() -> int { set calls (+ calls 1) return calls }
shadow next { set calls 0 assert (== (next) 1) }
fn main() -> int {
 set calls 0
 let _: int = -7
 assert (== f"value {_}" "value -7")
 let octet: u8 = 255
 assert (== (to_string octet) "255")
 assert (== (to_string true) "true")
 assert (== (to_string false) "false")
 assert (== (to_string "kept") "kept")
 assert (== (cast_string 42) "42")
 let mode: Mode = Mode.Loud
 assert (== (to_string mode) "7")
 assert (== f"{2.0}/{-1.5}/{true}" "2.0/-1.5/true")
 assert (== (to_string (float_from_bits -9223372036854775808)) "-0.0")
 assert (== f"{(next)}:{(next)}" "1:2")
 assert (== calls 2)
 return 0
}
shadow main { assert (== (main) 0) }
'''

class CanonicalStringConversion(unittest.TestCase):
    def checked(self, args):
        result = subprocess.run(args, cwd=ROOT, capture_output=True, text=True, timeout=120)
        self.assertEqual(result.returncode, 0, result.stdout+result.stderr)

    def test_values_and_evaluation_order(self):
        with tempfile.TemporaryDirectory(prefix='nano-string-conversion-') as tmp:
            source, output = Path(tmp)/'case.nano', Path(tmp)/'product'
            source.write_text(SOURCE)
            for compiler in ('nanoc_c','nanoc_stage1','nanoc_stage2'):
                for vm in (False, True) if compiler != 'nanoc_c' else (False,):
                    with self.subTest(compiler=compiler, vm=vm):
                        self.checked([ROOT/'bin'/compiler,source,*(['--emit-nvm'] if vm else []),'-o',output])
                        self.checked([ROOT/'bin/nano_vm',output] if vm else [output])

    def test_declared_function_keeps_its_result(self):
        with tempfile.TemporaryDirectory(prefix='nano-named-conversion-') as tmp:
            source, output = Path(tmp)/'case.nano', Path(tmp)/'product'
            source.write_text('fn to_string(value: int) -> int { return (+ value 7) }\n'
                'shadow to_string { assert (== (to_string 5) 12) }\n'
                'fn main() -> int { assert (== (to_string 5) 12) return 0 }\n'
                'shadow main { assert (== (main) 0) }\n')
            for compiler in ('nanoc_stage1','nanoc_stage2'):
                with self.subTest(compiler=compiler):
                    self.checked([ROOT/'bin'/compiler,source,'-o',output])
                    self.checked([output])

    def test_local_function_value_keeps_its_target(self):
        with tempfile.TemporaryDirectory(prefix='nano-local-conversion-') as tmp:
            source, output = Path(tmp)/'case.nano', Path(tmp)/'product'
            source.write_text('fn render(value: int) -> string { return "local" }\n'
                'shadow render { assert (== (render 5) "local") }\n'
                'fn main() -> int { let to_string: fn(int) -> string = render '
                'assert (== (to_string 5) "local") return 0 }\n'
                'shadow main { assert (== (main) 0) }\n')
            for compiler in ('nanoc_stage1','nanoc_stage2'):
                with self.subTest(compiler=compiler):
                    self.checked([ROOT/'bin'/compiler,source,'-o',output])
                    self.checked([output])

    def test_invalid_calls_preserve_output(self):
        for body in ('let result: string = (to_string)',
                     'let result: string = (to_string 1 2)',
                     'let to_string: int = 1 let result: string = (to_string 2)'):
            with tempfile.TemporaryDirectory(prefix='nano-invalid-conversion-') as tmp:
                source, output = Path(tmp)/'case.nano', Path(tmp)/'product'
                source.write_text('fn main() -> int { '+body+' return 0 }\nshadow main { assert true }\n')
                for compiler in ('nanoc_stage1','nanoc_stage2'):
                    with self.subTest(compiler=compiler,body=body):
                        output.write_text('prior output')
                        result = subprocess.run([ROOT/'bin'/compiler,source,'-o',output],cwd=ROOT,
                                                capture_output=True,text=True,timeout=120)
                        self.assertNotEqual(result.returncode,0,result.stdout+result.stderr)
                        self.assertEqual(output.read_text(),'prior output')
