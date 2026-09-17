"""I retain exact calculator host calls in both bytecode producers."""
import os
from pathlib import Path
import shlex
import sys
import subprocess
import tempfile
import unittest

ROOT=Path(__file__).resolve().parents[1]

class CalculatorHostAbi(unittest.TestCase):
    def checked(self, args):
        result=subprocess.run(list(map(str,args)),cwd=ROOT,capture_output=True,timeout=120,
                              env={**os.environ,'ASAN_OPTIONS': 'detect_leaks=0' if sys.platform == 'darwin' else 'detect_leaks=1'})
        self.assertEqual(result.returncode,0,(result.stdout+result.stderr)[-6000:])
        return result

    def paired(self, source):
        with tempfile.TemporaryDirectory(prefix='nano-calculator-abi-') as tmp:
            work=Path(tmp); program=work/'input.nano';program.write_text(source)
            for frontend in ('nano_virt','nanoisa_emit'):
                with self.subTest(frontend=frontend):
                    module=work/(frontend+'.nvm'); c_file=work/(frontend+'.c')
                    self.checked([ROOT/'bin'/frontend,program,'--emit-nvm','-o',module])
                    self.checked([ROOT/'bin/nano_vm','--verify-only',module])
                    vm=self.checked([ROOT/'bin/nano_vm',module])
                    self.checked([ROOT/'bin/nvm2c',module,'-o',c_file])
                    cc=shlex.split(os.environ.get('NANO_NATIVE_TEST_CC','cc'))
                    binary=work/(frontend+'-native')
                    flags=['-ldl','-rdynamic'] if sys.platform.startswith('linux') else []
                    self.checked([*cc,'-std=c11','-O1','-g','-Wall','-Wextra','-Werror',
                                  '-fsanitize=address,undefined','-fno-sanitize-recover=all',
                                  c_file,ROOT/'bin/nano_aot_runtime.o','-lm',*flags,'-o',binary])
                    self.assertEqual(self.checked([binary]).stdout,vm.stdout)

    def test_strlen_and_float_argument_result(self):
        self.paired('''extern fn strlen(value: string) -> int
extern fn atan(value: float) -> float
let mut angle: float = 0.0
fn call(value: float) -> float { return (atan value) }
shadow call { assert true }
fn main() -> int {
 assert (== (strlen "") 0)
 assert (== (strlen "lambda: λ") 10)
 assert (== (call 0.0) 0.0)
 assert (== (atan angle) 0.0)
 set angle 1.0
 assert (> (atan angle) 0.785)
 assert (> (call 1.0) 0.785)
 assert (< (call 1.0) 0.786)
 assert (< (call -1.0) -0.785)
 (println "host-abi") return 0
}
shadow main { assert true }
''')

    def test_declared_function_keeps_its_body(self):
        self.paired('''fn atan(value: float) -> float { return (+ value 42.0) }
shadow atan { assert true }
fn strlen(value: string) -> int { return 17 }
shadow strlen { assert true }
fn main() -> int { assert (== (atan 0.0) 42.0) assert (== (strlen "abc") 17) return 0 }
shadow main { assert true }
''')

    def test_named_library_does_not_acquire_builtin_adapter(self):
        with tempfile.TemporaryDirectory(prefix='nano-host-identity-') as tmp:
            work=Path(tmp);assembly=work/'input.nasm';module=work/'input.nvm';output=work/'previous.c'
            assembly.write_text('.import "other-library" "atan" float float\n.entry main\n'
                                '.function main 0 0 0 int 1\nPUSH_F64 1\nCALL_EXTERN 0\nPOP\n'
                                'PUSH_I64 0\nRET\n.end\n')
            self.checked([ROOT/'bin/nanoisa','asm',assembly,'-o',module])
            output.write_text('previous source')
            result=subprocess.run([ROOT/'bin/nvm2c',module,'-o',output],capture_output=True,cwd=ROOT)
            self.assertNotEqual(result.returncode,0)
            self.assertIn(b'exact builtin host ABI',result.stdout+result.stderr)
            self.assertEqual(output.read_text(),'previous source')

if __name__=='__main__':unittest.main()
