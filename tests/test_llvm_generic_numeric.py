"""I compare generic numeric operations across VM, LLVM and Wasm."""
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]


class GenericNumeric(unittest.TestCase):
    def setUp(self):
        for tool in ('llvm-as', 'lli', 'opt', 'clang', 'llc', 'wasm-ld', 'wasmtime'):
            self.assertIsNotNone(shutil.which(tool), tool)
        self.tmp = tempfile.TemporaryDirectory(prefix='nano-numeric-')
        self.addCleanup(self.tmp.cleanup)
        self.work = Path(self.tmp.name)

    def run_cmd(self, args, success=True):
        p = subprocess.run([str(a) for a in args], capture_output=True, text=True, timeout=30,
                           env={**os.environ, 'ASAN_OPTIONS': 'detect_leaks=1:abort_on_error=1'})
        self.assertEqual(p.returncode == 0, success, str(args) + '\n' + p.stdout + p.stderr)
        return p

    def program(self, body, suffix=''):
        return '.entry main\n.function main 0 2 0 int 1\n' + body + 'PUSH_I64 0\nRET\n.end\n' + suffix

    def compare(self, body, suffix='', trap=False):
        assembly, module = self.work/'input.nasm', self.work/'input.nvm'
        assembly.write_text(self.program(body, suffix))
        self.run_cmd([ROOT/'bin/nanoisa', 'asm', assembly, '-o', module])
        self.run_cmd([ROOT/'bin/nano_vm', module], success=not trap)
        ir, bc, exe, wasm = (self.work/n for n in ('out.ll','out.bc','native','out.wasm'))
        self.run_cmd([ROOT/'bin/nvm2llvm', module, '-o', ir])
        self.run_cmd(['llvm-as', ir, '-o', bc])
        self.run_cmd(['lli', bc], success=not trap)
        self.run_cmd(['opt', '-passes=default<O2>', bc, '-o', bc])
        self.run_cmd(['lli', bc], success=not trap)
        self.run_cmd(['clang', '-O2', '-fsanitize=address,undefined', '-fno-sanitize-recover=all', ir, '-o', exe])
        self.run_cmd([exe], success=not trap)
        self.run_cmd([ROOT/'bin/nvm2wasm', module, '-o', wasm])
        result = self.run_cmd(['wasmtime', 'run', '--invoke', 'nano_entry', wasm], success=not trap)
        if not trap:
            self.assertEqual(result.stdout, '0\n')
        return module

    def check(self, left, op, right, expected, tag):
        return (f'{left}\n' + (f'{right}\n' if right else '') + f'{op}\n'
                f'DUP\nTYPE_CHECK {tag}\nASSERT\n{expected}\nEQ\nASSERT\n')

    def test_integer_wrap_and_total_arithmetic(self):
        cases = [('ADD',9223372036854775807,1,-9223372036854775808),
                 ('SUB',-9223372036854775808,1,9223372036854775807),
                 ('MUL',9223372036854775807,2,-2),
                 ('DIV',-9223372036854775808,-1,-9223372036854775808),
                 ('MOD',-9223372036854775808,-1,0),
                 ('DIV',17,0,0),('MOD',17,0,0),('DIV',-17,5,-3),('MOD',-17,5,-2)]
        body = ''.join(self.check(f'PUSH_I64 {a}',op,f'PUSH_I64 {b}',f'PUSH_I64 {r}',1)
                       for op,a,b,r in cases)
        body += self.check('PUSH_I64 -9223372036854775808','NEG',None,'PUSH_I64 -9223372036854775808',1)
        self.compare(body)

    def test_float_and_mixed_promotion_both_orders(self):
        body = ''
        for op, result in [('ADD','10.5'),('SUB','5.5'),('MUL','20.0'),('DIV','3.2')]:
            body += self.check('PUSH_F64 8.0',op,'PUSH_F64 2.5',f'PUSH_F64 {result}',3)
            body += self.check('PUSH_I64 8',op,'PUSH_F64 2.5',f'PUSH_F64 {result}',3)
        for op, result in [('ADD','10.5'),('SUB','-5.5'),('MUL','20.0'),('DIV','0.3125')]:
            body += self.check('PUSH_F64 2.5',op,'PUSH_I64 8',f'PUSH_F64 {result}',3)
        body += self.check('PUSH_I64 9007199254740993','ADD','PUSH_F64 0.0','PUSH_F64 9007199254740992.0',3)
        body += self.check('PUSH_F64 0.0','ADD','PUSH_I64 -9007199254740993','PUSH_F64 -9007199254740992.0',3)
        self.compare(body)

    def test_nan_infinity_and_total_float_division(self):
        body = ''
        for numerator in ('PUSH_F64 nan','PUSH_F64 inf','PUSH_F64 -3.0','PUSH_I64 7'):
            for denominator in ('PUSH_F64 0.0','PUSH_F64 -0.0'):
                body += self.check(numerator,'DIV',denominator,'PUSH_F64 0.0',3)
        body += self.check('PUSH_F64 nan','DIV','PUSH_I64 0','PUSH_F64 0.0',3)
        body += self.check('PUSH_F64 inf','ADD','PUSH_I64 2','PUSH_F64 inf',3)
        for op in ('ADD','SUB','MUL','DIV'):
            body += f'PUSH_F64 nan\nPUSH_I64 2\n{op}\nDUP\nTYPE_CHECK 3\nASSERT\nDUP\nNE\nASSERT\n'
        body += self.check('PUSH_F64 3.5','NEG',None,'PUSH_F64 -3.5',3)
        body += self.check('PUSH_F64 -0.0','NEG',None,'PUSH_F64 0.0',3)
        self.compare(body)

    def test_calls_and_dynamic_join_preserve_tags(self):
        suffix = ('.function difference 2 2 0 float 1\n.parameters difference int float\n'
                  'LOAD_LOCAL 0\nLOAD_LOCAL 1\nSUB\nRET\n.end\n')
        body = ('PUSH_I64 8\nPUSH_F64 2.5\nCALL difference\nSTORE_LOCAL 0\n'
                'LOAD_LOCAL 0\nPUSH_F64 5.5\nEQ\nASSERT\n')
        for condition,tag in ((0,3),(1,1)):
            body += (f'PUSH_BOOL {condition}\nJMP_FALSE floating{condition}\nPUSH_I64 4\n'
                     f'JMP joined{condition}\nfloating{condition}:\nPUSH_F64 4.0\n'
                     f'joined{condition}:\nSTORE_LOCAL 1\nLOAD_LOCAL 1\nPUSH_I64 2\nMUL\n'
                     f'DUP\nTYPE_CHECK {tag}\nASSERT\nPUSH_I64 8\nEQ\nASSERT\n')
        self.compare(body,suffix)

    def test_invalid_scalar_tags_still_fail_at_runtime(self):
        for value in ('PUSH_U8 1','PUSH_BOOL 1','PUSH_VOID'):
            for op in ('ADD','SUB','MUL','DIV','MOD'):
                for reverse in (False,True):
                    with self.subTest(value=value,op=op,reverse=reverse):
                        operands = f'{value}\nPUSH_I64 2\n' if not reverse else f'PUSH_I64 2\n{value}\n'
                        self.compare(operands+op+'\nPOP\n',trap=True)
            with self.subTest(value=value,op='NEG'):
                self.compare(value+'\nNEG\nPOP\n',trap=True)
        for left,right in [('PUSH_F64 2.0','PUSH_I64 1'),('PUSH_I64 2','PUSH_F64 1.0'),('PUSH_F64 2.0','PUSH_F64 1.0')]:
            self.compare(left+'\n'+right+'\nMOD\nPOP\n',trap=True)

    def test_signed_zero_bits(self):
        cases = [('PUSH_F64 0.0\nNEG\n', 1 << 63),
                 ('PUSH_F64 -0.0\nNEG\n', 0),
                 ('PUSH_F64 -0.0\nPUSH_I64 2\nMUL\n', 1 << 63),
                 ('PUSH_F64 nan\nPUSH_F64 -0.0\nDIV\n', 0),
                 ('PUSH_F64 -inf\nPUSH_F64 -0.0\nDIV\n', 0)]
        for body, expected in cases:
            with self.subTest(body=body):
                asm, module, ir = (self.work/n for n in ('bits.nasm','bits.nvm','bits.ll'))
                asm.write_text(self.program('CALL value\nPOP\n',
                    '.function value 0 0 0 float 1\n'+body+'RET\n.end\n'))
                self.run_cmd([ROOT/'bin/nanoisa', 'asm', asm, '-o', module])
                result = self.run_cmd([ROOT/'obj/generic_numeric_bits', module])
                self.assertEqual(int(result.stdout.strip(), 16), expected)
                self.run_cmd([ROOT/'bin/nvm2llvm', module, '-o', ir])
                # I add an observation-only harness around the unchanged generated
                # function, since source equality deliberately equates signed zeros.
                text = ir.read_text().replace('define i32 @main(', 'define i32 @original_main(')
                text += ('define i32 @nano_bits_check() {\n %v = call %V @f1()\n'
                         ' %bits = extractvalue %V %v, 0\n'
                         f' %same = icmp eq i64 %bits, {expected}\n'
                         ' %bad = xor i1 %same, true\n %status = zext i1 %bad to i32\n'
                         ' ret i32 %status\n}\n'
                         'define i32 @main() {\n %status = call i32 @nano_bits_check()\n ret i32 %status\n}\n')
                ir.write_text(text)
                self.run_cmd(['lli', ir])
                exe, obj, wasm = (self.work/n for n in ('bits-native','bits.o','bits.wasm'))
                self.run_cmd(['clang','-O2','-fsanitize=address,undefined',ir,'-o',exe])
                self.run_cmd([exe])
                self.run_cmd(['llc','-mtriple=wasm32-unknown-unknown','-filetype=obj',ir,'-o',obj])
                self.run_cmd(['wasm-ld','--no-entry','--export=nano_bits_check',obj,'-o',wasm])
                self.assertEqual(self.run_cmd(['wasmtime','run','--invoke','nano_bits_check',wasm]).stdout,'0\n')

    def test_argument_evaluation_remains_eager(self):
        suffix = '.function right 0 0 0 int 1\nPUSH_BOOL 0\nASSERT\nPUSH_I64 0\nRET\n.end\n'
        self.compare('PUSH_I64 0\nCALL right\nMUL\nPOP\n',suffix,trap=True)


if __name__ == '__main__':
    unittest.main()
