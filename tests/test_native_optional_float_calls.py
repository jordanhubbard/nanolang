"""I retain exact float tags across concrete and optional native storage."""
from tests import test_native_optional_array_reads as base
import unittest

class OptionalFloatCalls(unittest.TestCase):
    checked = base.OptionalArrayReads.checked
    paired = base.OptionalArrayReads.paired

    def test_plain_and_array_read_call_orders(self):
        plain = 'PUSH_F64 1.5\n'
        read = 'PUSH_F64 1.5\nARR_LITERAL 3 1\nPUSH_I64 0\nARR_GET\n'
        for producers in ((plain, read), (read, plain)):
            with self.subTest(producers=producers):
                body = ''.join(value + 'CALL relay\nPUSH_F64 1.5\nF64_EQ\nASSERT\n' for value in producers)
                self.paired('.entry main\n.function main 0 0 0 int 1\n' + body +
                    'PUSH_I64 0\nRET\n.end\n.function relay 1 2 0 float 1\n'
                    'LOAD_LOCAL 0\nSTORE_LOCAL 1\nLOAD_LOCAL 1\nDUP\nTYPE_CHECK 3\nASSERT\nRET\n.end\n')

    def test_plain_then_optional_local(self):
        self.paired('''.entry main
.function main 0 1 0 int 1
PUSH_F64 -0.0
STORE_LOCAL 0
LOAD_LOCAL 0
CALL relay
POP
PUSH_F64 2.5
ARR_LITERAL 3 1
PUSH_I64 0
ARR_GET
STORE_LOCAL 0
LOAD_LOCAL 0
CALL relay
PUSH_F64 2.5
F64_EQ
ASSERT
PUSH_I64 0
RET
.end
.function relay 1 1 0 float 1
LOAD_LOCAL 0
RET
.end
''')

    def test_optional_absence_stays_void(self):
        for indices in ((0, 1), (1, 0)):
            body = 'PUSH_F64 1.5\nCALL present\nASSERT\n'
            for index in indices:
                body += f'PUSH_F64 1.5\nARR_LITERAL 3 1\nPUSH_I64 {index}\nARR_GET\nCALL present\n'
                if index: body += 'NOT\n'
                body += 'ASSERT\n'
            self.paired('.entry main\n.function main 0 0 0 int 1\n'+body+
                'PUSH_I64 0\nRET\n.end\n.function present 1 1 0 bool 1\n'
                'LOAD_LOCAL 0\nTYPE_CHECK 3\nRET\n.end\n')

    def test_incompatible_present_payloads_preserve_output(self):
        from pathlib import Path
        import subprocess
        import tempfile
        root = Path(__file__).resolve().parents[1]
        for tag, value in ((4, 'PUSH_BOOL 1'), (5, 'PUSH_STR text')):
            with self.subTest(tag=tag), tempfile.TemporaryDirectory(prefix='nano-float-call-refusal-') as tmp:
                p = Path(tmp); assembly = p/'input.nasm'; module = p/'input.nvm'; output = p/'output.c'
                assembly.write_text('.string text "value"\n.entry main\n.function main 0 0 0 int 1\n'
                    'PUSH_F64 1.5\nCALL present\nPOP\n'+value+f'\nARR_LITERAL {tag} 1\nPUSH_I64 0\nARR_GET\n'
                    'CALL present\nPOP\nPUSH_I64 0\nRET\n.end\n.function present 1 1 0 bool 1\n'
                    'LOAD_LOCAL 0\nTYPE_CHECK 3\nRET\n.end\n')
                self.checked([root/'bin/nanoisa', 'asm', assembly, '-o', module])
                self.checked([root/'bin/nano_vm', '--verify-only', module])
                self.checked([root/'bin/nano_vm', module])
                output.write_text('previous')
                result = subprocess.run([root/'bin/nvm2c',module,'-o',output],capture_output=True,text=True)
                self.assertEqual(result.returncode,1,result.stdout+result.stderr)
                self.assertIn('incompatible',result.stderr)
                self.assertEqual(output.read_text(),'previous')
