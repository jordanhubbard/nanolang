"""I propagate scalar array writes through shared record-field handles."""
from pathlib import Path
import subprocess
import tempfile
import unittest
from tests import test_native_closures as base

ROOT = base.ROOT


class NativeListRecordFields(unittest.TestCase):
    checked = base.NativeClosures.checked
    emit = base.NativeClosures.emit
    sanitized = base.NativeClosures.sanitized
    run_module = base.NativeClosures.run_module

    def test_retained_selfhosted_record_and_union_aliases(self):
        self.run_module((ROOT/'tests/nanoisa/fixtures/record_scalar_array_alias.nasm').read_text(), '')

    def test_scalar_writes_through_forwarded_record_fields(self):
        for tag, push, expected in [(1,'PUSH_I64 7','PUSH_I64 7'),
                                     (4,'PUSH_BOOL 1','PUSH_BOOL 1'),
                                     (3,'PUSH_F64 7.5','PUSH_F64 7.5'),
                                     (5,'PUSH_STR text','PUSH_STR text')]:
            with self.subTest(tag=tag):
                self.run_module('.string text "kept"\n.types 1 0 0\n.entry main\n'
                    '.function write 1 1 0 void 0\nLOAD_LOCAL 0\nAGG_GET 0\n'+push+'\nARR_PUSH\nPOP\nRET\n.end\n'
                    '.function forward 1 1 0 void 0\nLOAD_LOCAL 0\nCALL write\nRET\n.end\n'
                    '.function main 0 2 0 int 1\n'+f'ARR_NEW {tag}\nSTORE_LOCAL 0\n'
                    'LOAD_LOCAL 0\nAGG_PACK 0 0 0 1\nSTORE_LOCAL 1\nLOAD_LOCAL 1\nCALL forward\n'
                    'LOAD_LOCAL 0\nARR_LEN\nPUSH_I64 1\nEQ\nASSERT\n'
                    'LOAD_LOCAL 0\nPUSH_I64 0\nARR_GET\n'+expected+'\nEQ\nASSERT\nPUSH_I64 0\nRET\n.end\n', '')

    def test_empty_integer_array_field_uses_constructor_evidence(self):
        self.run_module('.types 1 0 0\n.entry main\n'
            '.function main 0 0 0 int 1\nARR_NEW 1\nAGG_PACK 0 0 0 1\n'
            'AGG_GET 0\nARR_LEN\nPUSH_I64 0\nEQ\nASSERT\nPUSH_I64 0\nRET\n.end\n', '')

    def test_unresolved_array_parameter_is_not_integer_constructor_evidence(self):
        with tempfile.TemporaryDirectory() as tmp:
            work=Path(tmp); source=work/'input.nasm'; module=work/'input.nvm'; output=work/'output.c'
            source.write_text('.types 1 0 0\n.entry main\n'
                '.function main 1 1 0 int 1\nLOAD_LOCAL 0\nARR_LEN\nPOP\n'
                'LOAD_LOCAL 0\nAGG_PACK 0 0 0 1\nPOP\nPUSH_I64 0\nRET\n.end\n')
            self.checked([ROOT/'bin/nanoisa','asm',source,'-o',module])
            result=subprocess.run([str(ROOT/'bin/nvm2c'),str(module),'-o',str(output)],capture_output=True,text=True,timeout=30)
            self.assertNotEqual(result.returncode,0,result.stdout+result.stderr)
            self.assertIn('cannot resolve AGG_PACK field',result.stdout+result.stderr)

    def test_scalar_set_through_record_alias(self):
        self.run_module('.types 1 0 0\n.entry main\n'
            '.function replace 1 1 0 void 0\nLOAD_LOCAL 0\nAGG_GET 0\n'
            'PUSH_I64 0\nPUSH_I64 9\nARR_SET\nPOP\nRET\n.end\n'
            '.function main 0 1 0 int 1\nPUSH_I64 7\nARR_LITERAL 1 1\nSTORE_LOCAL 0\n'
            'LOAD_LOCAL 0\nAGG_PACK 0 0 0 1\nCALL replace\n'
            'LOAD_LOCAL 0\nPUSH_I64 0\nARR_GET\nPUSH_I64 9\nEQ\nASSERT\nPUSH_I64 0\nRET\n.end\n', '')

    def test_incompatible_scalar_write_is_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            work=Path(tmp); source=work/'input.nasm'; module=work/'input.nvm'; output=work/'output.c'
            source.write_text('.string text "kept"\n.types 1 0 0\n.entry main\n'
                '.function write 1 1 0 void 0\nLOAD_LOCAL 0\nAGG_GET 0\nPUSH_I64 7\nARR_PUSH\nPOP\nRET\n.end\n'
                '.function main 0 0 0 int 1\nPUSH_STR text\nARR_LITERAL 5 1\nAGG_PACK 0 0 0 1\nCALL write\nPUSH_I64 0\nRET\n.end\n')
            self.checked([ROOT/'bin/nanoisa','asm',source,'-o',module])
            result=subprocess.run([str(ROOT/'bin/nvm2c'),str(module),'-o',str(output)],capture_output=True,text=True,timeout=30)
            self.assertNotEqual(result.returncode,0,result.stdout+result.stderr)
            self.assertRegex(result.stdout+result.stderr,'shape|representation|storage')
