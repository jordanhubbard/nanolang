"""I retain runtime scalar tags across differing union payload variants."""
from pathlib import Path
import subprocess
import tempfile
import unittest
from tests import test_native_optional_array_reads as base

ROOT = Path(__file__).resolve().parents[1]
HEADER = '.types 1 0 1\n.string kept "kept"\n.string seven "7"\n.string truth "true"\n.string real "1.5"\n.entry main\n'


class VariantScalarCarriers(unittest.TestCase):
    checked = base.OptionalArrayReads.checked
    paired = base.OptionalArrayReads.paired

    def test_variants_both_orders_and_same_variant_instantiations(self):
        for tags in ((0, 1), (0, 0)):
            for reverse in (False, True):
                with self.subTest(tags=tags, reverse=reverse):
                    values = [('PUSH_I64 7', tags[0], 'seven'), ('PUSH_STR kept', tags[1], 'kept')]
                    if reverse: values.reverse()
                    body = ''.join(f'{producer}\nAGG_PACK 1 0 {tag} 1\nCALL relay\nCALL display\nPUSH_STR {text}\nEQ\nASSERT\n'
                                   for producer,tag,text in values)
                    self.paired(HEADER+'.function main 0 0 0 int 1\n'+body+'PUSH_I64 0\nRET\n.end\n'+
                        '.function relay 1 2 0 union 1\nLOAD_LOCAL 0\nSTORE_LOCAL 1\nLOAD_LOCAL 1\nRET\n.end\n'+
                        '.function display 1 1 0 string 1\nLOAD_LOCAL 0\nAGG_GET 0\nCAST_STRING\nRET\n.end\n')

    def test_branch_checked_consumers_and_empty_padding(self):
        producers = (('PUSH_I64 7', 'seven'), ('PUSH_STR kept', 'kept'),
                     ('PUSH_BOOL 1', 'truth'), ('PUSH_F64 1.5', 'real'))
        body = ''.join(f'{producer}\nAGG_PACK 1 0 {tag} 1\nCALL read\nPUSH_STR {expected}\nEQ\nASSERT\n'
                       for tag,(producer,expected) in enumerate(producers))
        body += 'AGG_PACK 1 0 4 0\nCALL read\nPUSH_STR kept\nEQ\nASSERT\n'
        read = '.function read 1 1 0 string 1\n'
        operations = ('PUSH_I64 0\nI64_ADD\n', '', 'BOOL_NOT\nBOOL_NOT\n', 'F64_NEG\nF64_NEG\n')
        for tag,operation in enumerate(operations):
            read += (f'LOAD_LOCAL 0\nAGG_TAG\nPUSH_I64 {tag}\nI64_EQ\nJMP_FALSE next{tag}\n'
                     'LOAD_LOCAL 0\nAGG_GET 0\n'+operation+'CAST_STRING\nRET\n'+f'next{tag}:\n')
        read += 'PUSH_STR kept\nRET\n.end\n'
        self.paired(HEADER+'.function main 0 0 0 int 1\n'+body+'PUSH_I64 0\nRET\n.end\n'+read)

    def test_owned_string_alias_and_plain_string_cast(self):
        text = HEADER+'''.string suffix "tail"
.function main 0 2 0 int 1
PUSH_STR kept
PUSH_STR suffix
STR_CONCAT
CAST_STRING
AGG_PACK 1 0 1 1
CALL relay
STORE_LOCAL 0
LOAD_LOCAL 0
AGG_GET 0
STORE_LOCAL 1
PUSH_I64 7
AGG_PACK 1 0 0 1
CALL relay
POP
LOAD_LOCAL 1
PUSH_STR kept
PUSH_STR suffix
STR_CONCAT
EQ
ASSERT
LOAD_LOCAL 0
AGG_GET 0
LOAD_LOCAL 1
EQ
ASSERT
PUSH_I64 0
RET
.end
.function relay 1 1 0 union 1
LOAD_LOCAL 0
RET
.end
'''
        self.paired(text)

    def test_struct_tuple_and_heap_conflicts_preserve_output(self):
        for kind, second in ((0, 'PUSH_STR kept'), (2, 'PUSH_STR kept'), (1, 'ARR_NEW 1')):
            with self.subTest(kind=kind), tempfile.TemporaryDirectory(prefix='variant-refusal-') as tmp:
                p=Path(tmp); assembly=p/'input.nasm'; module=p/'input.nvm'; output=p/'output.c'
                result_tag = 'union' if kind == 1 else 'struct' if kind == 0 else 'tuple'
                assembly.write_text(HEADER+'.function main 0 0 0 int 1\n'
                    f'PUSH_I64 7\nAGG_PACK {kind} 0 0 1\nCALL relay\nPOP\n'
                    f'{second}\nAGG_PACK {kind} 0 0 1\nCALL relay\nPOP\nPUSH_I64 0\nRET\n.end\n'
                    f'.function relay 1 1 0 {result_tag} 1\nLOAD_LOCAL 0\nRET\n.end\n')
                self.checked([ROOT/'bin/nanoisa','asm',assembly,'-o',module])
                output.write_text('previous')
                result=subprocess.run([ROOT/'bin/nvm2c',module,'-o',output],capture_output=True,text=True)
                self.assertEqual(result.returncode,1,result.stdout+result.stderr)
                self.assertTrue('conflicting' in result.stderr or 'cannot convert aggregate storage' in result.stderr,result.stderr)
                self.assertEqual(output.read_text(),'previous')


if __name__ == '__main__': unittest.main()
