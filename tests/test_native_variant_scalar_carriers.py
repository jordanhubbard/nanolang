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

    def test_optional_variant_payload_preserves_absence_and_present_tags(self):
        for tag,value in ((1, 'PUSH_I64 7'), (4, 'PUSH_BOOL 1'),
                          (3, 'PUSH_F64 1.5'), (5, 'PUSH_STR kept')):
            for missing in (0, 1):
                for reverse in (False, True):
                    with self.subTest(tag=tag, missing=missing, reverse=reverse):
                        exact = value+f'\nAGG_PACK 1 0 0 1\nCALL relay\nAGG_GET 0\nTYPE_CHECK {tag}\nASSERT\n'
                        optional = value+f'\nARR_LITERAL {tag} 1\nPUSH_I64 {missing}\nARR_GET\nAGG_PACK 1 0 0 1\nCALL relay\nAGG_GET 0\nTYPE_CHECK {0 if missing else tag}\nASSERT\n'
                        body = optional+exact if reverse else exact+optional
                        self.paired(HEADER+'.function main 0 0 0 int 1\n'+body+'PUSH_I64 0\nRET\n.end\n'+
                            '.function relay 1 1 0 union 1\nLOAD_LOCAL 0\nRET\n.end\n')

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

    def test_projected_and_concrete_scalar_expression_joins(self):
        cases = ((1, 'int', 'PUSH_I64 7'), (4, 'bool', 'PUSH_BOOL 1'),
                 (3, 'float', 'PUSH_F64 1.5'), (5, 'string', 'PUSH_STR kept'))
        for tag, name, value in cases:
            for reverse in (False, True):
                for selected in (0, 1):
                    with self.subTest(tag=tag, reverse=reverse, selected=selected):
                        projected = 'LOAD_LOCAL 0\nAGG_GET 0\n'
                        left, right = (value+'\n', projected) if reverse else (projected, value+'\n')
                        text = HEADER+'.function main 0 0 0 int 1\n'+value+'\nAGG_PACK 1 0 0 1\nCALL read\n'
                        text += f'DUP\nTYPE_CHECK {tag}\nASSERT\n'+value+'\nEQ\nASSERT\nPUSH_I64 0\nRET\n.end\n'
                        text += f'.function read 1 2 0 {name} 1\nPUSH_BOOL {selected}\nJMP_FALSE other\n'+left
                        text += 'JMP joined\nother:\n'+right+'joined:\nSTORE_LOCAL 1\nLOAD_LOCAL 1\nCALL identity\nRET\n.end\n'
                        text += f'.function identity 1 1 0 {name} 1\nLOAD_LOCAL 0\nRET\n.end\n'
                        self.paired(text)

    def test_mixed_variant_record_branch_join(self):
        for selected in (0, 1):
            with self.subTest(selected=selected):
                expected = 'seven' if selected else 'kept'
                self.paired(HEADER+f'.function main 0 0 0 int 1\nPUSH_BOOL {selected}\nJMP_FALSE other\n'
                    'PUSH_I64 7\nAGG_PACK 1 0 0 1\nJMP joined\nother:\nPUSH_STR kept\nAGG_PACK 1 0 1 1\n'
                    'joined:\nCALL display\nPUSH_STR '+expected+'\nEQ\nASSERT\nPUSH_I64 0\nRET\n.end\n'
                    '.function display 1 1 0 string 1\nLOAD_LOCAL 0\nAGG_GET 0\nCAST_STRING\nRET\n.end\n')

    def test_owned_string_alias_and_plain_string_cast(self):
        text = HEADER+'.string garbage "'+('x'*512)+'"\n'+'''.string suffix "tail"
.function main 0 3 0 int 1
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
PUSH_I64 0
STORE_LOCAL 2
churn:
LOAD_LOCAL 2
PUSH_I64 512
I64_LT_S
JMP_FALSE done
CALL temporary
POP
LOAD_LOCAL 2
PUSH_I64 1
I64_ADD
STORE_LOCAL 2
JMP churn
done:
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
.function temporary 0 0 0 string 1
PUSH_STR garbage
PUSH_STR suffix
STR_CONCAT
RET
.end
'''
        self.paired(text)

    def test_struct_tuple_and_heap_conflicts_preserve_output(self):
        for kind, second in ((0, 'PUSH_STR kept'), (2, 'PUSH_STR kept'), (1, 'ARR_NEW 4'), (1, 'PUSH_U8 1')):
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
                self.assertTrue('conflicting' in result.stderr or 'cannot convert aggregate storage' in result.stderr or 'require proved scalar producers' in result.stderr,result.stderr)
                self.assertEqual(output.read_text(),'previous')


if __name__ == '__main__': unittest.main()
