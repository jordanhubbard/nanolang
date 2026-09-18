"""I preserve literal byte strings without a dynamic allocator or host imports."""
from tests import test_llvm_generic_numeric as numeric
from tests import test_llvm_scalar_globals as globals_test
import unittest

ROOT = numeric.ROOT


class LiteralStrings(unittest.TestCase):
    setUp = numeric.GenericNumeric.setUp
    run_cmd = numeric.GenericNumeric.run_cmd
    compare = numeric.GenericNumeric.compare
    assemble = globals_test.ScalarGlobals.assemble
    repeated = globals_test.ScalarGlobals.repeated

    def program(self, body, suffix=''):
        # Empty and NUL values must not borrow C strlen. A separate module-API
        # fixture bypasses normal pool deduplication to test distinct identities.
        return ('.string empty ""\n.string a "a\\x00z"\n.string b "a\\x00z"\n'
                '.string c "a\\x00y"\n.string prefix "a"\n.string utf "é雪"\n'
                '.string high "\\xff"\n.string low "\\x7f"\n' +
                numeric.GenericNumeric.program(self, body, suffix))

    def test_lengths_content_and_unsigned_byte_order(self):
        body = ''
        for name,length in [('empty',0),('a',3),('utf',5)]:
            body += f'PUSH_STR {name}\nSTR_LEN\nPUSH_I64 {length}\nEQ\nASSERT\n'
        for a,b,op in [('a','b','EQ'),('a','b','STR_EQ'),('a','c','GT'),('prefix','a','LT'),
                       ('empty','a','LT'),('high','low','GT'),('a','c','NE'),('a','b','LE'),('a','b','GE')]:
            body += f'PUSH_STR {a}\nPUSH_STR {b}\n{op}\nASSERT\n'
        self.compare(body)
        ir = (self.work/'out.ll').read_text()
        self.assertNotIn('inttoptr', ir)
        self.assertNotIn('@malloc', ir)
        self.assertIn('\\61\\00\\7A', ir)

    def test_empty_truthiness_and_mixed_tags(self):
        self.compare('PUSH_STR empty\nDUP\nASSERT\nCAST_BOOL\nASSERT\n'
                     'PUSH_STR empty\nNOT\nNOT\nASSERT\n'
                     'PUSH_STR empty\nPUSH_BOOL 1\nAND\nASSERT\n'
                     'PUSH_STR empty\nPUSH_BOOL 0\nOR\nASSERT\n'
                     'PUSH_STR empty\nJMP_TRUE yes\nPUSH_BOOL 0\nASSERT\nyes:\n'
                     'PUSH_STR a\nPUSH_I64 1\nNE\nASSERT\n'
                     'PUSH_STR a\nPUSH_I64 1\nGT\nASSERT\n'
                     'PUSH_I64 1\nPUSH_STR a\nLT\nASSERT\n')

    def test_call_return_local_branch_and_global_aliases(self):
        self.compare('PUSH_STR a\nCALL identity\nDUP\nSTORE_GLOBAL 0\nSTORE_LOCAL 0\n'
                     'PUSH_BOOL 0\nJMP_FALSE other\nPUSH_STR empty\nJMP joined\n'
                     'other:\nPUSH_STR b\njoined:\nSTORE_GLOBAL 1\n'
                     'PUSH_STR c\nSTORE_GLOBAL 0\n'
                     'LOAD_LOCAL 0\nLOAD_GLOBAL 1\nEQ\nASSERT\n'
                     'LOAD_GLOBAL 0\nPUSH_STR c\nSTR_EQ\nASSERT\n'
                     'LOAD_GLOBAL 1\nTYPE_CHECK 5\nASSERT\n',
                     '.function identity 1 1 0 string 1\n.parameters identity string\nLOAD_LOCAL 0\nRET\n.end\n')

    def test_implicit_string_return(self):
        self.compare('CALL text\nPUSH_STR b\nEQ\nASSERT\n',
                     '.function text 0 0 0 string 1\nPUSH_STR a\n.end\n')

    def test_instance_lifetime_and_initializer_result(self):
        text = ('.string first "a\\x00b"\n.string next ""\n.entry main\n'
                '.function main 0 0 0 int 1\nLOAD_GLOBAL 0\nTYPE_CHECK 0\nJMP_FALSE existing\n'
                'PUSH_STR first\nSTORE_GLOBAL 0\nPUSH_I64 1\nRET\nexisting:\n'
                'LOAD_GLOBAL 0\nSTR_LEN\nPUSH_I64 3\nEQ\nASSERT\n'
                'PUSH_STR next\nSTORE_GLOBAL 0\nPUSH_I64 2\nRET\n.end\n'
                '.function __init__ 0 0 0 string 1\nPUSH_STR next\nRET\n.end\n')
        self.repeated(text, [1,2,1])

    def test_distinct_pool_identities_compare_by_bytes(self):
        text = self.program('PUSH_STR a\nPUSH_STR b\nSTR_EQ\nASSERT\n'
                            'PUSH_STR a\nPUSH_STR b\nEQ\nASSERT\n'
                            'PUSH_STR a\nPUSH_STR b\nLE\nASSERT\n'
                            'PUSH_STR a\nPUSH_STR b\nGE\nASSERT\n')
        text = text.replace('.string b "a\\x00z"', '.string b "a\\x00x"')
        module = self.assemble(text)
        ir = self.work/'distinct.ll'
        self.run_cmd([ROOT/'obj/literal_string_aliases',module,ir])
        self.run_cmd(['lli','--entry-function=nano_entry',ir])
        obj, wasm = self.work/'distinct.o', self.work/'distinct.wasm'
        self.run_cmd(['llc','-mtriple=wasm32-unknown-unknown','-filetype=obj',ir,'-o',obj])
        self.run_cmd(['wasm-ld','--no-entry','--export=nano_entry','--fatal-warnings',obj,'-o',wasm])
        self.assertEqual(self.run_cmd(['wasmtime','run','--invoke','nano_entry',wasm]).stdout,'0\n')
        ir.write_text(ir.read_text()+'define i32 @main() { %r = call i32 @nano_entry()\n ret i32 %r\n}\n')
        exe = self.work/'distinct-native'
        self.run_cmd(['clang','-O2','-fsanitize=address,undefined',ir,'-o',exe])
        self.run_cmd([exe])

    def test_nonstring_string_operations_keep_runtime_errors(self):
        self.compare('PUSH_I64 1\nSTR_LEN\nPOP\n', trap=True)
        self.compare('PUSH_STR a\nPUSH_I64 1\nSTR_EQ\nPOP\n', trap=True)

    def test_string_handles_never_become_numeric_operands(self):
        for op,right in [('SUB','PUSH_I64 1'),('MUL','PUSH_I64 1'),
                         ('DIV','PUSH_I64 1'),('MOD','PUSH_I64 1'),('NEG',''),
                         ('I64_ADD','PUSH_I64 1'),('I64_EQ','PUSH_I64 1'),
                         ('F64_ADD','PUSH_F64 1'),('BOOL_NOT','')]:
            with self.subTest(op=op):
                body = 'PUSH_STR a\n'+(right+'\n' if right else '')+op+'\nPOP\n'
                if op in ('I64_ADD','I64_EQ','F64_ADD','BOOL_NOT'):
                    # Ordinary typed verification rejects known string operands
                    # before either translator can publish an executable.
                    source, module = self.work/'bad.nasm', self.work/'prior.nvm'
                    source.write_text(self.program(body))
                    module.write_bytes(b'previous')
                    result = self.run_cmd([ROOT/'bin/nanoisa','asm',source,'-o',module],success=False)
                    self.assertIn('operand is string', result.stderr)
                    self.assertEqual(module.read_bytes(), b'previous')
                else:
                    self.compare(body,trap=True)

    def test_managed_string_operations_and_conversion_refusals(self):
        for op in ('ADD','STR_CONCAT'):
            self.compare('PUSH_STR a\nPUSH_STR b\n'+op+'\nSTR_LEN\nPUSH_I64 6\nEQ\nASSERT\n')
        self.compare('PUSH_I64 1\nPUSH_I64 2\nADD\nPOP\n',
                     '.function unused 1 1 0 void 0\n.parameters unused string\nRET\n.end\n')
        cases = [self.program('PUSH_STR a\n'+op+'\nPOP\n') for op in ('CAST_STRING','CAST_FLOAT')]
        for text in cases:
            module = self.assemble(text)
            self.run_cmd([ROOT/'bin/nano_vm','--verify-only',module])
            for tool in ('nvm2llvm','nvm2wasm'):
                out = self.work/'previous'
                out.write_bytes(b'previous')
                self.run_cmd([ROOT/'bin'/tool,module,'-o',out],success=False)
                self.assertEqual(out.read_bytes(), b'previous')
        self.compare('PUSH_I64 1\nPUSH_I64 2\nADD\nPUSH_I64 3\nEQ\nASSERT\n')


if __name__ == '__main__':
    unittest.main()
