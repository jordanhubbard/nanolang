"""I retain exact void/scalar tags at native control-flow joins."""
import os
from pathlib import Path
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
NVM2C = Path(os.environ.get('NANO_JOIN_NVM2C', ROOT/'bin/nvm2c'))


class NativeScalarJoins(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix='nano-scalar-join-')
        self.addCleanup(self.temp.cleanup)
        self.work = Path(self.temp.name)

    def checked(self, args):
        result = subprocess.run(list(map(str, args)), capture_output=True, text=True, timeout=60)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        return result

    def module(self, source):
        assembly, module = self.work/'input.nasm', self.work/'input.nvm'
        assembly.write_text(source)
        self.checked([ROOT/'bin/nanoisa', 'asm', assembly, '-o', module])
        self.checked([ROOT/'bin/nano_vm', '--verify-only', module])
        return module

    def compare(self, source):
        module = self.module(source)
        vm = self.checked([ROOT/'bin/nano_vm', module])
        c, binary = self.work/'program.c', self.work/'program'
        self.checked([NVM2C, module, '-o', c])
        flags = os.environ.get('NANO_JOIN_CFLAGS', '').split()
        self.checked([os.environ.get('CC', 'cc'), '-std=c11', '-O1', '-Wall', '-Wextra', '-Werror',
                      *flags, c, '-lm', '-o', binary])
        native = self.checked([binary])
        self.assertEqual(native.stdout, vm.stdout)

    def program(self, body, functions=''):
        return '.entry main\n.function main 0 3 0 int 1\n'+body+'PUSH_I64 0\nRET\n.end\n'+functions

    def choose(self, instruction, tag, reverse=False):
        branch = ('LOAD_LOCAL 0\nJMP_FALSE missing\n'+instruction+'\nJMP join\nmissing:\nPUSH_VOID\n'
                  if reverse else 'LOAD_LOCAL 0\nJMP_TRUE present\nPUSH_VOID\nJMP join\npresent:\n'+instruction+'\n')
        return ('.function choose 1 2 0 bool 1\n.parameters choose bool\n'+branch+
                'join:\nDUP\nSTORE_LOCAL 1\nTYPE_CHECK 0\nLOAD_LOCAL 0\nBOOL_NOT\nEQ\nASSERT\n'
                f'LOAD_LOCAL 1\nDUP\nTYPE_CHECK {tag}\nLOAD_LOCAL 0\nEQ\nASSERT\nCAST_BOOL\nRET\n.end\n')

    def test_forward_scalar_joins_both_orders_and_aliases(self):
        for instruction, tag, truth in [('PUSH_I64 7',1,True), ('PUSH_I64 0',1,False),
                ('PUSH_BOOL 1',4,True), ('PUSH_BOOL 0',4,False),
                ('PUSH_F64 -0.0',3,False), ('PUSH_F64 1.25',3,True),
                ('PUSH_U8 255',2,True), ('PUSH_U8 0',2,False)]:
            for reverse in (False,True):
                with self.subTest(instruction=instruction, reverse=reverse):
                    body = ('PUSH_BOOL 0\nCALL choose\nBOOL_NOT\nASSERT\n'
                            'PUSH_BOOL 1\nCALL choose\n'+('' if truth else 'BOOL_NOT\n')+'ASSERT\n')
                    self.compare(self.program(body,self.choose(instruction,tag,reverse)))

    def test_explicit_void_local_retains_join_provenance(self):
        function = self.choose('PUSH_I64 7', 1).replace(
            '.parameters choose bool\n', '.parameters choose bool\nPUSH_VOID\nSTORE_LOCAL 1\n')
        function = function.replace('JMP_TRUE present\nPUSH_VOID', 'JMP_TRUE present\nLOAD_LOCAL 1')
        self.compare(self.program('PUSH_BOOL 0\nCALL choose\nBOOL_NOT\nASSERT\n'
                                  'PUSH_BOOL 1\nCALL choose\nASSERT\n', function))

    def test_loop_header_widens_before_emission(self):
        # First entry is concrete; the backedge carries explicit void.
        body = ('PUSH_I64 0\nSTORE_LOCAL 0\nPUSH_I64 7\nloop:\n'
                'DUP\nTYPE_CHECK 0\nLOAD_LOCAL 0\nPUSH_I64 0\nI64_GT_S\nEQ\nASSERT\n'
                'POP\nPUSH_VOID\nLOAD_LOCAL 0\nPUSH_I64 1\nI64_ADD\nSTORE_LOCAL 0\n'
                'LOAD_LOCAL 0\nPUSH_I64 3\nI64_LT_S\nJMP_TRUE loop\n'
                'TYPE_CHECK 0\nASSERT\n')
        self.compare(self.program(body))

    def test_negative_zero_bits_survive_boxing(self):
        function = self.choose('PUSH_F64 -0.0', 3).replace(
            'CAST_BOOL\nRET',
            'LOAD_LOCAL 0\nJMP_FALSE done\nDUP\nCAST_STRING\nPUSH_STR negative_zero\nEQ\nASSERT\ndone:\nCAST_BOOL\nRET')
        self.compare('.string negative_zero "-0"\n'+self.program(
            'PUSH_BOOL 0\nCALL choose\nBOOL_NOT\nASSERT\n'
            'PUSH_BOOL 1\nCALL choose\nBOOL_NOT\nASSERT\n', function))

    def test_parallel_edges_keep_two_distinct_tags(self):
        functions = ('.function choose 1 1 0 bool 1\n.parameters choose bool\n'
            'LOAD_LOCAL 0\nJMP_TRUE present\nPUSH_VOID\nPUSH_BOOL 1\nJMP join\npresent:\n'
            'PUSH_I64 7\nPUSH_VOID\njoin:\nTYPE_CHECK 0\nLOAD_LOCAL 0\nEQ\nASSERT\n'
            'TYPE_CHECK 0\nLOAD_LOCAL 0\nBOOL_NOT\nEQ\nRET\n.end\n')
        self.compare(self.program('PUSH_BOOL 0\nCALL choose\nASSERT\nPUSH_BOOL 1\nCALL choose\nASSERT\n', functions))

    def test_nested_join_preserves_provenance(self):
        functions = ('.function choose 1 1 0 bool 1\n.parameters choose bool\n'
            'LOAD_LOCAL 0\nJMP_TRUE outer\nPUSH_VOID\nJMP end\nouter:\n'
            'LOAD_LOCAL 0\nJMP_TRUE present\nPUSH_VOID\nJMP inner\npresent:\nPUSH_F64 -0.0\n'
            'inner:\nJMP end\nend:\nCAST_BOOL\nBOOL_NOT\nRET\n.end\n')
        self.compare(self.program('PUSH_BOOL 0\nCALL choose\nASSERT\nPUSH_BOOL 1\nCALL choose\nASSERT\n', functions))

    def test_heap_and_unrelated_union_still_refuse_publication(self):
        for left,right in [('PUSH_VOID','ARR_NEW 1'), ('ARR_NEW 1\nSTORE_GLOBAL 0\nLOAD_GLOBAL 0','PUSH_I64 1'),
                           ('PUSH_BOOL 1','PUSH_I64 1'),
                           ('PUSH_U8 1','PUSH_I64 1')]:
            with self.subTest(left=left,right=right):
                source=self.program('PUSH_BOOL 0\nJMP_TRUE right\n'+left+'\nJMP join\nright:\n'+right+'\njoin:\nPOP\n')
                module=self.module(source)
                target=self.work/'retained.c'
                target.write_text('previous')
                result=subprocess.run([NVM2C,module,'-o',target],capture_output=True,text=True,timeout=60)
                self.assertNotEqual(result.returncode,0)
                self.assertEqual(target.read_text(),'previous')


if __name__ == '__main__':
    unittest.main()
