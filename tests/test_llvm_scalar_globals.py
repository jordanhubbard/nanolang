"""I retain scalar global tags and initializer lifetime across target instances."""
import subprocess
import unittest
from tests import test_llvm_generic_numeric as numeric

ROOT = numeric.ROOT


class ScalarGlobals(unittest.TestCase):
    setUp = numeric.GenericNumeric.setUp
    run_cmd = numeric.GenericNumeric.run_cmd
    program = numeric.GenericNumeric.program
    compare = numeric.GenericNumeric.compare

    def assemble(self, text):
        source, module = self.work/'globals.nasm', self.work/'globals.nvm'
        source.write_text(text)
        self.run_cmd([ROOT/'bin/nanoisa','asm',source,'-o',module])
        return module

    def test_initial_void_and_highest_slot(self):
        self.compare('LOAD_GLOBAL 0\nTYPE_CHECK 0\nASSERT\n'
                     'LOAD_GLOBAL 4095\nTYPE_CHECK 0\nASSERT\n'
                     'PUSH_I64 42\nSTORE_GLOBAL 4095\nLOAD_GLOBAL 4095\nPUSH_I64 42\nEQ\nASSERT\n'
                     'LOAD_GLOBAL 0\nTYPE_CHECK 0\nASSERT\n')
        self.assertIn('@globals = internal global [4096 x %V]', (self.work/'out.ll').read_text())

    def test_unreached_function_still_contributes_storage(self):
        self.compare('LOAD_GLOBAL 0\nTYPE_CHECK 0\nASSERT\n',
                     '.function unused 0 0 0 void 0\nLOAD_GLOBAL 4095\nPOP\nRET\n.end\n')
        self.assertIn('@globals = internal global [4096 x %V]', (self.work/'out.ll').read_text())

    def test_exact_tags_and_last_write(self):
        body = ''
        for value,tag in [('PUSH_I64 -7',1),('PUSH_U8 255',2),('PUSH_F64 -3.5',3),
                          ('PUSH_BOOL 1',4),('PUSH_VOID',0)]:
            body += f'{value}\nSTORE_GLOBAL 3\nLOAD_GLOBAL 3\nDUP\nTYPE_CHECK {tag}\nASSERT\n{value}\nEQ\nASSERT\n'
        body += 'PUSH_F64 nan\nSTORE_GLOBAL 3\nLOAD_GLOBAL 3\nDUP\nNE\nASSERT\n'
        self.compare(body)

    def test_functions_share_branch_selected_writes(self):
        suffix = ('.function setter 1 1 0 void 0\n.parameters setter float\n'
                  'LOAD_LOCAL 0\nSTORE_GLOBAL 0\nRET\n.end\n'
                  '.function getter 0 0 0 float 1\nLOAD_GLOBAL 0\nRET\n.end\n')
        self.compare('PUSH_F64 1.0\nCALL setter\nPUSH_BOOL 0\nJMP_FALSE taken\n'
                     'PUSH_F64 9.0\nCALL setter\nJMP joined\ntaken:\n'
                     'PUSH_F64 2.5\nCALL setter\njoined:\nCALL getter\n'
                     'DUP\nTYPE_CHECK 3\nASSERT\nPUSH_F64 2.5\nEQ\nASSERT\n',suffix)

    def test_initializer_precedes_entry_and_discards_result(self):
        for tag,value,result in [('void','',0),('int','PUSH_I64 99\n',1),
                                 ('float','PUSH_F64 3.5\n',1),('bool','PUSH_BOOL 1\n',1),
                                 ('u8','PUSH_U8 255\n',1)]:
            with self.subTest(tag=tag):
                suffix = (f'.function __init__ 0 0 0 {tag} {result}\n'
                          'LOAD_GLOBAL 1\nTYPE_CHECK 0\nASSERT\n'
                          'PUSH_U8 7\nSTORE_GLOBAL 1\n'+value+'RET\n.end\n')
                self.compare('LOAD_GLOBAL 1\nDUP\nTYPE_CHECK 2\nASSERT\nPUSH_U8 7\nEQ\nASSERT\n',suffix)

    def test_initializer_failure_prevents_entry(self):
        self.compare('PUSH_I64 7\nSTORE_GLOBAL 0\n',
                     '.function __init__ 0 0 0 void 0\nPUSH_BOOL 0\nASSERT\nRET\n.end\n',trap=True)

    def repeated(self, text, expected):
        module = self.assemble(text)
        vm = self.run_cmd([ROOT/'obj/scalar_global_lifetime',module])
        self.assertEqual([int(x) for x in vm.stdout.splitlines()], expected)
        wasm = self.work/'repeat.wasm'
        self.run_cmd([ROOT/'bin/nvm2wasm',module,'-o',wasm])
        script = ('const fs=require("fs"); const m=new WebAssembly.Module(fs.readFileSync(process.argv[1]));'
                  'const a=new WebAssembly.Instance(m).exports; const b=new WebAssembly.Instance(m).exports;'
                  'console.log([a.nano_entry(),a.nano_entry(),b.nano_entry()].join("\\n"));')
        result = self.run_cmd(['node','-e',script,wasm])
        self.assertEqual([int(x) for x in result.stdout.splitlines()],expected)
        # LLVM/native entry calls observe one module instance too.
        ir = self.work/'repeat.ll'
        self.run_cmd([ROOT/'bin/nvm2llvm',module,'--entry-name','nano_entry','-o',ir])
        harness = ('define i32 @main() {\n %a = call i32 @nano_entry()\n %b = call i32 @nano_entry()\n'
                   f' %ca = icmp eq i32 %a, {expected[0]}\n %cb = icmp eq i32 %b, {expected[1]}\n'
                   ' %both = and i1 %ca, %cb\n %bad = xor i1 %both, true\n'
                   ' %status = zext i1 %bad to i32\n ret i32 %status\n}\n')
        ir.write_text(ir.read_text()+harness)
        self.run_cmd(['lli',ir])
        exe = self.work/'repeat-native'
        self.run_cmd(['clang','-O2','-fsanitize=address,undefined',ir,'-o',exe])
        self.run_cmd([exe])

    def increment(self):
        return ('LOAD_GLOBAL 0\nTYPE_CHECK 0\nJMP_FALSE existing\nPUSH_I64 0\nSTORE_GLOBAL 0\n'
                'existing:\nLOAD_GLOBAL 0\nPUSH_I64 1\nADD\nDUP\nSTORE_GLOBAL 0\n')

    def test_repeated_entry_retains_globals(self):
        self.repeated('.entry main\n.function main 0 0 0 int 1\n'+self.increment()+'RET\n.end\n',[1,2,1])

    def test_repeated_entry_reruns_initializer(self):
        self.repeated('.entry main\n.function main 0 0 0 int 1\nLOAD_GLOBAL 0\nRET\n.end\n'
                      '.function __init__ 0 0 0 void 0\n'+self.increment()+'POP\nRET\n.end\n',[1,2,1])

    def test_initializer_equal_to_entry_runs_twice(self):
        self.repeated('.entry __init__\n.function __init__ 0 0 0 int 1\n'+self.increment()+'RET\n.end\n',[2,4,2])

    def test_first_initializer_is_selected(self):
        module = self.assemble(self.program('LOAD_GLOBAL 0\nPUSH_I64 1\nEQ\nASSERT\n',
                     '.function __init__ 0 0 0 void 0\nPUSH_I64 1\nSTORE_GLOBAL 0\nRET\n.end\n'
                     '.function later 1 1 0 void 0\n.parameters later int\nPUSH_BOOL 0\nASSERT\nRET\n.end\n'))
        renamed = self.work/'display-names.nvm'
        self.run_cmd([ROOT/'obj/scalar_global_lifetime',module,renamed])
        self.run_cmd([ROOT/'bin/nano_vm',renamed])
        ir, exe, wasm = (self.work/n for n in ('first.ll','first-native','first.wasm'))
        self.run_cmd([ROOT/'bin/nvm2llvm',renamed,'-o',ir])
        self.run_cmd(['lli',ir])
        self.run_cmd(['clang','-O2','-fsanitize=address,undefined',ir,'-o',exe])
        self.run_cmd([exe])
        self.run_cmd([ROOT/'bin/nvm2wasm',renamed,'-o',wasm])
        self.assertEqual(self.run_cmd(['wasmtime','run','--invoke','nano_entry',wasm]).stdout,'0\n')

    def test_initializer_and_heap_refusals_preserve_output(self):
        cases = [self.program('', '.function __init__ 1 1 0 void 0\n.parameters __init__ int\nRET\n.end\n'),
                 '.string text "ordinary"\n'+self.program('PUSH_STR text\nSTR_TO_UPPER\nSTORE_GLOBAL 0\n'),
                 '.types 1 0 0\n'+self.program(''),
                 '.import "" "get_argc" int\n'+self.program('')]
        for text in cases:
            module = self.assemble(text)
            for tool in ('nvm2llvm','nvm2wasm'):
                out = self.work/'previous'
                out.write_bytes(b'previous artifact')
                self.run_cmd([ROOT/'bin'/tool,module,'-o',out],success=False)
                self.assertEqual(out.read_bytes(),b'previous artifact')


if __name__ == '__main__':
    unittest.main()
