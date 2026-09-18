"""I exercise actual emitted ownership across VM, native LLVM and Wasm."""
import json
import os
import re
from pathlib import Path
import shlex
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]


class ManagedStrings(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix='nano-managed-emitted-')
        self.addCleanup(self.tmp.cleanup)
        self.work = Path(self.tmp.name)
        self.clang = ['clang']+shlex.split(os.environ.get('NMS_NATIVE_CLANG_FLAGS',''))

    def run_cmd(self, args, success=True):
        p = subprocess.run([str(x) for x in args], text=True, capture_output=True, timeout=45,
            env={**os.environ, 'ASAN_OPTIONS':'detect_leaks=1:abort_on_error=1'})
        self.assertEqual(p.returncode == 0, success, str(args)+'\n'+p.stdout+p.stderr)
        return p

    def program(self, body, suffix='', result='PUSH_I64 0\nRET\n'):
        return ('.string a "a\\x00z"\n.string empty ""\n.string double "a\\x00za\\x00z"\n'
                '.entry main\n.function main 0 4 0 int 1\n'+body+result+'.end\n'+suffix)

    def compile(self, text, vm_ok=True):
        assembly, module = self.work/'input.nasm', self.work/'input.nvm'
        assembly.write_text(text)
        self.run_cmd([ROOT/'bin/nanoisa','asm',assembly,'-o',module])
        self.run_cmd([ROOT/'bin/nano_vm',module],success=vm_ok)
        native, wasm_ir, wasm = self.work/'native.ll', self.work/'wasm.ll', self.work/'out.wasm'
        for target, ir in [('native',native),('wasm32',wasm_ir)]:
            self.run_cmd([ROOT/'bin/nvm2llvm',module,'--entry-name','nano_entry',
                          '--runtime-target',target,'-o',ir])
            self.run_cmd(['opt','-passes=verify','-disable-output',ir])
            self.assertIn('@nms_module_begin',ir.read_text())
            self.assertEqual(ir.read_text().count('call void @llvm.trap()'),1)
        self.run_cmd(['clang','--target=wasm32-unknown-unknown','-nostdlib',wasm_ir,
            '-Wl,--no-entry','-Wl,--max-memory=1048576','-Wl,--export=nano_entry',
            '-Wl,--export=nano_try_entry','-Wl,--export=nano_dispose',
            '-Wl,--export=nms_module_live_objects','-Wl,--export=nms_module_live_bytes','-o',wasm])
        return module,native,wasm

    def native_harness(self, ir, body, extra='', flags=()):
        source, exe = self.work/'harness.c', self.work/'harness'
        source.write_text('#include <stdint.h>\n#include <stdlib.h>\n'
            'extern uint64_t nano_try_entry(void),nms_module_live_objects(void),nms_module_live_bytes(void);\n'
            'extern int nano_dispose(void),nano_entry(void);\n'+extra+'\nint main(void){'+body+'}\n')
        # Clang does not retroactively mark input IR functions for ASan. I
        # explicitly instrument the emitted functions, then link their object
        # with a sanitizer-built harness and runtime interceptors.
        marked, instrumented, obj = self.work/'asan-input.ll', self.work/'asan.ll', self.work/'asan.o'
        marked.write_text(re.sub(r'^(define [^\n]+) \{', r'\1 sanitize_address {', ir.read_text(), flags=re.M))
        self.run_cmd(['opt','-passes=asan','-S',marked,'-o',instrumented])
        self.assertIn('__asan_report_',instrumented.read_text())
        self.run_cmd(['llc','-relocation-model=pic','-filetype=obj',instrumented,'-o',obj])
        self.run_cmd(self.clang+['-O1','-g','-fsanitize=address,undefined','-fno-sanitize-recover=all',
                                obj,source,*flags,'-o',exe])
        self.run_cmd([exe])

    def node(self, wasm, body):
        script = "const fs=require('fs');const m=new WebAssembly.Module(fs.readFileSync(process.argv[1]));if(WebAssembly.Module.imports(m).length)throw Error('imports');let e=new WebAssembly.Instance(m).exports;function check(v){if(!v)throw Error('managed assertion');}\n"+body
        self.run_cmd(['node','-e',script,wasm])

    def test_content_aliases_branches_loop_and_nested_calls(self):
        body = ('PUSH_STR a\nPUSH_STR a\nSTR_CONCAT\nDUP\nSTORE_LOCAL 0\n'
                'PUSH_STR double\nSTR_EQ\nASSERT\n'
                'LOAD_LOCAL 0\nPUSH_STR empty\nCALL outer\nSTORE_LOCAL 1\n'
                'PUSH_I64 0\nSTORE_LOCAL 2\nloop:\nLOAD_LOCAL 2\nPUSH_I64 2000\nLT\nJMP_FALSE done\n'
                'LOAD_LOCAL 1\nPUSH_STR empty\nADD\nSTORE_LOCAL 1\n'
                'LOAD_LOCAL 2\nPUSH_I64 1\nADD\nSTORE_LOCAL 2\nJMP loop\ndone:\n'
                'LOAD_LOCAL 1\nSTR_LEN\nPUSH_I64 6\nEQ\nASSERT\n'
                'PUSH_BOOL 0\nJMP_FALSE other\nLOAD_LOCAL 0\nJMP joined\nother:\nLOAD_LOCAL 1\n'
                'joined:\nDUP\nPUSH_STR double\nEQ\nASSERT\nASSERT\n'
                'LOAD_LOCAL 0\nLOAD_LOCAL 1\nSWAP\nEQ\nASSERT\n'
                'LOAD_LOCAL 1\nPUSH_I64 10\nCALL recur\nPOP\n')
        suffix = ('.function outer 2 2 0 string 1\n.parameters outer string string\n'
                  'LOAD_LOCAL 0\nLOAD_LOCAL 1\nCALL inner\nRET\n.end\n'
                  '.function inner 2 2 0 string 1\n.parameters inner string string\n'
                  'LOAD_LOCAL 0\nLOAD_LOCAL 1\nADD\n.end\n'
                  '.function recur 2 2 0 string 1\n.parameters recur string int\n'
                  'LOAD_LOCAL 1\nPUSH_I64 0\nLE\nJMP_FALSE again\nLOAD_LOCAL 0\nRET\nagain:\n'
                  'LOAD_LOCAL 0\nPUSH_STR empty\nADD\nLOAD_LOCAL 1\nPUSH_I64 1\nSUB\nCALL recur\nRET\n.end\n')
        module,ir,wasm=self.compile(self.program(body,suffix))
        self.native_harness(ir,'for(int i=0;i<20;i++){if(nano_try_entry()||nms_module_live_objects())return 1;}return nano_dispose();')
        self.node(wasm,"check(e.nano_try_entry()===0n);let pages=e.memory.buffer.byteLength;for(let i=0;i<20;i++){check(e.nano_try_entry()===0n);check(e.nms_module_live_objects()===0n);}check(e.memory.buffer.byteLength===pages);check(e.nano_dispose()===0);")
        published=self.work/'published.wasm'
        self.run_cmd([ROOT/'bin/nvm2wasm',module,'-o',published])
        self.assertEqual(self.run_cmd(['wasmtime','run','--invoke','nano_entry',published]).stdout,'0\n')
        self.run_cmd([ROOT/'bin/nvm2llvm',module,'-o',self.work/'main.ll'])
        self.run_cmd(self.clang+['-fsanitize=address,undefined',self.work/'main.ll','-o',self.work/'main'])
        self.run_cmd([self.work/'main'])

    def test_managed_profile_preserves_numeric_tags_and_total_boundaries(self):
        body='PUSH_STR a\nPUSH_STR empty\nSTR_CONCAT\nSTORE_LOCAL 0\n'
        for left,op,right,expected,tag in [
            ('PUSH_I64 9223372036854775807','ADD','PUSH_I64 1','PUSH_I64 -9223372036854775808',1),
            ('PUSH_I64 -9223372036854775808','DIV','PUSH_I64 -1','PUSH_I64 -9223372036854775808',1),
            ('PUSH_I64 7','MOD','PUSH_I64 0','PUSH_I64 0',1),
            ('PUSH_I64 2','ADD','PUSH_F64 0.5','PUSH_F64 2.5',3),
            ('PUSH_F64 nan','DIV','PUSH_F64 -0.0','PUSH_F64 0.0',3),
            ('ENUM_VAL 0 3','ADD','PUSH_F64 0.5','PUSH_F64 3.5',3),
            ('ENUM_VAL 0 3','I64_ADD','PUSH_I64 2','PUSH_I64 5',1)]:
            body += f'{left}\n{right}\n{op}\nDUP\nTYPE_CHECK {tag}\nASSERT\n{expected}\nEQ\nASSERT\n'
        _,ir,wasm=self.compile('.types 0 1 0\n'+self.program(body))
        self.native_harness(ir,'if(nano_try_entry()||nms_module_live_objects())return 1;return nano_dispose();')
        self.node(wasm,"check(e.nano_try_entry()===0n);check(e.nms_module_live_objects()===0n);check(e.nano_dispose()===0);")

    def test_global_ownership_reentry_initializer_and_terminal_disposal(self):
        body=('LOAD_GLOBAL 0\nTYPE_CHECK 0\nJMP_FALSE exists\nPUSH_STR empty\nSTORE_GLOBAL 0\nexists:\n'
              'LOAD_GLOBAL 0\nPUSH_STR a\nADD\nSTORE_GLOBAL 0\nLOAD_GLOBAL 0\nSTR_LEN\nRET\n')
        suffix='.function __init__ 0 0 0 string 1\nPUSH_STR a\nPUSH_STR a\nSTR_CONCAT\nRET\n.end\n'
        # VM process reports its nonzero entry result as an exit code.
        _,ir,wasm=self.compile(self.program(body,suffix,result=''),vm_ok=False)
        self.native_harness(ir,'if(nano_try_entry()!=3||nano_try_entry()!=6||nms_module_live_objects()!=1||nms_module_live_bytes()!=6)return 1;if(nano_dispose()||nms_module_live_objects()||nano_dispose())return 2;return nano_try_entry()!=((uint64_t)5<<32);')
        self.node(wasm,"check(e.nano_try_entry()===3n);check(e.nano_try_entry()===6n);check(e.nms_module_live_objects()===1n);check(e.nms_module_live_bytes()===6n);check(e.nano_dispose()===0);check(e.nms_module_live_objects()===0n);check(e.nano_try_entry()===(5n<<32n));e=new WebAssembly.Instance(m).exports;check(e.nano_dispose()===0);check(e.nano_dispose()===0);check(e.nano_try_entry()===(5n<<32n));")

    def test_callee_errors_unwind_all_frames_and_preserve_prior_global_writes(self):
        for error,status in [('PUSH_BOOL 0\nASSERT\n',2),('LOAD_LOCAL 0\nPUSH_I64 1\nADD\nPOP\n',1),
                             ('LOAD_LOCAL 0\nNEG\nPOP\n',1),('LOAD_LOCAL 0\nPUSH_BOOL 1\nMOD\nPOP\n',1),
                             ('LOAD_GLOBAL 0\nI64_NEG\nPOP\n',1),
                             ('LOAD_GLOBAL 0\nPUSH_F64 1.0\nF64_ADD\nPOP\n',1)]:
            with self.subTest(error=error):
                body=('PUSH_STR a\nPUSH_STR a\nSTR_CONCAT\nDUP\nSTORE_GLOBAL 0\nDUP\nSTORE_LOCAL 0\n'
                      'LOAD_LOCAL 0\nCALL fail\nPOP\n')
                suffix=('.function fail 1 2 0 void 0\n.parameters fail string\n'
                        'LOAD_LOCAL 0\nPUSH_STR a\nSTR_CONCAT\nSTORE_LOCAL 1\n'+error+'RET\n.end\n')
                _,ir,wasm=self.compile(self.program(body,suffix),vm_ok=False)
                self.native_harness(ir,f'for(int i=0;i<20;i++){{if(nano_try_entry()!=((uint64_t){status}<<32)||nms_module_live_objects()!=1||nms_module_live_bytes()!=6)return 1;}}if(nano_dispose()||nms_module_live_objects())return 2;return 0;')
                self.node(wasm,f"for(let i=0;i<20;i++){{check(e.nano_try_entry()===({status}n<<32n));check(e.nms_module_live_objects()===1n);check(e.nms_module_live_bytes()===6n);}}check(e.nano_dispose()===0);check(e.nms_module_live_objects()===0n);")

    def test_dynamic_return_tag_errors_clean_explicit_and_implicit_frames(self):
        for ending in ('RET\n',''):
            body='PUSH_STR a\nPUSH_STR a\nSTR_CONCAT\nSTORE_GLOBAL 0\nCALL bad\nPOP\n'
            suffix='.function bad 0 0 0 int 1\nLOAD_GLOBAL 0\n'+ending+'.end\n'
            _,ir,wasm=self.compile(self.program(body,suffix),vm_ok=False)
            self.native_harness(ir,'if(nano_try_entry()!=((uint64_t)1<<32)||nms_module_live_objects()!=1)return 1;if(nano_dispose()||nms_module_live_objects())return 2;return 0;')
            self.node(wasm,"check(e.nano_try_entry()===(1n<<32n));check(e.nms_module_live_objects()===1n);check(e.nano_dispose()===0);check(e.nms_module_live_objects()===0n);")

    def test_native_byte_and_descriptor_allocation_failure_are_recoverable(self):
        _,ir,_=self.compile(self.program('PUSH_STR a\nPUSH_STR a\nSTR_CONCAT\nPOP\n'))
        for fail in (0,1):
            extra='static long budget=-1;extern void *__real_malloc(size_t);void *__wrap_malloc(size_t n){if(!budget)return 0;if(budget>0)--budget;return __real_malloc(n);}'
            self.native_harness(ir,f'budget={fail};if(nano_try_entry()!=((uint64_t)3<<32)||nms_module_live_objects())return 1;budget=-1;if(nano_try_entry()||nms_module_live_objects())return 2;return nano_dispose();',extra,['-Wl,--wrap=malloc'])

    def test_allocation_failure_unwinds_callee_and_preserves_global_owner(self):
        body=('PUSH_STR a\nPUSH_STR a\nSTR_CONCAT\nDUP\nSTORE_GLOBAL 0\nSTORE_LOCAL 0\n'
              'LOAD_LOCAL 0\nCALL append\nPOP\n')
        suffix=('.function append 1 1 0 string 1\n.parameters append string\n'
                'LOAD_LOCAL 0\nPUSH_STR a\nSTR_CONCAT\nRET\n.end\n')
        _,ir,_=self.compile(self.program(body,suffix))
        extra='static long budget=-1;extern void *__real_malloc(size_t);void *__wrap_malloc(size_t n){if(!budget)return 0;if(budget>0)--budget;return __real_malloc(n);}'
        self.native_harness(ir,'budget=2;if(nano_try_entry()!=((uint64_t)3<<32)||nms_module_live_objects()!=1||nms_module_live_bytes()!=6)return 1;budget=-1;if(nano_try_entry()||nms_module_live_objects()!=1)return 2;if(nano_dispose()||nms_module_live_objects())return 3;return 0;',extra,['-Wl,--wrap=malloc'])

    def test_wasm_memory_cap_failure_cleans_frames_and_reuses_storage(self):
        body=('PUSH_STR a\nPUSH_STR empty\nSTR_CONCAT\nSTORE_LOCAL 0\n'
              'PUSH_I64 0\nSTORE_LOCAL 1\nloop:\nLOAD_LOCAL 1\nPUSH_I64 19\nLT\nJMP_FALSE done\n'
              'LOAD_LOCAL 0\nDUP\nSTR_CONCAT\nSTORE_LOCAL 0\n'
              'LOAD_LOCAL 1\nPUSH_I64 1\nADD\nSTORE_LOCAL 1\nJMP loop\ndone:\n')
        _,ir,wasm=self.compile(self.program(body))
        self.native_harness(ir,'if(nano_try_entry()||nms_module_live_objects())return 1;return nano_dispose();')
        self.node(wasm,"check(e.nano_try_entry()===(3n<<32n));check(e.nms_module_live_objects()===0n);let size=e.memory.buffer.byteLength;for(let i=0;i<5;i++){check(e.nano_try_entry()===(3n<<32n));check(e.nms_module_live_objects()===0n);check(e.memory.buffer.byteLength===size);}check(e.nano_dispose()===0);")

    def test_substring_bytes_indices_aliases_and_reentry(self):
        body = ''
        for start, length, expected in [(0,3,'a'),(0,0,'empty'),(3,4,'empty'),
                                         (1,20,'tail'),(4294967296,3,'a'),(-1,3,'empty')]:
            body += (f'PUSH_STR a\nPUSH_I64 {start}\nPUSH_I64 {length}\nSTR_SUBSTR\n'
                     f'PUSH_STR {expected}\nSTR_EQ\nASSERT\n')
        body += ('PUSH_STR a\nPUSH_STR empty\nSTR_CONCAT\nSTORE_GLOBAL 0\n'
                 'LOAD_GLOBAL 0\nLOAD_GLOBAL 0\nPUSH_I64 3\nCALL slice\n'
                 'PUSH_STR a\nEQ\nASSERT\n'
                 'LOAD_GLOBAL 0\nPUSH_I64 0\nLOAD_GLOBAL 0\nCALL empty_slice\n'
                 'DUP\nASSERT\nPUSH_STR empty\nEQ\nASSERT\n')
        suffix = ('.function slice 3 3 0 string 1\n.parameters slice string string int\nLOAD_LOCAL 0\nLOAD_LOCAL 1\n'
                  'LOAD_LOCAL 2\nSTR_SUBSTR\nRET\n.end\n'
                  '.function empty_slice 3 3 0 string 1\n.parameters empty_slice string int string\n'
                  'LOAD_LOCAL 0\nLOAD_LOCAL 1\nLOAD_LOCAL 2\nSTR_SUBSTR\nRET\n.end\n')
        text = '.string tail "\\x00z"\n'+self.program(body,suffix)
        _, ir, wasm = self.compile(text)
        self.native_harness(ir,'for(int i=0;i<30;i++){uint64_t s=nano_try_entry();if(s||nms_module_live_objects()!=1||nms_module_live_bytes()!=3){fprintf(stderr,"round %d status %llu objects %llu bytes %llu\\n",i,(unsigned long long)s,(unsigned long long)nms_module_live_objects(),(unsigned long long)nms_module_live_bytes());return 1;}}return nano_dispose();','#include <stdio.h>')
        self.node(wasm,'for(let i=0;i<30;i++){check(e.nano_try_entry()===0n);check(e.nms_module_live_objects()===1n);check(e.nms_module_live_bytes()===3n);}check(e.nano_dispose()===0);check(e.nms_module_live_objects()===0n);')

    def test_substring_allocation_failure_and_type_error_cleanup(self):
        body = ('PUSH_STR a\nPUSH_STR empty\nSTR_CONCAT\nDUP\nSTORE_GLOBAL 0\n'
                'PUSH_I64 1\nPUSH_I64 2\nCALL slice\nPOP\n')
        suffix = ('.function slice 3 3 0 string 1\n.parameters slice string int int\n'
                  'LOAD_LOCAL 0\nLOAD_LOCAL 1\nLOAD_LOCAL 2\nSTR_SUBSTR\nRET\n.end\n')
        _, ir, wasm = self.compile(self.program(body,suffix))
        extra = 'static long budget=-1;extern void *__real_malloc(size_t);void *__wrap_malloc(size_t n){if(!budget)return 0;if(budget>0)--budget;return __real_malloc(n);}'
        self.native_harness(ir,'budget=2;if(nano_try_entry()!=((uint64_t)3<<32)||nms_module_live_objects()!=1||nms_module_live_bytes()!=3)return 1;budget=-1;if(nano_try_entry()||nms_module_live_objects()!=1)return 2;return nano_dispose();',extra,['-Wl,--wrap=malloc'])
        self.node(wasm,'check(e.nano_try_entry()===0n);check(e.nms_module_live_objects()===1n);check(e.nano_dispose()===0);')
        bad = ('PUSH_STR a\nPUSH_STR empty\nSTR_CONCAT\nSTORE_GLOBAL 0\n'
               'PUSH_I64 7\nSTORE_GLOBAL 1\nLOAD_GLOBAL 1\nLOAD_GLOBAL 0\nLOAD_GLOBAL 0\nSTR_SUBSTR\nPOP\n')
        _, ir, wasm = self.compile(self.program(bad),vm_ok=False)
        self.native_harness(ir,'for(int i=0;i<10;i++)if(nano_try_entry()!=((uint64_t)1<<32)||nms_module_live_objects()!=1)return 1;return nano_dispose();')
        self.node(wasm,'for(let i=0;i<10;i++){check(e.nano_try_entry()===(1n<<32n));check(e.nms_module_live_objects()===1n);}check(e.nano_dispose()===0);')

    def test_unsupported_string_cast_and_reserved_entry_refusals_preserve_output(self):
        for body in ('PUSH_STR a\nPUSH_STR empty\nSTR_SPLIT\nARR_POP\nPOP\n',
                     'ARR_NEW 1\nPOP\n'):
            asm,mod=self.work/'refuse.nasm',self.work/'refuse.nvm'
            asm.write_text(self.program(body))
            self.run_cmd([ROOT/'bin/nanoisa','asm',asm,'-o',mod])
            for tool in ('nvm2llvm','nvm2wasm'):
                out=self.work/'previous';out.write_bytes(b'previous')
                self.run_cmd([ROOT/'bin'/tool,mod,'-o',out],success=False)
                self.assertEqual(out.read_bytes(),b'previous')
        module,_,_=self.compile(self.program('PUSH_STR a\nPUSH_STR a\nSTR_CONCAT\nPOP\n'))
        for name in ('nano_dispose','nano_try_entry','nano_runtime_hidden'):
            out=self.work/'previous';out.write_bytes(b'previous')
            self.run_cmd([ROOT/'bin/nvm2llvm',module,'--entry-name',name,'-o',out],success=False)
            self.assertEqual(out.read_bytes(),b'previous')


if __name__=='__main__':
    unittest.main()
