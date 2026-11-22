"""I preserve the VM CAST_STRING contract for typed native non-scalar carriers."""
import json
import os
from pathlib import Path
import shlex
import unittest
from tests import test_native_regex_host_abi as abi
from tests.native_process import native_link_flags
from tests import nanolang_translator_routes as routes

ROOT=abi.ROOT


class NativeNonscalarCast(unittest.TestCase):
    command=abi.NativeRegexHostAbi.command
    directory=abi.NativeRegexHostAbi.directory

    def test_raw_tags_match_actual_vm_text(self):
        cases=[('void','PUSH_VOID',''),('opaque','OPAQUE_NULL',''),
               ('int','PUSH_I64 -42','-42'),('byte','PUSH_U8 255','255'),
               ('float','PUSH_F64 2.5','2.5'),('whole-float','PUSH_F64 1.0','1'),
               ('true','PUSH_BOOL 1','true'),('false','PUSH_BOOL 0','false'),
               ('string','PUSH_STR kept','kept'),
               ('record','PUSH_I64 7\nAGG_PACK 0 0 0 1',''),
               ('records','PUSH_I64 7\nAGG_PACK 0 0 0 1\nARR_LITERAL 8 1',''),
               ('function','FUNCREF leaf',''),('closure','CLOSURE_NEW leaf 0','')]
        for tag,value in [(1,'PUSH_I64 7'),(2,'PUSH_U8 255'),(3,'PUSH_F64 2.5'),
                          (4,'PUSH_BOOL 1'),(5,'PUSH_STR kept')]:
            cases.extend([(str(tag)+'-empty',f'ARR_LITERAL {tag} 0',''),
                          (str(tag)+'-values',value+f'\nARR_LITERAL {tag} 1','')])
        cases.append(('nested','PUSH_STR kept\nARR_LITERAL 5 1\nARR_LITERAL 7 1',''))
        for key in [1,5]:
            for value in [1,5]:cases.append((f'map-{key}-{value}',f'HM_NEW {key} {value}',''))
        translator=Path(os.environ.get('NANOLANG_NVM2C_TEST_TOOL',ROOT/'bin/nvm2c'))
        for label,body,expected in cases:
            with self.subTest(carrier=label):
                work=self.directory('nonscalar-cast-'+label+'-')
                assembly,module,generated,native=[work/n for n in ['input.nasm','input.nvm','input.c','input']]
                assembly.write_text('.entry main\n.string kept "kept"\n.function main 0 0 0 int 1\n'+
                                    body+'\nCAST_STRING\nPRINTLN\nPUSH_I64 0\nRET\n.end\n'
                                    '.function leaf 0 0 0 int 1\nPUSH_I64 0\nRET\n.end\n')
                self.command(work,'assemble',[ROOT/'bin/nanoisa','asm',assembly,'-o',module])
                self.command(work,'verify',[ROOT/'bin/nano_vm','--verify-only',module])
                oracle=self.command(work,'vm',[ROOT/'bin/nano_vm',module])
                self.assertEqual((oracle.stdout,oracle.stderr),(expected+'\n',''))
                self.command(work,'translate',[translator,module,'-o',generated],build=True)
                self.command(work,'build',[*shlex.split(os.environ.get('CC','cc')),'-std=c11','-O1','-g',
                    '-Wall','-Wextra','-Werror','-fsanitize=address,undefined','-fno-sanitize-recover=all',
                    generated,ROOT/'bin/nano_aot_runtime.o','-lm',*native_link_flags(),'-o',native],build=True)
                result=self.command(work,'native',[native])
                self.assertEqual((result.stdout,result.stderr),(oracle.stdout,''))
        self.assertEqual(len(cases),28)

    def test_zero_opaque_tag_and_declared_foreign_argument(self):
        work=self.directory('opaque-null-contract-')
        assembly,module,generated,native=[work/n for n in ['input.nasm','input.nvm','input.c','input']]
        assembly.write_text('''.entry main
.import "" "nl_regex_free" void opaque
.function main 0 1 0 int 1
OPAQUE_NULL
STORE_LOCAL 0
LOAD_LOCAL 0
DUP
TYPE_CHECK 14
ASSERT
CALL_EXTERN 0
PUSH_I64 0
TYPE_CHECK 14
BOOL_NOT
ASSERT
PUSH_VOID
TYPE_CHECK 14
BOOL_NOT
ASSERT
PUSH_I64 0
RET
.end
''')
        self.command(work,'assemble',[ROOT/'bin/nanoisa','asm',assembly,'-o',module])
        self.command(work,'verify',[ROOT/'bin/nano_vm','--verify-only',module])
        self.command(work,'vm',[ROOT/'bin/nano_vm',module])
        translator=Path(os.environ.get('NANOLANG_NVM2C_TEST_TOOL',ROOT/'bin/nvm2c'))
        self.command(work,'translate',[translator,module,'-o',generated],build=True)
        self.command(work,'build',[*shlex.split(os.environ.get('CC','cc')),'-std=c11','-O1','-g',
            '-Wall','-Wextra','-Werror','-fsanitize=address,undefined','-fno-sanitize-recover=all',
            generated,ROOT/'bin/nano_aot_runtime.o','-lm',*native_link_flags(),'-o',native],build=True)
        result=self.command(work,'native',[native]);self.assertEqual((result.stdout,result.stderr),('',''))

    def test_source_effects_aliases_and_single_evaluation(self):
        source='''let mut counter:int=0
fn items()->array<string>{set counter (+ counter 1) return ["kept"]}
shadow items{let saved:int=counter set counter 0 let values:array<string> = (items) assert (== counter 1) assert (== (at values 0) "kept") set counter saved}
fn main()->int{
 set counter 0
 let values:array<string> = (items)
 let first:string=(to_string values)
 assert (== first "") assert (== counter 1) assert (== (at values 0) "kept")
 let second:string=(to_string (items))
 assert (== second "") assert (== counter 2) assert (== (at values 0) "kept")
 return 0
}
shadow main{assert (== (main) 0)}
'''
        pure=routes.source_compiler_command();self.assertTrue(pure)
        translator=Path(os.environ.get('NANOLANG_NVM2C_TEST_TOOL',ROOT/'bin/nvm2c'))
        for label,compiler in [('c',[ROOT/'bin/nano_virt']),('pure',pure)]:
            with self.subTest(producer=label):
                work=self.directory('cast-source-effects-'+label+'-')
                path,module,generated,native=[work/n for n in ['main.nano','main.nvm','main.c','main']]
                path.write_text(source)
                self.command(work,'compile',[*compiler,path,'--emit-nvm','-o',module],build=True)
                self.command(work,'vm',[ROOT/'bin/nano_vm',module])
                self.command(work,'translate',[translator,module,'-o',generated],build=True)
                self.command(work,'build',[*shlex.split(os.environ.get('CC','cc')),'-std=c11','-O1','-g',
                    '-fsanitize=address,undefined','-fno-sanitize-recover=all',generated,
                    ROOT/'bin/nano_aot_runtime.o','-lm',*native_link_flags(),'-o',native],build=True)
                result=self.command(work,'native',[native]);self.assertEqual((result.stdout,result.stderr),('',''))
