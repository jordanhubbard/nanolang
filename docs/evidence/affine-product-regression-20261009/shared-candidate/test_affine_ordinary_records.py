"""I keep copyable record identity distinct from ownership authority."""
import os,platform,shlex,struct,subprocess,tempfile,unittest
from pathlib import Path
REPO = Path(__file__).resolve().parents[1]
ROOT = Path(os.environ.get("NANO_AFFINE_RECORD_TOOLS", REPO / "bin"))
NONE=0xffffffff
INT=(1,NONE); BOOL=(4,NONE); A=(8,0); B=(8,1); OWNER=(8,2); BOX=(8,3)
def desc(t):return struct.pack('<BBHI',t[0],0,0,t[1])
def source(body,extra=(),helpers=(),flags=(1,1,3,1),fields=None):
    fields=fields or [[INT],[INT],[INT],[A]]
    layouts=struct.pack('<I',len(fields))
    for children in fields:
        layouts+=struct.pack('<BBHI',0,0,len(children),NONE)
        for tag,layout in children:layouts+=struct.pack('<B3xII',tag,layout,NONE)
    funcs=[('main',0,[OWNER,INT,*extra],INT,body),*helpers]
    own=struct.pack('<II',1,len(flags))+bytes(flags)
    own+=bytes((-len(own))%4)+struct.pack('<I',len(funcs))
    text=f'.types {len(fields)} 0 0\n.entry 0\n.layouts "{layouts.hex()}"\n'
    for name,arity,locals_,result,code in funcs:
        own+=struct.pack('<HH',len(locals_),arity)+desc(result)+b''.join(map(desc,locals_))
        text+=f'.function {name} {arity} {len(locals_)} 0 '+('struct' if result[0]==8 else 'int')+' 1\n'+code+'\n.end\n'
    for i,(_,arity,locals_,_,_) in enumerate(funcs):
        if arity:text+=f'.parameters {i} '+' '.join('struct' if t[0]==8 else 'int' for t in locals_[:arity])+'\n'
    return text+f'.ownership "{own.hex()}"\n'
PREFIX='PUSH_I64 7\nOWN_PACK 2\nOWN_STORE_LOCAL 0\nOWN_UNPACK_LOCAL 0\nSTORE_LOCAL 1\n'
PACK='PUSH_I64 42\nAGG_PACK 0 0 0 1\n'
END='PUSH_I64 0\nRET'
class Records(unittest.TestCase):
    def checked(self,args,**kw):
        p=subprocess.run(list(map(str,args)),capture_output=True,text=True,timeout=90,**kw)
        self.assertEqual(p.returncode,0,p.stdout+p.stderr);return p
    def assemble(self,text,work,accepted):
        path=work/'input.nvm';path.write_bytes(b'prior artifact')
        assembly=work/'input.nasm';assembly.write_text(text)
        result=subprocess.run([str(ROOT/'nanoisa'),'asm',str(assembly),'-o',str(path)],capture_output=True,text=True,timeout=90)
        if accepted:self.assertEqual(result.returncode,0,result.stdout+result.stderr)
        else:
            self.assertNotEqual(result.returncode,0,'I unexpectedly admitted malformed bytecode')
            self.assertEqual(path.read_bytes(),b'prior artifact')
        return path
    def test_malformed_authority_and_identity(self):
        good=PREFIX+PACK+'STORE_LOCAL 2\n'+END
        cases={
            'owner_construction_as_copy':source(PREFIX+'PUSH_I64 1\nAGG_PACK 0 2 0 1\nPOP\n'+END),
            'copy_construction_as_owner':source(PREFIX+'PUSH_I64 1\nOWN_PACK 0\nPOP\n'+END),
            'wrong_field_tag':source(PREFIX+'PUSH_BOOL 1\nAGG_PACK 0 0 0 1\nPOP\n'+END),
            'wrong_field_count':source(PREFIX+'AGG_PACK 0 0 0 0\nPOP\n'+END),
            'wrong_local_layout':source(good,[B]),
            'incomplete_layout':source(good,[A],flags=(0,1,3,0)),
            'wrong_nested_layout':source(PREFIX+'PUSH_I64 3\nAGG_PACK 0 1 0 1\nAGG_PACK 0 3 0 1\nPOP\n'+END),
            'owner_child_in_copyable_layout':source(good,[A],fields=[[INT],[INT],[INT],[OWNER]]),
            'owner_observation_escape':source('PUSH_I64 7\nOWN_PACK 2\nOWN_STORE_LOCAL 0\nLOAD_LOCAL 0\nSTORE_LOCAL 2\n'+END,[A]),
            'live_owner_exit':source('PUSH_I64 7\nOWN_PACK 2\nOWN_STORE_LOCAL 0\n'+END),
            'uninitialized_join':source(PREFIX+'PUSH_BOOL 1\nJMP_FALSE join\n'+PACK+'STORE_LOCAL 2\njoin:\nLOAD_LOCAL 2\nPOP\n'+END,[A]),
            'different_layout_join':source(PREFIX+'PUSH_BOOL 1\nJMP_FALSE other\n'+PACK+'JMP join\nother:\nPUSH_I64 1\nAGG_PACK 0 1 0 1\njoin:\nPOP\n'+END),
            'wrong_argument_layout':source(PREFIX+'PUSH_I64 1\nAGG_PACK 0 1 0 1\nCALL 1\nPOP\n'+END,helpers=[('read',1,[A],INT,'LOAD_LOCAL 0\nAGG_GET 0\nRET')]),
            'wrong_result_layout':source(PREFIX+'CALL 1\nPOP\n'+END,helpers=[('make',0,[],A,'PUSH_I64 1\nAGG_PACK 0 1 0 1\nRET')]),
        }
        for label,text in cases.items():
            with self.subTest(case=label),tempfile.TemporaryDirectory() as directory:self.assemble(text,Path(directory),False)
    def test_vm_native_alias_nested_calls_and_cleanup(self):
        self.check_runtime(INT, 'PUSH_I64 42', 'PUSH_I64 42')
        self.check_runtime((5,NONE), 'PUSH_STR 0', 'PUSH_STR 0')

    def check_runtime(self,leaf,construct,expected):
        pack=construct+'\nAGG_PACK 0 0 0 1\n'
        code=PREFIX+pack+'STORE_LOCAL 2\nLOAD_LOCAL 2\nCALL 1\nSTORE_LOCAL 3\nLOAD_LOCAL 2\nAGG_PACK 0 3 0 1\nSTORE_LOCAL 4\nLOAD_LOCAL 4\nAGG_GET 0\nAGG_GET 0\n'+expected+'\nEQ\nASSERT\nLOAD_LOCAL 3\nAGG_GET 0\n'+expected+'\nEQ\nASSERT\n'+END
        text='.string "retained string"\n'+source(code,[A,A,BOX],[('copy',1,[A],A,'LOAD_LOCAL 0\nRET')],fields=[[leaf],[INT],[INT],[A]])
        with self.subTest(leaf=leaf),tempfile.TemporaryDirectory() as directory:
            work=Path(directory);module=self.assemble(text,work,True)
            self.checked([ROOT/'nano_vm','--verify-only',module]);self.checked([ROOT/'nano_vm',module])
            self.checked([ROOT/'nvm2c',module,'-o',work/'records.c'])
            (work/'harness.c').write_text('''#include <assert.h>
#include <stdint.h>
#include <stdlib.h>
#include <stdio.h>
static size_t live,attempt,fail_at;
static void *allocate(size_t count,size_t size){
 if(++attempt==fail_at)return NULL;
 void *value=calloc(count,size);if(value)live++;return value;
}
static void release(void *value){if(value){assert(live);live--;free(value);}}
#define NOWN_ALLOC allocate
#define NOWN_FREE release
#define NVM2C_NO_MAIN
#include "records.c"
int main(void){
 for(size_t failure=1;;failure++){
  int64_t result=-1;attempt=0;fail_at=failure;
  int status=nvm_owned_entry(&result);
  assert(live==0);
  if(!status){assert(result==0);assert(failure>2);break;}
  assert(result==-1);assert(failure<32);
 }
 puts("copyable record cleanup passed");return 0;
}
''')
            compiler=os.environ.get('NANO_NATIVE_TEST_CC') or os.environ.get('CC') or ('/opt/homebrew/opt/llvm/bin/clang' if platform.system()=='Darwin' and Path('/opt/homebrew/opt/llvm/bin/clang').is_file() else 'cc')
            self.checked([*shlex.split(compiler),'-std=c11','-Wall','-Wextra','-Werror','-fsanitize=address,undefined','-fno-sanitize-recover=all',work/'harness.c','-o',work/'harness'])
            result=self.checked([work/'harness'],env={**os.environ,'ASAN_OPTIONS':'detect_leaks=1:halt_on_error=1','UBSAN_OPTIONS':'halt_on_error=1'})
            self.assertIn('copyable record cleanup passed',result.stdout)
if __name__=='__main__':unittest.main(verbosity=2)
