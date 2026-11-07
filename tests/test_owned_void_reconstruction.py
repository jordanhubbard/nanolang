"""I preserve zero-result ownership calls and ordered, non-elidable assertions."""
import copy
import os
from pathlib import Path
import shlex
import struct
import sys
import unittest
import zlib
from tests import test_owned_branch_reconstruction as support
from tests.native_process import native_link_flags
ROOT=support.ROOT
sys.path.insert(0,str(ROOT/'scripts'))
from nanoisa_owned_reconstruction import OwnedAnalyze
from nanoisa_reconstruction import Refusal

class OwnedVoidReconstruction(unittest.TestCase):
    setUpClass=classmethod(support.OwnedBranchReconstruction.setUpClass.__func__)
    command=support.OwnedBranchReconstruction.command
    facts=support.OwnedBranchReconstruction.facts
    native=support.OwnedBranchReconstruction.native

    def module(self,case,compiler=None):
        compiler=compiler or self.compilers[0]
        source=(ROOT/'tests/nanoisa/fixtures/source_owned_value_graph.nano' if case=='graph' else
                ROOT/'tests/nanoisa/fixtures/owned_void'/(case+'.nano'))
        module=self.work/(case+'-'+compiler.name+'.nvm')
        self.command([compiler,source,'--emit-nvm','-o',module])
        return module

    def signatures(self,facts):
        return [(f['params'],f['result_type'],f['result_count']) for f in facts['functions']]

    def test_actual_void_helpers_and_structured_outputs(self):
        for case,expected in (('graph',0),('control',3)):
            for compiler in self.compilers:
                with self.subTest(case=case,producer=str(compiler)):
                    module=self.module(case,compiler);facts=self.facts(module)
                    self.assertIn([0,4294967295],[f['result_type'] for f in facts['functions']])
                    self.command([ROOT/'bin/nano_vm','--verify-only',module])
                    self.command([ROOT/'bin/nano_vm',module],expected)
                    for language in ('c','nano'):
                        output=module.with_suffix('.'+language)
                        self.command([ROOT/'bin/nvm2hl','--language',language,module,'-o',output])
                        self.assertNotRegex(output.read_text(),r'\bgoto\b|nano_vm|nvm_blob|switch\s*\(')
                        if language=='c':self.native(output,expected)
                        else:
                            for rebuild in self.compilers:
                                rebuilt=output.with_suffix('.'+rebuild.name+'.nvm')
                                self.command([rebuild,output,'--emit-nvm','-o',rebuilt])
                                self.command([ROOT/'bin/nano_vm','--verify-only',rebuilt])
                                self.command([ROOT/'bin/nano_vm',rebuilt],expected)
                                regenerated=self.facts(rebuilt)
                                self.assertEqual(facts['layouts'],regenerated['layouts'])
                                for signature in self.signatures(facts):self.assertIn(signature,self.signatures(regenerated))
                                def void_calls(m):
                                    void={n for n,f in enumerate(m['functions']) if f['result_count']==0}
                                    return sum(i['op']=='CALL' and i['arg'] in void for f in m['functions'] for i in f['code'])
                                self.assertEqual(void_calls(facts),void_calls(regenerated))
                                native=rebuilt.with_suffix('.c')
                                self.command([ROOT/'bin/nvm2c',rebuilt,'-o',native])
                                self.native(native,expected)

    def native_failure(self,source,diagnostic):
        for opt in ('-O0','-O2'):
            for defines in ([],['-DNDEBUG']):
                binary=source.with_suffix('.'+opt[1:]+('NDEBUG' if defines else '')+'.exe')
                self.command([*shlex.split(os.environ.get('NANO_NATIVE_TEST_CC','cc')),'-std=c11',opt,
                    *defines,'-Wall','-Wextra','-Werror','-fsanitize=address,undefined','-fno-sanitize-recover=all',
                    source,ROOT/'bin/nano_aot_runtime.o','-lm',*native_link_flags(),'-o',binary],build=True)
                # I change only this child's core-file limit; sanitizer settings stay intact.
                result=self.command([sys.executable,'-c',
                    'import os,resource,sys; resource.setrlimit(resource.RLIMIT_CORE,(0,0)); os.execv(sys.argv[1],sys.argv[1:])',binary],None)
                self.assertNotEqual(result.returncode,0)
                if diagnostic is None:
                    self.assertEqual(result.returncode,1)
                    self.assertEqual(result.stderr,'')
                else:
                    self.assertIn(diagnostic,result.stderr)
                self.assertNotIn('ERROR: AddressSanitizer',result.stderr)

    def native_assert_status(self,source):
        # The owned AOT public main maps any nonzero status to exit 1 silently.
        # I additionally observe its existing status API without changing the product.
        probe=source.with_suffix('.status.c')
        probe.write_text('#define NVM2C_NO_MAIN\n#include "'+str(source)+'"\n'
            'int main(void) { int64_t value=0; int status=nvm_owned_entry(&value); '
            'fprintf(stderr,"owned status=%d\\n",status); return status; }\n')
        for opt in ('-O0','-O2'):
            binary=probe.with_suffix('.'+opt[1:]+'.exe')
            self.command([*shlex.split(os.environ.get('NANO_NATIVE_TEST_CC','cc')),'-std=c11',opt,
                '-DNDEBUG','-Wall','-Wextra','-Werror','-fsanitize=address,undefined','-fno-sanitize-recover=all',
                probe,ROOT/'bin/nano_aot_runtime.o','-lm',*native_link_flags(),'-o',binary],build=True)
            result=self.command([binary],2)
            self.assertEqual(result.stderr,'owned status=2\n')

    def vm_failure(self,module):
        self.command([ROOT/'bin/nano_vm','--verify-only',module])
        result=self.command([ROOT/'bin/nano_vm',module],None)
        self.assertNotEqual(result.returncode,0)
        self.assertIn('Assertion failed',result.stdout+result.stderr)

    def test_false_assertion_traps_before_later_division_and_cleanup(self):
        for compiler in self.compilers:
            module=self.module('assert-failure',compiler)
            self.vm_failure(module)
            for language in ('c','nano'):
                output=module.with_suffix('.'+language)
                self.command([ROOT/'bin/nvm2hl','--language',language,module,'-o',output])
                if language=='c':self.native_failure(output,'Assertion failed')
                else:
                    for rebuild in self.compilers:
                        rebuilt=output.with_suffix('.'+rebuild.name+'.nvm')
                        self.command([rebuild,output,'--emit-nvm','-o',rebuilt])
                        self.vm_failure(rebuilt)
                        native=rebuilt.with_suffix('.c')
                        self.command([ROOT/'bin/nvm2c',rebuilt,'-o',native])
                        self.native_failure(native,None)
                        self.native_assert_status(native)

    def test_crc_valid_counts_layouts_and_void_parameters_refuse_atomically(self):
        module=self.module('graph');facts=self.facts(module);original=module.read_bytes()
        header,count=struct.unpack_from('<II',original,20);sections={}
        for n in range(count):
            kind,_,offset,size=struct.unpack_from('<IIQQ',original,header+24*n);sections[kind]=(offset,size)
        at=sections[13][0];layouts=struct.unpack_from('<I',original,at+4)[0]
        at+=((8+layouts+3)//4)*4+4;descriptors=[]
        for fn in facts['functions']:
            locals_count,params=struct.unpack_from('<HH',original,at)
            self.assertEqual((locals_count,params),(fn['locals'],len(fn['params'])))
            descriptors.append(at+4);at+=4+8*(1+locals_count)
        void=next(n for n,f in enumerate(facts['functions']) if f['result_count']==0)
        # I read v2 function signature indices and padded signature records.
        sig_count=struct.unpack_from('<I',original,sections[3][0])[0]
        sig_at=sections[3][0]+4;result_offsets=[]
        for _ in range(sig_count):
            params,results=struct.unpack_from('<HH',original,sig_at)
            result_offsets.append(sig_at+2)
            sig_at+=4+((params+3)//4)*4+((results+3)//4)*4
        function_signature=lambda n:struct.unpack_from('<I',original,sections[5][0]+4+32*n+4)[0]
        void_count=result_offsets[function_signature(void)]
        value_count=result_offsets[function_signature(0)]
        self.assertEqual(struct.unpack_from('<H',original,void_count)[0],0)
        self.assertEqual(struct.unpack_from('<H',original,value_count)[0],1)
        cases={
            'void-count-one':(void_count,struct.pack('<H',1)),
            'value-count-zero':(value_count,struct.pack('<H',0)),
            'void-result-layout':(descriptors[void]+4,struct.pack('<I',0)),
            'void-parameter':(descriptors[void]+8,b'\x00'),
            'missing-layout-authority':(sections[13][0]+8,b'\x00'),
        }
        for case,(at,payload) in cases.items():
            self.assertNotEqual(original[at:at+len(payload)],payload)
            data=bytearray(original);data[at:at+len(payload)]=payload
            struct.pack_into('<I',data,36,zlib.crc32(data[header:]))
            bad=self.work/(case+'.nvm');bad.write_bytes(data)
            verification=self.command([ROOT/'bin/nano_vm','--verify-only',bad],None)
            self.assertNotEqual(verification.returncode,0)
            for language in ('c','nano'):
                output=self.work/('prior.'+language);output.write_text('prior checked source')
                result=self.command([ROOT/'bin/nvm2hl','--language',language,bad,'-o',output],None)
                self.assertNotEqual(result.returncode,0)
                self.assertEqual(output.read_text(),'prior checked source')
                self.assertIn('I did not publish',result.stderr)
                self.assertNotIn('checksum',result.stderr.lower())

    def test_void_stack_and_false_assertion_do_not_forge_cleanup(self):
        original=self.facts(self.module('graph'))
        void=next(n for n,f in enumerate(original['functions']) if f['result_count']==0)
        def refusal(m,index,message):
            with self.assertRaisesRegex(Refusal,message):OwnedAnalyze(m,index).run()
        def nop(ins):ins.update(op='NOP',arg=0,args=[])
        extra=copy.deepcopy(original)
        nop(next(x for x in extra['functions'][void]['code'] if x['op']=='POP'))
        refusal(extra,void,'empty operand stack at VOID return')
        missing=copy.deepcopy(original);code=missing['functions'][0]['code']
        nop(code[-2]);refusal(missing,0,'populated scalar operand stack')
        nonbool=copy.deepcopy(original)
        nop(next(x for x in nonbool['functions'][void]['code'] if x['op']=='EQ'))
        refusal(nonbool,void,'exact scalar operand types')
        dirty=copy.deepcopy(original);code=dirty['functions'][void]['code']
        assertion=next(i for i,x in enumerate(code) if x['op']=='ASSERT')
        nop(code[assertion-3]);nop(code[assertion-2])
        code[assertion-1].update(op='PUSH_BOOL',arg=0,args=[0])
        for ins in code:
            if ins['op'] in ('OWN_UNPACK_LOCAL','POP'):nop(ins)
        refusal(dirty,void,'all owners consumed')
        entry=copy.deepcopy(original);entry['functions'][0].update(result_type=[0,4294967295],result=0,result_count=0)
        refusal(entry,0,'scalar entry')

if __name__=='__main__':unittest.main()
