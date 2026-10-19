"""I retain structured owner states across actual source-produced branches."""
import copy
import json
import os
from pathlib import Path
import shlex
import struct
import sys
import tempfile
import unittest
import zlib
from tests.native_process import native_link_flags, run_captured

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
from nanoisa_owned_reconstruction import reconstruct_owned
from nanoisa_reconstruction import Refusal

CASES={'terminal':15,'diamond':15,'keep-live':17,'branch-local':15,
       'early-return':15,'owner-terminal':14,'field-condition':15,'call-condition':15,'nested':34}

class OwnedBranchReconstruction(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.work=Path(tempfile.mkdtemp(prefix='nano-owned-branches-'))
        cls.compilers=[ROOT/'bin/nano_virt']+[Path(p) for p in shlex.split(os.environ.get('NANOLANG_OWNED_RECONSTRUCTION_COMPILERS',''))]
        cls.serial=0
        print('I retain owned branch evidence at '+str(cls.work),flush=True)

    def command(self,args,expected=0,build=False):
        type(self).serial+=1
        name=str(self.serial)
        result=run_captured(args,cwd=ROOT,timeout=180,build=build,text=True)
        (self.work/(name+'.json')).write_text(json.dumps({'command':list(map(str,args)),'returncode':result.returncode,'stdout':result.stdout,'stderr':result.stderr},indent=2)+'\n')
        if expected is not None:self.assertEqual(result.returncode,expected,result.stdout+result.stderr)
        return result

    def module(self,case,compiler=None):
        compiler=compiler or self.compilers[0]
        source=ROOT/'tests/nanoisa/fixtures/owned_branches'/(case+'.nano')
        module=self.work/(case+'-'+compiler.name+'.nvm')
        self.command([compiler,source,'--emit-nvm','-o',module])
        return module

    def facts(self,module):
        return json.loads(self.command([ROOT/'bin/nanoisa_hl_facts',module]).stdout)

    def native(self,source,expected):
        for opt in ('-O0','-O2'):
            binary=source.with_suffix('.'+opt[1:]+'.exe')
            self.command([*shlex.split(os.environ.get('NANO_NATIVE_TEST_CC','cc')),'-std=c11',opt,
                '-Wall','-Wextra','-Werror','-fsanitize=address,undefined','-fno-sanitize-recover=all',
                source,ROOT/'bin/nano_aot_runtime.o','-lm',*native_link_flags(),'-o',binary],build=True)
            self.command([binary],expected)

    def test_actual_producers_and_both_structured_outputs(self):
        for case,expected in CASES.items():
            for compiler in self.compilers:
                with self.subTest(case=case,producer=str(compiler)):
                    module=self.module(case,compiler)
                    self.command([ROOT/'bin/nano_vm','--verify-only',module])
                    self.command([ROOT/'bin/nano_vm',module],expected)
                    facts=self.facts(module)
                    for language in ('c','nano'):
                        output=module.with_suffix('.'+language)
                        self.command([ROOT/'bin/nvm2hl','--language',language,module,'-o',output])
                        self.assertRegex(output.read_text(),r'\bif\b')
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
                                signatures=lambda m:sorted(repr((f['params'],f['result_type'])) for f in m['functions'])
                                for signature in signatures(facts):self.assertIn(signature,signatures(regenerated))
                                # Generated Nano may add checked scalar arithmetic helpers.
                                def calls(m):
                                    owners={n for n,f in enumerate(m['functions'])
                                            if any(t[0]==8 for t in f['params']) or f['result_type'][0]==8}
                                    return sum(i['op']=='CALL' and i['arg'] in owners
                                               for f in m['functions'] for i in f['code'])
                                self.assertEqual(calls(facts),calls(regenerated))
                                native=rebuilt.with_suffix('.c')
                                self.command([ROOT/'bin/nvm2c',rebuilt,'-o',native])
                                self.native(native,expected)

    def test_crc_valid_branch_mutations_refuse_atomically(self):
        module=self.module('diamond');facts=self.facts(module);raw=module.read_bytes()
        header,count=struct.unpack_from('<II',raw,20)
        sections={}
        for n in range(count):
            kind,_,offset,size=struct.unpack_from('<IIQQ',raw,header+24*n);sections[kind]=(offset,size)
        fn=facts['functions'][1];code=fn['code'];base=sections[6][0]+facts['functions'][0]['size']
        branch=next(ins for ins in code if ins['op']=='JMP_FALSE')
        read=next(ins for ins in code if ins['op']=='REF_GET')
        move=next(ins for ins in code if ins['op']=='OWN_MOVE_LOCAL')
        cases={'backedge':(base+branch['pc']+1,struct.pack('<i',-branch['pc'])),
               'non-boundary':(base+branch['pc']+1,struct.pack('<i',1)),
               'borrow-interior':(base+branch['pc']+1,struct.pack('<i',read['pc']-branch['pc'])),
               'lost-owner':(base+move['pc'],b'\x00\x00\x00'),
               'missing-authority':(sections[13][0]+8,b'\x00')}
        for case,(at,payload) in cases.items():
            changed=bytearray(raw);changed[at:at+len(payload)]=payload
            struct.pack_into('<I',changed,36,zlib.crc32(changed[header:]))
            bad=self.work/(case+'.nvm');bad.write_bytes(changed)
            for language in ('c','nano'):
                output=self.work/('previous.'+language);output.write_text('prior checked source')
                result=self.command([ROOT/'bin/nvm2hl','--language',language,bad,'-o',output],None)
                self.assertNotEqual(result.returncode,0)
                self.assertEqual(output.read_text(),'prior checked source')
                self.assertIn('I did not publish',result.stderr)
                self.assertNotIn('checksum',result.stderr.lower())

    def test_join_state_refusals_from_actual_facts(self):
        original=self.facts(self.module('diamond'))
        from nanoisa_owned_reconstruction import OwnedAnalyze
        def nop(ins):
            ins.update(op='NOP',arg=0,args=[])
        def refusal(facts,diagnostic):
            with self.assertRaisesRegex(Refusal,diagnostic):OwnedAnalyze(facts,1).run()
        mismatch=copy.deepcopy(original)
        code=mismatch['functions'][1]['code']
        unpack=next(i for i,x in enumerate(code) if x['op']=='OWN_UNPACK_LOCAL')
        for ins in code[unpack:unpack+3]:nop(ins)
        refusal(mismatch,'live-owner agreement')

        escape=copy.deepcopy(mismatch);code=escape['functions'][1]['code']
        stores=[x for x in code if x['op']=='OWN_STORE_LOCAL']
        first,second=stores[0]['arg'],stores[1]['arg']
        for ins in code:
            if ins['op'] in ('OWN_STORE_LOCAL','OWN_UNPACK_LOCAL') and ins['arg']==second:
                ins['arg']=first;ins['args']=[first]
            if ins['op']=='BORROW_LOCAL_SHARED' and ins['args'][1]==second:ins['args'][1]=first
        unpack=next(i for i,x in enumerate(code) if x['op']=='OWN_UNPACK_LOCAL')
        for ins in code[unpack:unpack+3]:nop(ins)
        refusal(escape,'branch-local owners escaping')

        staged=copy.deepcopy(original);code=staged['functions'][1]['code']
        code[0].update(op='OWN_MOVE_LOCAL',arg=0,args=[0]);nop(code[1])
        refusal(staged,'empty operand stacks at owned branches')

        crossing=self.facts(self.module('nested'));code=crossing['functions'][1]['code']
        inner=[x for x in code if x['op']=='JMP_FALSE'][1]
        inner['target']=code[-1]['pc']
        refusal(crossing,'forward owned region boundary')

        partial=copy.deepcopy(original);code=partial['functions'][1]['code']
        # I remove the initial value and the complete right-arm scalar assignment.
        for ins in code[:2]:nop(ins)
        stores=[i for i,x in enumerate(code) if x['op']=='STORE_LOCAL' and x['arg']==2]
        for ins in code[stores[-1]-3:stores[-1]+1]:nop(ins)
        refusal(partial,'initialized')

        analyzer=OwnedAnalyze(original,1)
        with self.assertRaisesRegex(Refusal,'nesting'):
            analyzer.region(0,len(analyzer.code),{0,1},analyzer.live,analyzer.used,33)

    def test_ephemeral_sequence_cannot_cross_region_end(self):
        facts=self.facts(self.module('field-condition'))
        fn=facts['functions'][1];code=fn['code']
        # I supplement real-module refusals with an analyzer region-boundary control.
        region=next(i for i,ins in enumerate(code) if ins['op']=='REGION_BEGIN')
        from nanoisa_owned_reconstruction import OwnedAnalyze
        analyzer=OwnedAnalyze(facts,1)
        with self.assertRaisesRegex(Refusal,'complete ephemeral read'):
            analyzer.region(region,region+3,set(range(len(analyzer.params))),analyzer.live,analyzer.used)

if __name__=='__main__':unittest.main()
