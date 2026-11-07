"""I retain exact owner lifetimes across source-produced pure-pretest loops."""
import copy
import json
from pathlib import Path
import sys
import struct
import zlib
import unittest
from tests import test_owned_branch_reconstruction as branch_support
ROOT=branch_support.ROOT
sys.path.insert(0,str(ROOT/'scripts'))
from nanoisa_owned_reconstruction import OwnedAnalyze
from nanoisa_reconstruction import Refusal

class OwnedLoopReconstruction(unittest.TestCase):
    setUpClass=classmethod(branch_support.OwnedBranchReconstruction.setUpClass.__func__)
    command=branch_support.OwnedBranchReconstruction.command
    facts=branch_support.OwnedBranchReconstruction.facts
    native=branch_support.OwnedBranchReconstruction.native

    def module(self,case,compiler=None):
        compiler=compiler or self.compilers[0]
        source=ROOT/'tests/nanoisa/fixtures/owned_loops'/(case+'.nano')
        module=self.work/(case+'-'+compiler.name+'.nvm')
        self.command([compiler,source,'--emit-nvm','-o',module])
        return module

    def test_actual_loop_lifetimes_and_both_outputs(self):
        for case,expected in {'preserve':17,'borrow-body':35,'fresh-body':20,
                              'early-return':114,'body-return':15,'nested':26,'borrow-condition':17,'arithmetic-condition':17}.items():
            for compiler in self.compilers:
                with self.subTest(case=case,producer=str(compiler)):
                    module=self.module(case,compiler);facts=self.facts(module)
                    self.command([ROOT/'bin/nano_vm','--verify-only',module])
                    self.command([ROOT/'bin/nano_vm',module],expected)
                    for language in ('c','nano'):
                        output=module.with_suffix('.'+language)
                        self.command([ROOT/'bin/nvm2hl','--language',language,module,'-o',output])
                        if case!='body-return':self.assertRegex(output.read_text(),r'\bwhile\b')
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
                                signatures=lambda m:[(f['params'],f['result_type']) for f in m['functions']]
                                for signature in signatures(facts):self.assertIn(signature,signatures(regenerated))
                                native=rebuilt.with_suffix('.c')
                                self.command([ROOT/'bin/nvm2c',rebuilt,'-o',native])
                                self.native(native,expected)

    def test_actual_arithmetic_pretests_refuse_atomically(self):
        for case in ('division-condition',):
            for compiler in self.compilers:
                module=self.module(case,compiler)
                self.command([ROOT/'bin/nano_vm','--verify-only',module])
                self.command([ROOT/'bin/nano_vm',module],17)
                for language in ('c','nano'):
                    output=self.work/('prior.'+language);output.write_text('prior checked source')
                    result=self.command([ROOT/'bin/nvm2hl','--language',language,module,'-o',output],None)
                    self.assertNotEqual(result.returncode,0)
                    self.assertIn('pure scalar owned pretest',result.stderr)
                    self.assertIn('I did not publish',result.stderr)
                    self.assertEqual(output.read_text(),'prior checked source')

    def test_verified_inverted_pretest_and_unreachable_backedge(self):
        original=self.module('preserve');facts=self.facts(original);raw=bytearray(original.read_bytes())
        header,count=struct.unpack_from('<II',raw,20)
        sections={}
        for n in range(count):
            kind,_,offset,size=struct.unpack_from('<IIQQ',raw,header+24*n)
            sections[kind]=(offset,size)
        base=sections[6][0]+facts['functions'][0]['size']
        code=facts['functions'][1]['code']
        comparison=next(x for x in code if x['op']=='LT')
        branch=next(x for x in code if x['op']=='JMP_FALSE')
        self.assertEqual(raw[base+comparison['pc']],0x2a)
        self.assertEqual(raw[base+branch['pc']],0x3a)
        raw[base+comparison['pc']]=0x2d  # I invert LT to GE together with the branch.
        raw[base+branch['pc']]=0x39
        struct.pack_into('<I',raw,36,zlib.crc32(raw[header:]))
        inverted=self.work/'inverted.nvm';inverted.write_bytes(raw)

        terminal=self.module('body-return')
        assembly=self.command([ROOT/'bin/nanoisa','dump',terminal]).stdout
        self.assertIn('.function pick 2 8 0 int 1\n',assembly)
        self.assertIn('  RET\nL0:',assembly)
        assembly=assembly.replace('.function pick 2 8 0 int 1\n',
                                  '.function pick 2 8 0 int 1\nLOOP_HEADER:\n')
        assembly=assembly.replace('  RET\nL0:', '  RET\n  JMP LOOP_HEADER\nL0:')
        source=self.work/'return-backedge.nasm';source.write_text(assembly)
        returning=self.work/'return-backedge.nvm'
        self.command([ROOT/'bin/nanoisa','asm',source,'-o',returning])
        for module,expected in ((inverted,17),(returning,15)):
            self.command([ROOT/'bin/nano_vm','--verify-only',module])
            self.command([ROOT/'bin/nano_vm',module],expected)
            for language in ('c','nano'):
                output=module.with_suffix('.'+language)
                self.command([ROOT/'bin/nvm2hl','--language',language,module,'-o',output])
                if language=='c':self.native(output,expected)
                else:
                    for compiler in self.compilers:
                        rebuilt=output.with_suffix('.'+compiler.name+'.nvm')
                        self.command([compiler,output,'--emit-nvm','-o',rebuilt])
                        self.command([ROOT/'bin/nano_vm','--verify-only',rebuilt])
                        self.command([ROOT/'bin/nano_vm',rebuilt],expected)
                        native=rebuilt.with_suffix('.c')
                        self.command([ROOT/'bin/nvm2c',rebuilt,'-o',native])
                        self.native(native,expected)

    def test_crc_valid_loop_edges_refuse_atomically(self):
        module=self.module('preserve');facts=self.facts(module);raw=module.read_bytes()
        header,count=struct.unpack_from('<II',raw,20)
        sections={}
        for n in range(count):
            kind,_,offset,size=struct.unpack_from('<IIQQ',raw,header+24*n)
            sections[kind]=(offset,size)
        code=facts['functions'][1]['code']
        base=sections[6][0]+facts['functions'][0]['size']
        branch=next(x for x in code if x['op']=='JMP_FALSE')
        back=next(x for x in code if x['op']=='JMP')
        condition=[x for x in code if x['pc']>=back['target']][1]
        borrow=next(x for x in code if x['op']=='BORROW_LOCAL_SHARED')
        for case,ins,target in (
            ('middle-header',back,condition['pc']),
            ('wrong-exit',branch,back['pc']),
            ('conditional-backedge',branch,back['target']),
            ('borrow-interior',back,borrow['pc'])):
            changed=bytearray(raw)
            struct.pack_into('<i',changed,base+ins['pc']+1,target-ins['pc'])
            struct.pack_into('<I',changed,36,zlib.crc32(changed[header:]))
            bad=self.work/(case+'.nvm');bad.write_bytes(changed)
            for language in ('c','nano'):
                output=self.work/('prior-edge.'+language);output.write_text('prior checked source')
                result=self.command([ROOT/'bin/nvm2hl','--language',language,bad,'-o',output],None)
                self.assertNotEqual(result.returncode,0)
                self.assertIn('I did not publish',result.stderr)
                self.assertNotIn('checksum',result.stderr.lower())
                self.assertEqual(output.read_text(),'prior checked source')

    def test_loop_state_refusals_from_actual_facts(self):
        original=self.facts(self.module('preserve'))
        def nop(ins):ins.update(op='NOP',arg=0,args=[])
        def refusal(facts,message):
            with self.assertRaisesRegex(Refusal,message):OwnedAnalyze(facts,1).run()
        lost=copy.deepcopy(original);code=lost['functions'][1]['code']
        branch=next(i for i,x in enumerate(code) if x['op']=='JMP_FALSE')
        code[branch+1].update(op='OWN_UNPACK_LOCAL',arg=0,args=[0])
        for ins in code[branch+2:branch+4]:ins.update(op='POP',arg=0,args=[])
        nop(code[branch+4])
        refusal(lost,'carried owner identities')

        staged=copy.deepcopy(original);code=staged['functions'][1]['code']
        code[0].update(op='OWN_MOVE_LOCAL',arg=0,args=[0]);nop(code[1])
        refusal(staged,'empty owned loop-entry stack')

        fresh=self.facts(self.module('fresh-body'));code=fresh['functions'][1]['code']
        unpack=next(i for i,x in enumerate(code) if x['op']=='OWN_UNPACK_LOCAL')
        for ins in code[unpack:unpack+3]:nop(ins)
        refusal(fresh,'carried owner identities')

        partial=copy.deepcopy(original);fn=partial['functions'][1];code=fn['code']
        slot=fn['locals'];fn['locals']+=1;fn['local_types'].append([1,4294967295])
        code[branch+4].update(arg=slot,args=[slot])
        loads=[x for x in code if x['op']=='LOAD_LOCAL' and x['arg']==2]
        loads[-1].update(arg=slot,args=[slot])
        refusal(partial,'definitely initialized')

if __name__=='__main__':unittest.main()
