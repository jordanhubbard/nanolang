"""I retain ordered exact exclusive helper mutation through both outputs."""
import copy
import struct
import zlib
import unittest
from tests import test_owned_shared_reconstruction as shared
from nanoisa_owned_reconstruction import reconstruct_owned, OwnedAnalyze, OwnedEmit
from nanoisa_reconstruction import Refusal
ROOT=shared.ROOT

class OwnedExclusiveReconstruction(unittest.TestCase):
    setUpClass=shared.OwnedSharedReconstruction.__dict__['setUpClass']
    command=shared.OwnedSharedReconstruction.command
    facts=shared.OwnedSharedReconstruction.facts
    native=shared.OwnedSharedReconstruction.native

    def module(self, compiler=None, text=None, name='exclusive'):
        compiler=compiler or self.compilers[0]
        source=ROOT/'tests/nanoisa/fixtures/source_borrow_exclusive.nano'
        if text is not None:
            source=self.work/(name+'.nano');source.write_text(text)
        module=self.work/(name+'-'+compiler.name+'.nvm')
        self.command([compiler,source,'--emit-nvm','-o',module])
        self.command([ROOT/'bin/nano_vm','--verify-only',module])
        self.command([ROOT/'bin/nano_vm',module])
        return module

    def qualify(self,module):
        facts=self.facts(module)
        self.assertEqual(facts['functions'][1]['param_modes'],[2])
        functions=[OwnedAnalyze(facts,i).run() for i in range(2)]
        emitter=OwnedEmit(functions,'nano',facts['layouts'])
        local_costs=[emitter.nano_local_cost(fn) for fn in functions]
        for language in ('c','nano'):
            output=module.with_suffix('.'+language)
            self.command([ROOT/'bin/nvm2hl','--language',language,module,'-o',output])
            if language=='c': self.native(output,0)
            else:
                self.assertEqual(sum(line.startswith('fn ') for line in output.read_text().splitlines()),2)
                self.assertIn('&mut NlrOwned',output.read_text())
                for compiler in self.compilers:
                    rebuilt=output.with_suffix('.'+compiler.name+'.nvm')
                    self.command([compiler,output,'--emit-nvm','-o',rebuilt])
                    self.command([ROOT/'bin/nano_vm','--verify-only',rebuilt])
                    self.command([ROOT/'bin/nano_vm',rebuilt])
                    regenerated=self.facts(rebuilt)
                    self.assertEqual(regenerated['layouts'],facts['layouts'])
                    self.assertEqual([fn['locals'] for fn in regenerated['functions']],local_costs)
                    native=rebuilt.with_suffix('.c')
                    self.command([ROOT/'bin/nvm2c',rebuilt,'-o',native])
                    self.native(native,0)

    def test_existing_exclusive_fixture(self):
        for compiler in self.compilers:
            with self.subTest(compiler=str(compiler)):
                self.qualify(self.module(compiler))

    def test_endpoints_and_sequential_mutation(self):
        low=-(1<<63);high=(1<<63)-1
        def literal(value): return '(+ -9223372036854775807 -1)' if value==low else str(value)
        def wrap(value): return (value+(1<<63))%(1<<64)-(1<<63)
        for case,(initial,delta) in enumerate(((high,1),(low,-1),(low,low),(7,-3))):
            first=wrap(initial+delta);second=wrap(first+delta)
            text=f'''resource struct Counter {{ value: int, active: bool }}
fn update(view: &mut Counter) -> int {{
 set view.value (+ view.value {literal(delta)})
 set view.active (not view.active)
 return view.value
}}
shadow update {{ let mut owner: Counter = Counter {{ value: {literal(initial)}, active: false }}
 assert (== (update &mut owner) {literal(first)})
 let Counter {{ value, active }} = owner assert (== value {literal(first)}) assert active }}
fn main() -> int {{
 let mut owner: Counter = Counter {{ value: {literal(initial)}, active: false }}
 assert (== (update &mut owner) {literal(first)})
 assert owner.active
 assert (== (update &mut owner) {literal(second)})
 let Counter {{ value, active }} = owner
 assert (== value {literal(second)}) assert (not active)
 return 0
}}
shadow main {{ assert true }}
'''
            for compiler in self.compilers:
                with self.subTest(case=case,compiler=str(compiler)):
                    self.qualify(self.module(compiler,text,'endpoint'+str(case)))

    def test_generated_consistency_shadow_execution(self):
        module=self.module(self.compilers[0]);text=reconstruct_owned(self.facts(module),'nano')
        def body_bounds(start):
            opening=text.index('{',start);depth=1;end=opening+1
            while depth:
                depth+=(text[end]=='{')-(text[end]=='}');end+=1
            return opening,end
        shadow_start=text.index('shadow nlr_f1')
        begin,end=body_bounds(shadow_start)
        shadow_body=text[begin+1:end-1]
        a,b=body_bounds(text.index('fn main()'))
        selected=text[:a+1]+shadow_body+'\n return 0\n'+text[b-1:]
        source=self.work/'selected-generated-shadow.nano';source.write_text(selected)
        for compiler in self.compilers:
            output=self.work/('selected-shadow-'+compiler.name+'.nvm')
            self.command([compiler,source,'--emit-nvm','-o',output])
            self.command([ROOT/'bin/nano_vm','--verify-only',output])
            self.command([ROOT/'bin/nano_vm',output])
            native=output.with_suffix('.c')
            self.command([ROOT/'bin/nvm2c',output,'-o',native]);self.native(native,0)

    def test_defensive_profile_refusals(self):
        facts=self.facts(self.module(self.compilers[0]))
        for case in ('shared-write','wrong-slot','wrong-field','wrong-type','helper-call','helper-sub','call-mode'):
            changed=copy.deepcopy(facts);helper=changed['functions'][1]
            write=next(i for i in helper['code'] if i['op']=='REF_SET')
            if case=='shared-write':helper['param_modes']=[1]
            elif case=='wrong-slot':write.update(arg=1,args=[1,0])
            elif case=='wrong-field':write['args'][1]=99
            elif case=='wrong-type':write['args'][1]=1
            elif case=='helper-call':write['op']='CALL_REF'
            elif case=='helper-sub':next(i for i in helper['code'] if i['op']=='ADD')['op']='SUB'
            elif case=='call-mode':next(i for i in changed['functions'][0]['code'] if i['op']=='BORROW_LOCAL_EXCLUSIVE')['op']='BORROW_LOCAL_SHARED'
            else:continue
            for language in ('c','nano'):
                with self.subTest(case=case,language=language),self.assertRaises(Refusal):
                    reconstruct_owned(changed,language)

    def test_original_verifier_rejects_malformed_references_atomically(self):
        module=self.module();facts=self.facts(module);raw=module.read_bytes()
        header,count=struct.unpack_from('<II',raw,20)
        sections={struct.unpack_from('<IIQQ',raw,header+24*n)[0]:struct.unpack_from('<IIQQ',raw,header+24*n)[2] for n in range(count)}
        code=facts['functions'][0]['code'];base=sections[6]
        call=next(i for i in code if i['op']=='CALL_REF')
        borrow=next(i for i in code if i['op']=='BORROW_LOCAL_EXCLUSIVE')
        end=next(i for i in code if i['op']=='REGION_END')
        helper=base+facts['functions'][0]['size']
        at=sections[13];layouts=struct.unpack_from('<I',raw,at+4)[0]
        at+=((8+layouts+3)//4)*4+4
        at+=4+8*(1+facts['functions'][0]['locals'])
        parameter=at+4+8
        self.assertEqual(raw[parameter+1],2)
        cases={'parameter-shared':(parameter+1,b'\x01'),
               'parameter-nominal':(parameter+4,struct.pack('<I',1)),
               'borrowed-load':(helper,b'\x10\x00\x00\x00\x00'),
               'borrowed-move':(helper,b'\x0b\x00\x00\x00\x00'),
               'borrowed-write':(helper,b'\x1b'),
               'call-reference':(base+call['pc']+5,struct.pack('<H',255)),
               'call-target':(base+call['pc']+1,struct.pack('<I',0)),
               'absent-owner':(base+borrow['pc']+3,struct.pack('<H',0)),
               'missing-end':(base+end['pc'],b'\x00'),
               'helper-reference':(helper+1,struct.pack('<H',255)),
               'helper-field':(helper+3,struct.pack('<H',255)),
               'missing-authority':(sections[13]+8,b'\x00')}
        for name,(offset,payload) in cases.items():
            with self.subTest(name=name):
                changed=bytearray(raw);changed[offset:offset+len(payload)]=payload
                self.assertNotEqual(changed,raw)
                struct.pack_into('<I',changed,36,zlib.crc32(changed[header:]))
                bad=self.work/(name+'.nvm');bad.write_bytes(changed)
                self.assertNotEqual(self.command([ROOT/'bin/nano_vm','--verify-only',bad],None).returncode,0)
                for language in ('c','nano'):
                    output=self.work/('prior.'+language);output.write_text('prior checked source')
                    result=self.command([ROOT/'bin/nvm2hl','--language',language,bad,'-o',output],None)
                    self.assertNotEqual(result.returncode,0)
                    self.assertNotIn('checksum',result.stderr.lower())
                    self.assertEqual(output.read_text(),'prior checked source')


    def test_inline_placement_and_unchanged_helper(self):
        import hashlib
        from nanoisa_owned_reconstruction import OwnedAnalyze, OwnedEmit
        from nanoisa_reconstruction import Expr, INT, NANO_INTEGER_HELPERS
        self.assertEqual(hashlib.sha256(NANO_INTEGER_HELPERS['add'].encode()).hexdigest(),
                         '4c512271c5710e292930601d4cfd302341be24dc5253340e7c486c128267877e')
        facts=self.facts(self.module())
        functions=[OwnedAnalyze(facts,i).run() for i in range(2)]
        helper=functions[1]
        at=next(i for i,s in enumerate(helper.body) if s[0]=='let' and s[2].kind=='arithmetic')
        original=helper.body[at];value=original[2]
        for replacement in [('return',value),('let',original[1],Expr(INT,'arithmetic','add',(value,value.args[1])))]:
            helper.body[at]=replacement
            with self.assertRaisesRegex(Refusal,'explicit inline statement boundary'):
                OwnedEmit(functions,'nano',facts['layouts'])
        helper.body[at]=original

    def test_two_function_budget_refusal_preserves_output(self):
        module=self.module()
        assembly=self.command([ROOT/'bin/nanoisa','dump',module]).stdout
        before='  PUSH_I64 12\n  STORE_LOCAL 0'
        self.assertEqual(assembly.count(before),1)
        assembly=assembly.replace(before,'  PUSH_I64 12\n  PUSH_I64 1\n  MUL\n  STORE_LOCAL 0')
        source=self.work/'exclusive-extra-helper.nasm';source.write_text(assembly)
        module=source.with_suffix('.nvm')
        self.command([ROOT/'bin/nanoisa','asm',source,'-o',module])
        self.command([ROOT/'bin/nano_vm','--verify-only',module])
        self.command([ROOT/'bin/nano_vm',module])
        c=module.with_suffix('.c');self.command([ROOT/'bin/nvm2hl','--language','c',module,'-o',c]);self.native(c,0)
        output=module.with_suffix('.nano');output.write_text('prior checked source')
        result=self.command([ROOT/'bin/nvm2hl','--language','nano',module,'-o',output],None)
        self.assertNotEqual(result.returncode,0)
        self.assertIn('two-function borrowed Nano profile',result.stderr)
        self.assertEqual(output.read_text(),'prior checked source')

    def test_assertion_after_mutation_is_not_successful_cleanup(self):
        from tests.test_owned_void_reconstruction import OwnedVoidReconstruction
        original=(ROOT/'tests/nanoisa/fixtures/source_borrow_exclusive.nano').read_text()
        text=original.replace(' assert owner.active',' assert false')
        self.assertNotEqual(text,original)
        source=self.work/'exclusive-assert.nano';source.write_text(text)
        for compiler in self.compilers:
            module=source.with_suffix('.'+compiler.name+'.nvm')
            self.command([compiler,source,'--emit-nvm','-o',module])
            OwnedVoidReconstruction.vm_failure(self,module)
            for language in ('c','nano'):
                output=module.with_suffix('.'+language)
                self.command([ROOT/'bin/nvm2hl','--language',language,module,'-o',output])
                if language=='c':OwnedVoidReconstruction.native_failure(self,output,'Assertion failed')
                else:
                    for rebuild in self.compilers:
                        regenerated=output.with_suffix('.'+rebuild.name+'.nvm')
                        self.command([rebuild,output,'--emit-nvm','-o',regenerated])
                        OwnedVoidReconstruction.vm_failure(self,regenerated)
                        native=regenerated.with_suffix('.c')
                        self.command([ROOT/'bin/nvm2c',regenerated,'-o',native])
                        OwnedVoidReconstruction.native_failure(self,native,None)

    def test_overlapping_reference_holds_refuse_before_publication(self):
        module=self.module();facts=self.facts(module)
        borrow=next(i for i in facts['functions'][0]['code'] if i['op']=='BORROW_LOCAL_EXCLUSIVE')
        reference,slot=borrow['args']
        assembly=self.command([ROOT/'bin/nanoisa','dump',module]).stdout
        token=f'  BORROW_LOCAL_EXCLUSIVE {reference} {slot}\n'
        self.assertIn(token,assembly)
        for mode in ('SHARED','EXCLUSIVE'):
            source=self.work/('overlap-'+mode+'.nasm')
            source.write_text(assembly.replace(token,token+f'  BORROW_LOCAL_{mode} {reference+1} {slot}\n',1))
            bad=source.with_suffix('.nvm')
            bad.write_text('prior checked module')
            result=self.command([ROOT/'bin/nanoisa','asm',source,'-o',bad],None)
            self.assertNotEqual(result.returncode,0)
            self.assertIn('Assembled module failed verification',result.stderr)
            self.assertIn('available scalar-leaf resource owner',result.stderr)
            self.assertEqual(bad.read_text(),'prior checked module')

    def test_shared_output_is_byte_identical(self):
        source=ROOT/'tests/nanoisa/fixtures/source_borrow_shared.nano'
        module=self.work/'shared-unchanged.nvm'
        self.command([self.compilers[0],source,'--emit-nvm','-o',module])
        facts=self.facts(module)
        for language in ('c','nano'):
            baseline=ROOT/'docs/evidence/owned-exclusive-helper-20261019'/('shared-baseline.'+language+'.txt')
            self.assertEqual(reconstruct_owned(facts,language),baseline.read_text())

    def test_generated_local_boundary_is_checked_before_publication(self):
        for count in (15,16):
            updates='\n'.join(' set view.value (+ view.value 1)' for _ in range(count))
            text=f'''resource struct Counter {{ value: int, active: bool }}
fn update(view: &mut Counter) -> int {{
{updates}
 return view.value
}}
shadow update {{
 let mut owner: Counter = Counter {{ value: 0, active: false }}
 assert (== (update &mut owner) {count})
 let Counter {{ value, active }} = owner
 assert (== value {count}) assert (not active)
}}
fn main() -> int {{
 let mut owner: Counter = Counter {{ value: 0, active: false }}
 assert (== (update &mut owner) {count})
 let Counter {{ value, active }} = owner
 assert (== value {count}) assert (not active) return 0
}}
shadow main {{ assert true }}
'''
            for compiler in self.compilers:
                module=self.module(compiler,text,'local-bound'+str(count))
                facts=self.facts(module)
                self.assertEqual(facts['functions'][1]['locals'],1)
                functions=[OwnedAnalyze(facts,i).run() for i in range(2)]
                emitter=OwnedEmit(functions,'nano',facts['layouts'])
                self.assertEqual(emitter.nano_local_cost(functions[1]),2+8*count)
                if count==15:
                    self.qualify(module)
                    continue
                output=module.with_suffix('.nano');output.write_text('prior checked source')
                result=self.command([ROOT/'bin/nvm2hl','--language','nano',module,'-o',output],None)
                self.assertNotEqual(result.returncode,0)
                self.assertIn('128 local slots',result.stderr)
                self.assertEqual(output.read_text(),'prior checked source')
                c=module.with_suffix('.c');self.command([ROOT/'bin/nvm2hl','--language','c',module,'-o',c]);self.native(c,0)
                # I bypass only my output preflight to measure actual producer storage.
                emitter.nano_local_cost=lambda fn:0
                diagnostic=self.work/('diagnostic130-'+compiler.name+'.nano')
                diagnostic.write_text(emitter.run(0))
                rebuilt=diagnostic.with_suffix('.nvm')
                self.command([compiler,diagnostic,'--emit-nvm','-o',rebuilt])
                self.command([ROOT/'bin/nano_vm','--verify-only',rebuilt])
                self.command([ROOT/'bin/nano_vm',rebuilt])
                assembly=self.command([ROOT/'bin/nanoisa','dump',rebuilt]).stdout
                self.assertRegex(assembly,r'(?m)^\.function \S+ 1 130 ')
                self.assertNotEqual(self.command([ROOT/'bin/nanoisa_hl_facts',rebuilt],None).returncode,0)
