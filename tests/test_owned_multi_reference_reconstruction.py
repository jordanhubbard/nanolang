"""I retain actual three-root calls and a checked direct-call shadow witness."""
import copy
import struct
import unittest
import zlib
from tests import test_owned_exclusive_reconstruction as old
from nanoisa_owned_reconstruction import OwnedAnalyze, OwnedEmit, reconstruct_owned
from nanoisa_reconstruction import Refusal
ROOT=old.ROOT

class OwnedMultiReferenceReconstruction(unittest.TestCase):
    setUpClass=old.OwnedExclusiveReconstruction.__dict__['setUpClass']
    command=old.OwnedExclusiveReconstruction.command
    facts=old.OwnedExclusiveReconstruction.facts
    native=old.OwnedExclusiveReconstruction.native

    def module(self, compiler=None, text=None, name='multi', expected=0):
        compiler=compiler or self.compilers[0]
        source=ROOT/'tests/nanoisa/fixtures/source_borrow_multi.nano'
        if text is not None:
            source=self.work/(name+'.nano');source.write_text(text)
        module=self.work/(name+'-'+compiler.name+'.nvm')
        self.command([compiler,source,'--emit-nvm','-o',module])
        self.command([ROOT/'bin/nano_vm','--verify-only',module])
        if expected is not None:self.command([ROOT/'bin/nano_vm',module],expected)
        return module

    def qualify(self,module):
        facts=self.facts(module)
        self.assertEqual(facts['functions'][1]['param_modes'],[2,1,2])
        functions=[OwnedAnalyze(facts,i).run() for i in range(2)]
        emitter=OwnedEmit(functions,'nano',facts['layouts'])
        costs=[emitter.nano_local_cost(fn) for fn in functions]
        for language in ('c','nano'):
            output=module.with_suffix('.'+language)
            self.command([ROOT/'bin/nvm2hl','--language',language,module,'-o',output])
            if language=='c':self.native(output,0)
            else:
                self.assertEqual(sum(line.startswith('fn ') for line in output.read_text().splitlines()),2)
                for compiler in self.compilers:
                    rebuilt=output.with_suffix('.'+compiler.name+'.nvm')
                    self.command([compiler,output,'--emit-nvm','-o',rebuilt])
                    self.command([ROOT/'bin/nano_vm','--verify-only',rebuilt])
                    self.command([ROOT/'bin/nano_vm',rebuilt])
                    regenerated=self.facts(rebuilt)
                    self.assertEqual(regenerated['layouts'],facts['layouts'])
                    self.assertEqual([f['params'] for f in regenerated['functions']],[f['params'] for f in facts['functions']])
                    self.assertEqual([f['locals'] for f in regenerated['functions']],costs)
                    native=rebuilt.with_suffix('.c')
                    self.command([ROOT/'bin/nvm2c',rebuilt,'-o',native]);self.native(native,0)
        return facts,emitter

    def test_existing_source_roundtrips(self):
        for compiler in self.compilers:
            with self.subTest(compiler=str(compiler)):self.qualify(self.module(compiler))

    @staticmethod
    def endpoint_source(first, middle, last, repeats=2):
        def wrap(value):return (value+(1<<63))%(1<<64)-(1<<63)
        def literal(value):return '(+ -9223372036854775807 -1)' if value==-(1<<63) else str(value)
        body=f' let mut one: Left = Left {{ value: {first}, active: false }}\n let middle_owner: Right = Right {{ value: {middle}, active: true, stamp: 41 }}\n let mut three: Last = Last {{ value: {last} }}\n'
        for _ in range(repeats):
            first,last=wrap(first+middle),wrap(last-middle)
            body+=f' assert (== (adjust &mut one &middle_owner &mut three) {literal(wrap(first+last))})\n assert (== one.value {literal(first)}) assert (== three.value {literal(last)})\n'
        body+=f' let Left {{ value, active }} = one assert (== value {literal(first)}) assert (not active)\n let Last {{ value }} = three assert (== value {literal(last)})\n let Right {{ value, active, stamp }} = middle_owner assert (== value {middle}) assert active assert (== stamp 41)\n'
        return 'resource struct Left { value: int, active: bool }\nresource struct Right { value: int, active: bool, stamp: int }\nresource struct Last { value: int }\nfn adjust(first: &mut Left, middle: &Right, last: &mut Last) -> int {\n set first.value (+ first.value middle.value)\n set last.value (- last.value middle.value)\n assert middle.active\n return (+ first.value last.value)\n}\nshadow adjust {\n'+body+'}\nfn main() -> int {\n'+body+' return 0\n}\nshadow main { assert true }\n'

    def test_endpoints_distinct_nominals_and_unequal_fields(self):
        high=(1<<63)-1
        for index,values in enumerate(((high,1,-high),(-high,-1,high),(7,-3,12))):
            for compiler in self.compilers:
                with self.subTest(case=index,compiler=str(compiler)):
                    self.qualify(self.module(compiler,self.endpoint_source(*values),'endpoint'+str(index)))

    def test_existing_caller_branches_and_loops_preserve_holds(self):
        text=(ROOT/'tests/nanoisa/fixtures/source_borrow_multi.nano').read_text()
        head,main=text.split('fn main()',1)
        call=' assert (== (adjust &mut three &middle_owner &mut one) 30)'
        main=main.replace(call,' let mut run: bool = true\n while run {\n if run {\n'+call+'\n }\n set run false\n }\n while false {\n'+call+'\n }')
        for compiler in self.compilers:self.qualify(self.module(compiler,head+'fn main()'+main,'caller-control'))

    def test_generated_shadow_is_direct_and_consumes_witness(self):
        module=self.module();facts=self.facts(module)
        functions=[OwnedAnalyze(facts,i).run() for i in range(2)]
        emitter=OwnedEmit(functions,'nano',facts['layouts'])
        self.assertEqual([[v.value for v in row] for row in emitter.witness],[[10,True],[3,True],[20,False]])
        text=emitter.run(0)
        def bounds(start):
            begin=text.index('{',start);end=begin+1;depth=1
            while depth:depth+=(text[end]=='{')-(text[end]=='}');end+=1
            return begin,end
        begin,end=bounds(text.index('shadow nlr_f1'))
        shadow=text[begin+1:end-1]
        self.assertEqual(shadow.count('(nlr_f1'),2)
        self.assertNotIn('(main)',shadow)
        begin,end=bounds(text.index('fn main()'))
        selected=text[:begin+1]+shadow+'\n return 0\n'+text[end-1:]
        for compiler in self.compilers:
            module=self.module(compiler,selected,'selected-shadow')
            selected_facts=self.facts(module)
            self.assertEqual(selected_facts['functions'][0]['locals'],14+6*sum(len(v) for v in emitter.witness))
            native=module.with_suffix('.c')
            self.command([ROOT/'bin/nvm2c',module,'-o',native]);self.native(native,0)

    def test_generated_local_boundary_precedes_publication(self):
        for count in (11,12):
            text=self.endpoint_source(0,0,0,1)
            update=' set first.value (+ first.value middle.value)'
            text=text.replace(update,'\n'.join([update]*count),1)
            for compiler in self.compilers:
                module=self.module(compiler,text,'local-bound'+str(count))
                facts=self.facts(module)
                self.assertLess(facts['functions'][1]['locals'],128)
                functions=[OwnedAnalyze(facts,i).run() for i in range(2)]
                emitter=OwnedEmit(functions,'nano',facts['layouts'])
                cost=emitter.nano_local_cost(functions[1])
                self.assertEqual(cost,22+9*count)
                if cost<=128:
                    self.qualify(module)
                    continue
                output=module.with_suffix('.nano');output.write_text('prior checked source')
                result=self.command([ROOT/'bin/nvm2hl','--language','nano',module,'-o',output],None)
                self.assertNotEqual(result.returncode,0);self.assertIn('128 local slots',result.stderr)
                self.assertEqual(output.read_text(),'prior checked source')
                emitter.nano_local_cost=lambda function:0
                diagnostic=self.work/('diagnostic-bound-'+compiler.name+'.nano');diagnostic.write_text(emitter.run(0))
                rebuilt=diagnostic.with_suffix('.nvm')
                self.command([compiler,diagnostic,'--emit-nvm','-o',rebuilt])
                self.command([ROOT/'bin/nano_vm','--verify-only',rebuilt]);self.command([ROOT/'bin/nano_vm',rebuilt])
                assembly=self.command([ROOT/'bin/nanoisa','dump',rebuilt]).stdout
                self.assertRegex(assembly,rf'(?m)^\.function \S+ 3 {cost} ')
                self.assertNotEqual(self.command([ROOT/'bin/nanoisa_hl_facts',rebuilt],None).returncode,0)

    def test_shadow_local_budget_has_actual_boundary(self):
        for extra_left in (False,True):
            text=self.endpoint_source(0,0,0,1)
            additional=', '.join(f'extra{i}: int' for i in range(13))
            entries=', '.join(f'extra{i}: {i}' for i in range(13))
            names=', '.join(f'extra{i}' for i in range(13))
            text=text.replace('stamp: int }','stamp: int, '+additional+' }').replace('stamp: 41 }','stamp: 41, '+entries+' }').replace('active, stamp }','active, stamp, '+names+' }')
            if extra_left:
                text=text.replace('struct Left { value: int, active: bool }','struct Left { value: int, active: bool, marker: int }').replace('Left { value: 0, active: false }','Left { value: 0, active: false, marker: 7 }').replace('let Left { value, active }','let Left { value, active, marker }')
            for compiler in self.compilers:
                module=self.module(compiler,text,'shadow-bound'+str(int(extra_left)))
                facts=self.facts(module);functions=[OwnedAnalyze(facts,i).run() for i in range(2)]
                emitter=OwnedEmit(functions,'nano',facts['layouts'])
                count=emitter.multi_shadow_local_cost()
                self.assertEqual(count,134 if extra_left else 128)
                if extra_left:
                    with self.assertRaisesRegex(Refusal,'multi-reference shadow'):emitter.multi_shadow()
                    output=module.with_suffix('.nano');output.write_text('prior checked source')
                    self.assertNotEqual(self.command([ROOT/'bin/nvm2hl','--language','nano',module,'-o',output],None).returncode,0)
                    self.assertEqual(output.read_text(),'prior checked source')
                # I bypass output guards only to measure selected-shadow producer storage.
                emitter.nano_local_cost=lambda function:0
                emitter.multi_shadow_local_cost=lambda:0
                rendered=emitter.run(0)
                def bounds(start):
                    begin=rendered.index('{',start);end=begin+1;depth=1
                    while depth:depth+=(rendered[end]=='{')-(rendered[end]=='}');end+=1
                    return begin,end
                begin,end=bounds(rendered.index('shadow nlr_f1'))
                shadow=rendered[begin+1:end-1]
                begin,end=bounds(rendered.index('fn main()'))
                selected=rendered[:begin+1]+shadow+'\n return 0\n'+rendered[end-1:]
                rebuilt=self.module(compiler,selected,'diagnostic-shadow'+str(count))
                assembly=self.command([ROOT/'bin/nanoisa','dump',rebuilt]).stdout
                self.assertRegex(assembly,rf'(?m)^\.function \S+ 0 {count} ')
                if extra_left:self.assertNotEqual(self.command([ROOT/'bin/nanoisa_hl_facts',rebuilt],None).returncode,0)
                else:self.assertEqual(self.facts(rebuilt)['functions'][0]['locals'],count)

    def test_single_exclusive_output_is_byte_identical(self):
        import hashlib
        text=(ROOT/'tests/nanoisa/fixtures/source_borrow_exclusive.nano').read_text()
        facts=self.facts(self.module(self.compilers[0],text,'single-golden'))
        expected={'c':'7213a41579bf1fb0376f3090b6a7cc657364a9228570a942e3fafda99a98a54e',
                  'nano':'ed6b789bd3a3a5c85352e867501e4794a5dcafda5b1cfac5fb7a74c30f2e1601'}
        # I retain these exact bytes from owned-exclusive-helper-20261019/artifacts.json.
        for language,digest in expected.items():
            self.assertEqual(hashlib.sha256(reconstruct_owned(facts,language).encode()).hexdigest(),digest)

    def test_actual_computed_witness_and_transformed_assert_refuse(self):
        original=(ROOT/'tests/nanoisa/fixtures/source_borrow_multi.nano').read_text()
        head,main=original.split('fn main()',1)
        cases={'computed-owner':head+'fn main()'+main.replace('value: 3, active: true','value: (+ 1 2), active: true'),
               'transformed-assert':original.replace(' assert middle.active',' assert (not (not middle.active))')}
        for name,text in cases.items():
            for compiler in self.compilers:
                module=self.module(compiler,text,name)
                for language in ('c','nano'):
                    output=self.work/('prior.'+language);output.write_text('prior checked source')
                    result=self.command([ROOT/'bin/nvm2hl','--language',language,module,'-o',output],None)
                    self.assertNotEqual(result.returncode,0)
                    self.assertIn('actual first-call literal owner snapshots' if name=='computed-owner' else 'true actual shared-BOOL assertion witness',result.stderr)
                    self.assertEqual(output.read_text(),'prior checked source')

    def test_false_first_witness_refuses_atomically(self):
        text=(ROOT/'tests/nanoisa/fixtures/source_borrow_multi.nano').read_text()
        # I preserve its successful declared helper shadow and alter runtime entry only.
        head,main=text.split('fn main()',1)
        main=main.replace('Right { value: 3, active: true }','Right { value: 3, active: false }')
        for compiler in self.compilers:
            module=self.module(compiler,head+'fn main()'+main,'false-witness',None)
            from tests.test_owned_void_reconstruction import OwnedVoidReconstruction
            OwnedVoidReconstruction.vm_failure(self,module)
            for language in ('c','nano'):
                output=self.work/('prior.'+language);output.write_text('prior checked source')
                result=self.command([ROOT/'bin/nvm2hl','--language',language,module,'-o',output],None)
                self.assertNotEqual(result.returncode,0)
                self.assertIn('shared-BOOL assertion witness',result.stderr)
                self.assertEqual(output.read_text(),'prior checked source')

    def test_later_false_call_retains_runtime_assertion(self):
        text=(ROOT/'tests/nanoisa/fixtures/source_borrow_multi.nano').read_text()
        head,main=text.split('fn main()',1)
        main=main.replace(' assert (== (adjust &mut three &middle_owner &mut one) 30)',
                          ' let bad: Right = Right { value: 0, active: false }\n assert (== (adjust &mut three &bad &mut one) 30)')
        main=main.replace(' return 0',' let Right { value, active } = bad\n return 0')
        from tests.test_owned_void_reconstruction import OwnedVoidReconstruction
        for compiler in self.compilers:
            module=self.module(compiler,head+'fn main()'+main,'later-false',None)
            OwnedVoidReconstruction.vm_failure(self,module)
            for language in ('c','nano'):
                output=module.with_suffix('.'+language)
                self.command([ROOT/'bin/nvm2hl','--language',language,module,'-o',output])
                if language=='c':OwnedVoidReconstruction.native_failure(self,output,'Assertion failed')
                else:
                    for producer in self.compilers:
                        rebuilt=output.with_suffix('.'+producer.name+'.nvm')
                        self.command([producer,output,'--emit-nvm','-o',rebuilt])
                        OwnedVoidReconstruction.vm_failure(self,rebuilt)
                        native=rebuilt.with_suffix('.c')
                        self.command([ROOT/'bin/nvm2c',rebuilt,'-o',native])
                        OwnedVoidReconstruction.native_failure(self,native,None)

    def test_defensive_exact_reference_and_witness_controls(self):
        facts=self.facts(self.module())
        for case in ('empty-borrow','shared-write','wrong-slot','wrong-field','wrong-mode','overlap','reference-gap','wrong-base','unknown-witness','false-witness','mutable-assert'):
            changed=copy.deepcopy(facts);entry,helper=changed['functions']
            writes=[i for i in helper['code'] if i['op']=='REF_SET']
            borrows=[i for i in entry['code'] if i['op'].startswith('BORROW_LOCAL')][:3]
            if case=='empty-borrow':borrows[0]['args']=[]
            elif case=='shared-write':writes[0].update(arg=1,args=[1,0])
            elif case=='wrong-slot':writes[0].update(arg=3,args=[3,0])
            elif case=='wrong-field':writes[0]['args'][1]=99
            elif case=='wrong-mode':helper['param_modes']=[2,2,2]
            elif case=='overlap':borrows[2]['args'][1]=borrows[0]['args'][1]
            elif case=='reference-gap':borrows[1]['args'][0]=9
            elif case=='wrong-base':next(i for i in entry['code'] if i['op']=='CALL_REF')['args'][1]=9
            elif case=='false-witness':[i for i in entry['code'] if i['op']=='PUSH_BOOL'][1]['arg']=0
            elif case=='unknown-witness':
                instruction=next(i for i in entry['code'] if i['op']=='PUSH_I64')
                instruction.update(op='LOAD_LOCAL',arg=0,args=[0])
            elif case=='mutable-assert':
                read=next(i for i in helper['code'] if i['op']=='REF_GET' and i['args']==[1,1])
                read.update(arg=0,args=[0,1])
            for language in ('c','nano'):
                with self.subTest(case=case,language=language),self.assertRaises(Refusal):reconstruct_owned(changed,language)

    def test_original_verifier_rejects_reference_corruption(self):
        module=self.module();facts=self.facts(module);raw=module.read_bytes()
        header,count=struct.unpack_from('<II',raw,20)
        sections={struct.unpack_from('<IIQQ',raw,header+24*n)[0]:struct.unpack_from('<IIQQ',raw,header+24*n)[2] for n in range(count)}
        code=facts['functions'][0]['code'];base=sections[6]
        borrows=[i for i in code if i['op'].startswith('BORROW_LOCAL')][:3]
        call=next(i for i in code if i['op']=='CALL_REF')
        at=sections[13];layouts=struct.unpack_from('<I',raw,at+4)[0]
        at+=((8+layouts+3)//4)*4+4
        at+=4+8*(1+facts['functions'][0]['locals'])
        parameter=at+4+8
        self.assertEqual(raw[parameter+8+1],1)
        for name,offset,payload in (
            ('parameter-mode',parameter+8+1,b'\x02'),
            ('parameter-nominal',parameter+8+4,struct.pack('<I',0)),
            ('overlap',base+borrows[2]['pc']+3,struct.pack('<H',borrows[0]['args'][1])),
            ('gap',base+borrows[1]['pc']+1,struct.pack('<H',9)),
            ('base',base+call['pc']+5,struct.pack('<H',9))):
            changed=bytearray(raw);changed[offset:offset+len(payload)]=payload
            self.assertNotEqual(changed,raw);struct.pack_into('<I',changed,36,zlib.crc32(changed[header:]))
            bad=self.work/(name+'.nvm');bad.write_bytes(changed)
            self.assertNotEqual(self.command([ROOT/'bin/nano_vm','--verify-only',bad],None).returncode,0)
            for language in ('c','nano'):
                output=self.work/('prior.'+language);output.write_text('prior checked source')
                self.assertNotEqual(self.command([ROOT/'bin/nvm2hl','--language',language,bad,'-o',output],None).returncode,0)
                self.assertEqual(output.read_text(),'prior checked source')

if __name__=='__main__':unittest.main()
