"""I preserve immutable parameter strings and ordered owned-graph output bytes."""
import copy
import json
import os
from pathlib import Path
import shlex
import signal
from collections import Counter
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'scripts'))
from nanoisa_owned_reconstruction import OwnedAnalyze, OwnedEmit, reconstruct_owned
from nanoisa_owned_strings import literal_facts
from nanoisa_reconstruction import Refusal
from tests import test_file_source_plan as retained_runner
from tests.native_process import native_link_flags, native_compile_timeout

BASELINE = ROOT/'docs/evidence/owned-string-reconstruction-baseline-20261019'


class OwnedStringReconstruction(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.work = Path(tempfile.mkdtemp(prefix='owned-string-reconstruction-'))
        cls.serial = 0
        cls.compilers = [ROOT/'bin/nano_virt'] + [Path(p) for p in shlex.split(
            os.environ.get('NANOLANG_OWNED_RECONSTRUCTION_COMPILERS', ''))]
        print('I retain owned STRING evidence at '+str(cls.work), flush=True)

    def command(self, args, expected=0, build=False):
        type(self).serial += 1
        name = str(self.serial)
        try:
            out, err = retained_runner.FileSourcePlan.command.__func__(type(self), name, args, timeout=native_compile_timeout(default=900) if build else 180)
        except AssertionError:
            status = json.loads((self.work/(name+'-status.json')).read_text())
            if status['timeout'] or status['cleanup_errors'] or not status['group_absent'] or status.get('launch_or_wait_error'):
                raise
            out = (self.work/(name+'.stdout')).read_bytes()
            err = (self.work/(name+'.stderr')).read_bytes()
        status = json.loads((self.work/(name+'-status.json')).read_text())
        if expected is not None: self.assertEqual(status['returncode'], expected, (out[-4000:], err[-4000:]))
        return status['returncode'], out, err

    def facts(self, module):
        return json.loads(self.command([ROOT/'bin/nanoisa_hl_facts', module])[1])

    def module(self, compiler, name, source=None):
        path = ROOT/'examples/language/nl_affine_resource_demo.nano'
        if source is not None:
            path = self.work/(name+'.nano');path.write_text(source)
        output = self.work/(name+'-'+compiler.name+'.nvm')
        self.command([compiler, path, '--emit-nvm', '-o', output])
        self.command([ROOT/'bin/nano_vm', '--verify-only', output])
        return output

    def native(self, source, expected_output, failure=False, reconstructed=False):
        for opt in ('-O0', '-O2'):
            output = source.with_suffix('.'+opt[1:]+'.exe')
            self.command([*shlex.split(os.environ.get('NANO_NATIVE_TEST_CC', 'cc')), '-std=c11', opt,
                '-DNDEBUG', '-Wall', '-Wextra', '-Werror', '-fsanitize=address,undefined',
                '-fno-sanitize-recover=all', source, ROOT/'bin/nano_aot_runtime.o', '-lm',
                *native_link_flags(), '-o', output], build=True)
            code, stdout, stderr = self.command([output], None if failure else 0)
            if failure:
                self.assertEqual(code, -signal.SIGABRT if reconstructed else 1, stderr)
                if reconstructed:self.assertIn(b'Assertion failed', stderr)
            for diagnostic in (b'AddressSanitizer',b'UndefinedBehaviorSanitizer',b'LeakSanitizer',b'runtime error:'):
                self.assertNotIn(diagnostic,stderr)
            self.assertEqual(stdout, expected_output)

    def qualify(self, module, expected, failure=False):
        code, stdout, stderr = self.command([ROOT/'bin/nano_vm', module], None if failure else 0)
        if failure:
            self.assertEqual(code,1);self.assertIn(b'Assertion failed',stderr)
        self.assertEqual(stdout, expected)
        original = module.with_suffix('.original.c')
        self.command([ROOT/'bin/nvm2c', module, '-o', original]);self.native(original, expected, failure)
        facts = self.facts(module)
        functions = [OwnedAnalyze(facts, i).run() for i in range(len(facts['functions']))]
        emitter = OwnedEmit(functions, 'nano', facts['layouts'])
        costs = [emitter.nano_local_cost(fn) for fn in functions]
        for language in ('c', 'nano'):
            output = module.with_suffix('.rebuilt.'+language)
            self.command([ROOT/'bin/nvm2hl', '--language', language, module, '-o', output])
            text = output.read_text()
            self.assertNotRegex(text, r'\bgoto\b|\bswitch\b|nvm_blob|nano_vm')
            if language == 'c':self.native(output, expected, failure, reconstructed=True)
            else:
                self.assertNotRegex(text, r'let(?: mut)? [^\n:]+: string')
                self.assertNotIn('\nshadow ', text)
                for compiler in self.compilers:
                    rebuilt = output.with_suffix('.'+compiler.name+'.nvm')
                    self.command([compiler, output, '--emit-nvm', '-o', rebuilt])
                    self.command([ROOT/'bin/nano_vm', '--verify-only', rebuilt])
                    code, stdout, stderr = self.command([ROOT/'bin/nano_vm', rebuilt], None if failure else 0)
                    if failure:
                        self.assertEqual(code,1);self.assertIn(b'Assertion failed',stderr)
                    self.assertEqual(stdout, expected)
                    regenerated = self.facts(rebuilt)
                    self.assertEqual([fn['locals'] for fn in regenerated['functions']], costs)
                    native = rebuilt.with_suffix('.c')
                    self.command([ROOT/'bin/nvm2c', rebuilt, '-o', native]);self.native(native, expected, failure)
        return facts, costs

    def test_unchanged_ordinary_example(self):
        for compiler in self.compilers:
            facts, counts = self.qualify(self.module(compiler, 'ordinary'), b'audit.log\nresource closed\n')
            self.assertEqual(counts, [3, 5, 11])
            self.assertEqual(facts['functions'][1]['params'], [[5, 4294967295]])
            self.assertEqual(facts['functions'][1]['local_types'], [[5, 4294967295], [1, 4294967295]])
            self.assertEqual(literal_facts(facts), {5:b'audit.log', 6:b'resource closed'})

    def test_independent_original_shadow_executable(self):
        assembly = BASELINE/'original-shadows.nasm'
        driver = os.environ.get('NANOLANG_OWNED_STRING_SHADOW_EMITTER')
        if driver:
            assembly = self.work/'original-shadow.nasm'
            assembly.write_bytes(self.command([driver, ROOT/'examples/language/nl_affine_resource_demo.nano', 0, 'raw'])[1])
        module = self.work/'original-shadows.nvm'
        self.command([ROOT/'bin/nanoisa', 'asm', assembly, '-o', module])
        facts, costs = self.qualify(module, b'audit.log\nresource closed\nresource closed\naudit.log\nresource closed\n')
        self.assertEqual(len(facts['functions']), 4);self.assertEqual(costs, [11, 5, 11, 3])

    def test_exact_bytes_forwarding_and_positional_owners(self):
        from tests.test_source_borrow_emission import SourceBorrowEmission
        source = SourceBorrowEmission.owned_string_fixture(None).replace(' café', ' \x01\x7f café')
        expected = b'A\tB\rC\n"\'\\0\\q \x01\x7f caf\xc3\xa9\n'
        for compiler in self.compilers:
            facts,_=self.qualify(self.module(compiler, 'forward-bytes', source), expected)
            uses=Counter(ins['arg'] for fn in facts['functions'] for ins in fn['code'] if ins['op']=='PUSH_STR')
            self.assertTrue(any(n>1 for n in uses.values()))
            self.assertEqual(set(uses),set(literal_facts(facts)))

    def test_assert_failure_prefix_and_no_later_print(self):
        template = '''resource struct Leaf { value: int }
fn consume(text: string, owner: Leaf) -> void {
 BEFORE
 (print text)
 let Leaf { value } = owner
 assert (== value 7)
 (println "after")
}
shadow consume { assert true }
fn main() -> int { (consume "before" Leaf { value: 8 }) return 0 }
shadow main { assert true }
'''
        for compiler in self.compilers:
            for early in (False, True):
                source = template.replace('BEFORE', 'assert false' if early else '')
                self.qualify(self.module(compiler, 'failure-'+str(early), source), b'' if early else b'before', True)

    def test_canonical_roundtrip_and_stripped_advisory_metadata(self):
        module=self.module(self.compilers[0],'transport')
        text=self.command([ROOT/'bin/nanoisa','dump',module])[1]
        for cycle in range(2):
            asm=self.work/('cycle-'+str(cycle)+'.nasm');asm.write_bytes(text)
            rebuilt=asm.with_suffix('.nvm');self.command([ROOT/'bin/nanoisa','asm',asm,'-o',rebuilt])
            self.assertEqual(module.read_bytes(),rebuilt.read_bytes())
            text=self.command([ROOT/'bin/nanoisa','dump',rebuilt])[1]
        stripped=self.work/'without-advisory.nasm'
        stripped.write_bytes(b'\n'.join(line for line in text.splitlines() if not line.startswith(b'.metadata'))+b'\n')
        rebuilt=stripped.with_suffix('.nvm');self.command([ROOT/'bin/nanoisa','asm',stripped,'-o',rebuilt])
        self.command([ROOT/'bin/nano_vm','--verify-only',rebuilt])
        facts=self.facts(module);other=self.facts(rebuilt)
        self.assertEqual(literal_facts(facts),literal_facts(other))
        for lang in ('c','nano'):self.assertEqual(reconstruct_owned(facts,lang),reconstruct_owned(other,lang))
        self.qualify(rebuilt,b'audit.log\nresource closed\n')

    def test_name_independence_preserves_shared_literal_indexes(self):
        source=(BASELINE/'nano_virt.dump.nasm').read_text().replace('PUSH_STR 5','PUSH_STR 2',1)
        # Index2 is both the main function name and the literal printed by open_file.
        reference=self.work/'shared-name.nasm';reference.write_text(source)
        module=reference.with_suffix('.nvm');self.command([ROOT/'bin/nanoisa','asm',reference,'-o',module])
        self.command([ROOT/'bin/nano_vm','--verify-only',module]);before=self.facts(module)
        referenced=set(literal_facts(before));self.assertIn(2,referenced)
        renamed=source
        for index,name in enumerate(('FileHandle','fd','main','open_file','close_file')):
            if index in referenced:continue
            replacement='renamed_'+str(index)
            renamed=renamed.replace('.string "'+name+'"','.string "'+replacement+'"',1)
            renamed=renamed.replace('.function '+name+' ','.function '+replacement+' ',1)
        renamed='\n'.join(line for line in renamed.splitlines() if not line.startswith('.metadata'))+'\n'
        self.assertIn('.string "main"',renamed);self.assertIn('.function main ',renamed)
        asm=self.work/'renamed-shared-name.nasm';asm.write_text(renamed)
        rebuilt=asm.with_suffix('.nvm');self.command([ROOT/'bin/nanoisa','asm',asm,'-o',rebuilt])
        self.command([ROOT/'bin/nano_vm','--verify-only',rebuilt]);after=self.facts(rebuilt)
        self.assertEqual(literal_facts(before),literal_facts(after))
        for language in ('c','nano'):self.assertEqual(reconstruct_owned(before,language),reconstruct_owned(after,language))
        self.qualify(rebuilt,b'main\nresource closed\n')

    def test_defensive_string_table_and_parameter_refusals(self):
        facts = self.facts(self.module(self.compilers[0], 'defensive'))
        cases=[]
        for field, value in [('index', True), ('length', True), ('length', 0), ('hex', '00'), ('index', -1)]:
            changed=copy.deepcopy(facts);changed['string_literals'][0][field]=value;cases.append(changed)
        changed=copy.deepcopy(facts)
        literal=next(ins for fn in changed['functions'] for ins in fn['code'] if ins['op']=='PUSH_STR')
        literal['arg']=True;literal['args']=[1];changed['string_literals'][0]['index']=1;cases.append(changed)
        changed=copy.deepcopy(facts);changed['string_constant_count']=True;cases.append(changed)
        changed=copy.deepcopy(facts);changed['string_literals'][1]=copy.deepcopy(changed['string_literals'][0]);cases.append(changed)
        changed=copy.deepcopy(facts);changed['string_literals'].pop();cases.append(changed)
        changed=copy.deepcopy(facts);changed['string_literals'][0]['hex']='00'*9;cases.append(changed)
        changed=copy.deepcopy(facts);changed['functions'][1]['local_types'][1]=[5,4294967295];cases.append(changed)
        changed=copy.deepcopy(facts);changed['functions'][1]['code'][0]['op']='STORE_LOCAL';cases.append(changed)
        changed=copy.deepcopy(facts);changed['functions'][1]['result_type']=[5,4294967295];changed['functions'][1]['result']=5;cases.append(changed)
        for changed in cases:
            for language in ('c','nano'):
                with self.assertRaises(Refusal):reconstruct_owned(changed,language)

    def test_literal_payload_bound_and_atomic_overlimit(self):
        source = (BASELINE/'nano_virt.dump.nasm').read_text()
        for extra in (0,1):
            asm=self.work/('literal-cap-'+str(extra)+'.nasm')
            asm.write_text(source.replace('.string "audit.log"', '.string "'+'x'*(1048576-15+extra)+'"',1))
            module=asm.with_suffix('.nvm');self.command([ROOT/'bin/nanoisa','asm',asm,'-o',module])
            self.command([ROOT/'bin/nano_vm','--verify-only',module])
            for language in ('c','nano'):
                output=self.work/('literal-cap-'+str(extra)+'.'+language);output.write_bytes(b'prior source')
                result=self.command([ROOT/'bin/nvm2hl','--language',language,module,'-o',output],None)
                if extra:
                    self.assertNotEqual(result[0],0);self.assertEqual(output.read_bytes(),b'prior source')
                else:self.assertEqual(result[0],0,result[2])
        # I do not compile or execute megabyte output to establish a publication bound.

    def test_invalid_utf8_is_target_specific(self):
        assembly=self.work/'invalid-utf8.nasm'
        assembly.write_text((BASELINE/'nano_virt.dump.nasm').read_text().replace('.string "audit.log"', '.string "\\xff"',1))
        module=assembly.with_suffix('.nvm');self.command([ROOT/'bin/nanoisa','asm',assembly,'-o',module])
        self.command([ROOT/'bin/nano_vm','--verify-only',module])
        self.assertEqual(self.command([ROOT/'bin/nano_vm',module])[1],b'\xff\nresource closed\n')
        c=module.with_suffix('.c');self.command([ROOT/'bin/nvm2hl','--language','c',module,'-o',c])
        self.native(c,b'\xff\nresource closed\n')
        nano=module.with_suffix('.nano');nano.write_bytes(b'prior source')
        result=self.command([ROOT/'bin/nvm2hl','--language','nano',module,'-o',nano],None)
        self.assertNotEqual(result[0],0);self.assertIn(b'UTF-8',result[2]);self.assertEqual(nano.read_bytes(),b'prior source')

    def test_original_verifier_rejects_wrong_print_argument_and_constant(self):
        source=(BASELINE/'nano_virt.dump.nasm').read_text()
        for name,text in [('constant',source.replace('PUSH_STR 5','PUSH_STR 999999',1)),
                          ('argument',source.replace('.parameters 1 string','.parameters 1 int',1)),
                          ('print',source.replace('PUSH_STR 6','PUSH_I64 6',1))]:
            asm=self.work/(name+'.nasm');asm.write_text(text);module=asm.with_suffix('.nvm')
            code,_,_=self.command([ROOT/'bin/nanoisa','asm',asm,'-o',module],None)
            if code==0:self.assertNotEqual(self.command([ROOT/'bin/nano_vm','--verify-only',module],None)[0],0)
            else:self.assertFalse(module.exists())


if __name__=='__main__':unittest.main()
