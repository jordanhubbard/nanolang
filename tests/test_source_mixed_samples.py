"""I preserve every original Samples/PREFIX shadow while checking exact mixed lowering."""
import ast
import os
from pathlib import Path
import subprocess
import unittest
from tests import test_source_borrow_emission as support
from tests.test_owned_record_patterns import PREFIX, OwnedRecordPatterns
ROOT = support.ROOT

class SourceMixedSamples(support.SourceBorrowEmission):
    @classmethod
    def tearDownClass(cls):
        cls.temporary._finalizer.detach()
        print(f'I retain mixed source fixtures at {cls.work}', flush=True)

    def execute_pair(self, module, expected=0, expected_output=None):
        self.command(ROOT/'bin/nano_vm', '--verify-only', module)
        vm = subprocess.run([ROOT/'bin/nano_vm',module],cwd=ROOT,capture_output=True,timeout=180)
        self.assertEqual(vm.returncode,expected,vm.stdout+vm.stderr)
        if expected_output is not None:self.assertEqual(vm.stdout,expected_output)
        source,native=self.work/'mixed-native.c',self.work/'mixed-native'
        self.command(ROOT/'bin/nvm2c',module,'-o',source)
        self.command(os.environ.get('CC','cc'),'-std=c11','-O2','-Wall','-Wextra','-Werror',
                     '-fsanitize=address,undefined','-fno-omit-frame-pointer',source,'-lm','-o',native)
        result=subprocess.run([native],cwd=ROOT,capture_output=True,timeout=30,
                              env={**os.environ,'ASAN_OPTIONS':'detect_leaks=1:halt_on_error=1'})
        self.assertEqual(result.returncode,expected,result.stdout+result.stderr)
        self.assertNotIn(b'Sanitizer',result.stderr)
        if expected_output is not None:self.assertEqual(result.stdout,expected_output)
        if expected==0:self.assertEqual(vm.stdout,result.stdout)

    @staticmethod
    def original():
        tree=ast.parse((ROOT/'tests/test_owned_record_patterns.py').read_text())
        method=next(n for n in ast.walk(tree) if isinstance(n,ast.FunctionDef) and n.name=='test_ordinary_inferred_field_and_alias')
        return PREFIX+ast.literal_eval(method.body[0].value.args[0])

    def test_original_all_shadows_routes_and_equal_metadata(self):
        text=self.original()
        normal,shadow=self.graph_positive('original-samples',text,b'',b'')
        self.assertIn('AGG_PACK 0 1 0 1',normal)
        self.assertIn('ARR_LITERAL 3 2',normal)
        self.assertIn('ARR_GET',normal)
        self.assertNotIn('OWN_PACK 1',normal)
        self.assertIn('OWN_PACK 0',shadow)
        # I execute the original test method unchanged, including all four drivers.
        original=OwnedRecordPatterns('test_ordinary_inferred_field_and_alias')
        original.test_ordinary_inferred_field_and_alias()

    def test_alias_scope_empty_and_constructor_field_order(self):
        text=self.original().replace('let values = record.values',
            'let another: Samples = record let values = another.values')
        text=text.replace('    return 0','    if true { let alias = record.values assert (== (at alias 0) 1.5) }\n    assert (== (at alias 1) 2.5)\n    return 0')
        self.graph_positive('lexical-alias',text,b'',b'')
        empty=self.original().replace('    let values = record.values',
            '    let empty: Samples = Samples { values: [] }\n    let values = record.values')
        self.graph_positive('context-empty',empty,b'',b'')
        ordered=PREFIX+'''struct Samples { first: int, values: array<float>, last: int }
fn main() -> int {
    let record: Samples = Samples { last: (close Handle { fd: 9 }), values: [1.5], first: (close Handle { fd: 7 }) }
    assert (== record.first 7) assert (== record.last 9)
    let values = record.values assert (== (at values 0) 1.5)
    return 0
}
shadow main { assert (== (main) 0) }
'''
        normal,_=self.graph_positive('ordered-fields',ordered,b'',b'')
        self.assertLess(normal.index('PUSH_I64 9'),normal.index('PUSH_I64 7'))

    def test_false_shadows_preserve_prior_publication(self):
        base=self.original()
        for name,text in [('close',base.replace('(close Handle { fd: 7 }) 7','(close Handle { fd: 7 }) 8')),
                          ('main',base.replace('(main) 0','(main) 1'))]:
            source=self.work/f'false-{name}.nano';source.write_text(text)
            for compiler in ('nanoc_c','nanoc_stage1','nanoc_stage2','nano_virt'):
                with self.subTest(shadow=name,compiler=compiler):
                    output=self.work/'prior';output.write_bytes(b'previous verified output')
                    args=[ROOT/'bin'/compiler,source,'-o',output]
                    if compiler=='nano_virt':args.append('--emit-nvm')
                    run=subprocess.run(args,cwd=ROOT,capture_output=True,text=True,timeout=180)
                    self.assertGreater(run.returncode,0,run.stdout+run.stderr)
                    self.assertRegex(run.stdout+run.stderr,r'(?i)shadow|assert')
                    self.assertEqual(output.read_bytes(),b'previous verified output')

    def test_exact_type_nominal_and_profile_refusals(self):
        base=self.original()
        cases={
            'wrong-element':base.replace('[1.5, 2.5]','[1, 2]'),
            'missing':base.replace('Samples { values: [1.5, 2.5] }','Samples {}'),
            'duplicate':base.replace('Samples { values: [1.5, 2.5] }','Samples { values: [1.5], values: [2.5] }'),
            'unknown':base.replace('Samples { values: [1.5, 2.5] }','Samples { other: [1.5, 2.5] }'),
            'nominal':base.replace('fn main()', 'struct Other { values: array<float> }\nfn main()').replace('= Samples {','= Other {'),
            'owner-array':base.replace('struct Samples','resource struct Samples'),
            'untyped-empty':base.replace('let values = record.values','let unused = [] let values = record.values'),
            'assignment':base.replace('let alias = values','let mut alias = values set alias values'),
            'signature':base+'\nfn unsupported(value: Samples) -> int { return 0 } shadow unsupported { assert true }\n',
            'owner-float':base.replace('fd: int','fd: float'),
            'borrow-mix':base+'\nfn read(value: &Handle) -> int { return value.fd } shadow read { assert true }\n',
            'shadow-unsupported':base.replace('shadow main { assert (== (main) 0) }',
                'shadow main { let mut values: array<float> = [1.5] (array_push values 2.5) assert (== (main) 0) }'),
        }
        for name,text in cases.items():
            source=self.work/f'refused-{name}.nano';source.write_text(text)
            for compiler in [ROOT/'bin'/x for x in ('nano_virt','nanoc_stage1','nanoc_stage2')]+self.emitters:
                # Raw program emitters do not select shadow bodies; public compilers do.
                if name == 'shadow-unsupported' and compiler in self.emitters:
                    continue
                with self.subTest(case=name,compiler=compiler.name):
                    output=self.work/'prior';output.write_bytes(b'previous verified output')
                    args=[compiler,source,'-o',output]
                    if compiler not in self.emitters:args.append('--emit-nvm')
                    run=subprocess.run(args,cwd=ROOT,capture_output=True,text=True,timeout=180)
                    self.assertGreater(run.returncode,0,run.stdout+run.stderr)
                    self.assertEqual(output.read_bytes(),b'previous verified output')
                    self.assertNotRegex(run.stdout+run.stderr,r'(?i)parse (?:error|failed)|unexpected token')
                    self.assertRegex(run.stdout+run.stderr,r'(?i)type|field|array|owner|require|unsupported|inference|infer')

def load_tests(loader,standard_tests,pattern):
    return unittest.TestSuite(SourceMixedSamples(name) for name in (
        'test_original_all_shadows_routes_and_equal_metadata',
        'test_alias_scope_empty_and_constructor_field_order',
        'test_false_shadows_preserve_prior_publication',
        'test_exact_type_nominal_and_profile_refusals'))
