"""I retain actual scalar async contracts through C and source-created VM/native routes."""
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from tests import test_bstring_native_contracts as native
from tests import test_bstring_source_vm as source

ROOT = source.ROOT


class AsyncContracts:
    command = native.BstringNativeContracts.command
    exercise = native.BstringNativeContracts.exercise

    def test_unchanged_original_async_fixture(self):
        path = ROOT/'tests/test_async.nano'
        self.exercise(path.read_text(), 'async/await tests PASSED\n', source_path=path)

    def test_scalar_types_primary_binding_and_once_only_effects(self):
        self.exercise('''let mut trace:int=0
fn mark(value:int,digit:int)->int{
 set trace (+ (* trace 10) digit)
 return value
}
shadow mark{set trace 0 assert (== (mark 7 1) 7) assert (== trace 1)}
async fn compute(value:int)->int{
 return (+ await (mark value 1) await (mark 2 2))
}
shadow compute{set trace 0 assert (== (compute 40) 42) assert (== trace 12)}
async fn text(value:string)->string{return await value}
shadow text{assert (== (text "held") "held")}
async fn decimal(value:float)->float{return await value}
shadow decimal{assert (== (decimal 1.5) 1.5)}
async fn truth(value:bool)->bool{return await value}
shadow truth{assert (truth true) assert (not (truth false))}
fn main()->int{
 set trace 0
 assert (== (compute 40) 42)
 assert (== trace 12)
 assert (== (text "retained") "retained")
 assert (== (decimal 2.5) 2.5)
 assert (truth true)
 let async_value:int=await 40 + 2
 let awaitable:int=await await async_value
 assert (== awaitable 42)
 return 0
}
shadow main{assert (== (main) 0)}
''')

    def test_malformed_types_and_failed_shadows_preserve_prior_output(self):
        cases = {
            'missing-fn': 'async let value:int=1',
            'missing-name': 'async fn',
            'missing-operand': 'async fn bad()->int{return await}',
            'wrong-return': 'async fn bad()->int{return await true}',
            'wrong-binding': 'fn main()->int{let value:int=await "wrong" return value}',
            'wrong-argument': 'async fn value(n:int)->int{return await n} fn main()->int{return (value true)}',
            'failed-shadow': 'async fn bad()->int{return await 1} shadow bad{assert false} fn main()->int{return 0}',
        }
        output = ROOT/'.test_output'/self.artifact_directory
        output.mkdir(parents=True, exist_ok=True)
        for label, text in cases.items():
            with self.subTest(case=label):
                work = Path(tempfile.mkdtemp(prefix=label+'-', dir=output))
                path, module = work/'main.nano', work/'program.nvm'
                path.write_text(text+'\n')
                previous = b'I retain my previous async program.\n'
                module.write_bytes(previous)
                before = {str(p): hashlib.sha256(p.read_bytes()).hexdigest()
                          for p in [path, self.compiler]}
                try:
                    result = self.command(work, 'refuse', [self.compiler, path, '--emit-nvm', '-o', module],
                                          build=True, accepted=False)
                    self.assertNotEqual(result.returncode, 0)
                    self.assertRegex(result.stdout+result.stderr, r'(?i)(parse|async|type|expected|error|shadow|assert)')
                    self.assertEqual(module.read_bytes(), previous)
                finally:
                    after = {str(p): hashlib.sha256(p.read_bytes()).hexdigest()
                             for p in [path, self.compiler]}
                    (work/'inputs.json').write_text(json.dumps({'scope': self.scope,
                        'before': before, 'after': after,
                        'previous_output_preserved': module.read_bytes() == previous}, indent=2)+'\n')
                    self.assertEqual(before, after)


class CSeedAsyncContracts(AsyncContracts, unittest.TestCase):
    compiler = ROOT/'bin/nano_virt'
    artifact_directory = 'async-cseed-contracts'
    scope = 'I execute actual C-created VM and O0/O2 sanitized native scalar async products.'


class SourceAsyncContracts(AsyncContracts, unittest.TestCase):
    artifact_directory = 'async-source-contracts'
    scope = 'I execute actual source-created VM and O0/O2 sanitized native scalar async products; fresh stages and scheduling remain separate.'

    @classmethod
    def setUpClass(cls):
        source.BstringSourceVmContracts.setUpClass()
        cls.compiler = source.BstringSourceVmContracts.compiler
