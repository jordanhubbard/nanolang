"""I preserve checked higher-order array aliases across producers and runtimes."""
import os
from pathlib import Path
import subprocess
import tempfile
import unittest
from tests import test_selfhost_effects as effects

ROOT = Path(__file__).resolve().parents[1]
DRIVERS = os.environ.get('NANOLANG_ARRAY_ALIAS_COMPILERS', str(ROOT/'bin/nano_virt')).split(os.pathsep)


class ArrayBuiltinAliases(unittest.TestCase):
    compile = effects.SelfhostEffects.compile
    execute_module = effects.SelfhostEffects.execute_module

    def check(self, helpers, body, accepted=True, files=None):
        for driver in DRIVERS:
            with self.subTest(driver=driver), tempfile.TemporaryDirectory() as directory:
                source = Path(directory)/'main.nano'
                output = Path(directory)/'main.nvm'
                for name, contents in (files or {}).items():
                    (Path(directory)/name).write_text(contents)
                source.write_text(helpers+'\nfn main()->int{'+body+'}\nshadow main{assert (== (main) 7)}\n')
                output.write_bytes(b'prior artifact')
                source_native = os.environ.get('NANOLANG_ARRAY_ALIAS_ROUTE') == 'source-native'
                if source_native:
                    result = subprocess.run([driver, str(source), '-o', str(output)], cwd=ROOT,
                                            capture_output=True, text=True, timeout=90)
                else:
                    result = self.compile(driver, source, output)
                if accepted:
                    self.assertEqual(result.returncode, 0, result.stdout+result.stderr)
                    if source_native:
                        ran = subprocess.run([output], capture_output=True, text=True, timeout=10)
                        self.assertEqual(ran.returncode, 7, ran.stdout+ran.stderr)
                    else:
                        self.execute_module(output, 7)
                else:
                    self.assertNotEqual(result.returncode, 0, result.stdout+result.stderr)
                    self.assertEqual(output.read_bytes(), b'prior artifact')

    def test_pipeline(self):
        self.check('fn plus(n:int)->int{return (+ n 1)} shadow plus{assert true} fn keep(n:int)->bool{return (> n 2)} shadow keep{assert true} fn add(a:int,b:int)->int{return (+ a b)} shadow add{assert true}', 'let values:array<int> = (array_map [1,2] plus) let selected:array<int> = (array_filter values keep) return (array_fold selected 4 add)')

    def test_record_result_inference(self):
        self.check('struct Point{x:int} fn same(p:Point)->Point{return p} shadow same{assert true}', 'let values = (array_map [Point{x:7}] same) let point = (at values 0) return point.x')

    def test_string_accumulator(self):
        self.check('fn append(text:string,n:int)->string{return (str_concat text "x")} shadow append{assert true}', 'let text:string = (array_fold [1,2,3] "abcd" append) return (str_length text)')

    def test_callable_shadowing(self):
        for name in ('array_map','array_filter','array_fold'):
            with self.subTest(name=name):
                self.check('fn plus(n:int)->int{return (+ n 1)} shadow plus{assert true}', 'let '+name+':fn(int)->int=plus return ('+name+' 6)')

    def test_matching_arity_shadow(self):
        self.check('fn plus(n:int)->int{return (+ n 1)} shadow plus{assert true} fn unchanged(values:array<int>,callback:fn(int)->int)->array<int>{return values} shadow unchanged{assert true}', 'let array_map:fn(array<int>,fn(int)->int)->array<int> = unchanged let values:array<int> = (array_map [7] plus) return (at values 0)')

    def test_signature_refusals(self):
        for helpers, body in (
            ('fn bad(s:string)->int{return 7} shadow bad{assert true}', 'let values:array<int> = (array_map [1] bad) return 7'),
            ('fn bad(n:int)->int{return n} shadow bad{assert true}', 'let values:array<int> = (array_filter [1] bad) return 7'),
            ('fn bad(a:float,b:float)->float{return (+ a b)} shadow bad{assert true}', 'return (array_fold [1] 0 bad)'),
        ):
            with self.subTest(body=body):
                self.check(helpers, body, False)
