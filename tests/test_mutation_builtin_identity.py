"""I keep mutation builtin typing subordinate to exact lexical bindings."""
import os
from pathlib import Path
import subprocess
import tempfile
import unittest

ROOT = Path(os.environ.get('NANO_MUTATION_TEST_ROOT', Path(__file__).resolve().parents[1]))
DRIVER = '''import "src_nano/typecheck.nano"
import "src_nano/parser.nano"
import "src_nano/compiler/lexer.nano"
import "src_nano/compiler/module_bindings.nano"
from "src_nano/compiler/diagnostics.nano" import diag_list_new
extern fn get_argc() -> int
extern fn get_argv(index: int) -> string
extern fn file_read(path: string) -> string
fn main() -> int {
 if (!= (get_argc) 2) { return 0 }
 let path: string = (get_argv 1)
 let diagnostics: List<CompilerDiagnostic> = (diag_list_new)
 let tokens: List<LexerToken> = (tokenize_string (file_read path) path diagnostics)
 let parsed: Parser = (parse_program tokens (list_LexerToken_length tokens) path)
 if (parser_has_error parsed) { (println "PARSE") return 2 }
 (mb_reset [])
 let checked: TypecheckPhaseOutput = (typecheck_phase_with_shadows parsed path [] 0)
 if checked.had_error { (println "TYPECHECK") return 1 }
 (println "CHECKED")
 return 0
}
shadow main { assert (== (main) 0) }
'''

class MutationBuiltinIdentity(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.work = Path(tempfile.mkdtemp(prefix='nanolang-mutation-identity-'))
        print('I retain mutation checker artifacts in ' + str(cls.work), flush=True)
        source = cls.work/'checker.nano'
        source.write_text(DRIVER)
        cls.drivers = []
        for name in ('nanoc_c', 'nanoc_stage1', 'nanoc_stage2'):
            binary = cls.work/('checker-' + name)
            result = subprocess.run([str(ROOT/'bin'/name), str(source), '-o', str(binary)],
                                    cwd=ROOT, text=True, capture_output=True, timeout=900)
            (cls.work/(name+'.setup.log')).write_text(result.stdout+result.stderr)
            if result.returncode:
                raise AssertionError(name + ': ' + result.stdout + result.stderr)
            cls.drivers.append(binary)

    def command(self, args, expected=0):
        result = subprocess.run([str(x) for x in args], cwd=ROOT, text=True,
                                capture_output=True, timeout=180)
        self.assertEqual(result.returncode, expected, result.stdout+result.stderr)
        return result

    def check(self, label, source, expected=0):
        path = self.work/(label+'.nano')
        path.write_text(source)
        for driver in self.drivers:
            with self.subTest(case=label, driver=driver.name):
                result = self.command([driver, path], expected)
                self.assertEqual(result.stdout.strip(), 'CHECKED' if not expected else 'TYPECHECK')

    def test_declared_local_formal_and_initializer(self):
        for name in ('array_push', 'array_set'):
            params = 'a:array<float>,v:float' if name == 'array_push' else 'a:array<float>,i:int,v:float'
            types = 'array<float>,float' if name == 'array_push' else 'array<float>,int,float'
            args = '[1.0] 2.0' if name == 'array_push' else '[1.0] 0 2.0'
            declaration = f'fn {name}({params})->int{{return 7}} shadow {name}{{assert true}}\n'
            self.check(name+'-declared', declaration + f'fn main()->int{{let result:int=({name} {args}) return result}} shadow main{{assert true}}')
            callback = f'fn selected({params})->bool{{return true}} shadow selected{{assert true}}\n'
            self.check(name+'-local', declaration + callback + f'fn main()->int{{let {name}:fn({types})->bool=selected let result:bool=({name} {args}) assert result return 0}} shadow main{{assert true}}')
            self.check(name+'-formal', declaration + f'fn caller({name}:fn({types})->bool)->bool{{return ({name} {args})}} shadow caller{{assert true}} fn main()->int{{return 0}} shadow main{{assert true}}')
            self.check(name+'-initializer', declaration + f'fn main()->int{{let {name}:int=({name} {args}) assert (== {name} 7) return 0}} shadow main{{assert true}}')
            self.check(name+'-wrong-result', declaration + f'fn main()->int{{let result:bool=({name} {args}) return 0}} shadow main{{assert true}}', 1)
            self.check(name+'-wrong-arity', declaration + f'fn main()->int{{let result:int=({name}) return 0}} shadow main{{assert true}}', 1)

    def test_unbound_mutation(self):
        source = self.work/'unbound.nano'
        source.write_text('''fn main()->int {
 let values:array<float> =[1.0]
 let alias:array<float> =(array_push values 2.5)
 (array_set alias 0 3.5)
 assert (== (array_length values) 2)
 assert (== (at values 0) 3.5)
 assert (== (at alias 1) 2.5)
 return 0
}
shadow main { assert (== (main) 0) }
''')
        for name in ('nanoc_c', 'nanoc_stage1', 'nanoc_stage2'):
            with self.subTest(producer=name):
                output=self.work/('unbound-'+name)
                self.command([ROOT/'bin'/name, source, '-o', output])
                self.command([output])

    def test_declared_refusals_preserve_output(self):
        for name in ('array_push', 'array_set'):
            declaration=f'fn {name}(a:int)->int{{return a}} shadow {name}{{assert true}}\n'
            for label, expression in (('arity',f'({name} 1 2)'), ('operand',f'({name} "wrong")')):
                source=self.work/(name+'-'+label+'-refusal.nano')
                source.write_text(declaration+f'fn main()->int{{return {expression}}} shadow main{{assert true}}')
                for producer in ('nanoc_stage1','nanoc_stage2'):
                    for flags in (('--target','c'),('--emit-nvm',)):
                        with self.subTest(name=name,case=label,producer=producer,flags=flags):
                            output=self.work/'prior-output'
                            output.write_bytes(b'prior output')
                            result=self.command([ROOT/'bin'/producer, source,*flags,'-o',output],1)
                            self.assertEqual(output.read_bytes(),b'prior output')
                            self.assertNotRegex(result.stdout+result.stderr,r'(?i)parse error')
                            self.assertRegex(result.stdout+result.stderr,r'(?i)(type|argument|expected)')

    def test_c_reserved_set_policy(self):
        source=self.work/'reserved-set.nano'
        source.write_text('fn array_set(a:int)->int{return a} shadow array_set{assert true} fn main()->int{return 0} shadow main{assert true}')
        output=self.work/'reserved-prior'
        output.write_bytes(b'prior output')
        result=subprocess.run([str(ROOT/'bin/nanoc_c'),str(source),'-o',str(output)],cwd=ROOT,capture_output=True,text=True,timeout=180)
        self.assertNotEqual(result.returncode,0,result.stdout+result.stderr)
        self.assertEqual(output.read_bytes(),b'prior output')
        self.assertRegex(result.stdout+result.stderr,r'(?i)(reserved|built.?in)')

if __name__=='__main__': unittest.main()
