"""I retain bare print statements and ordinary declared prefix-call identity."""
import hashlib
import json
import os
from pathlib import Path
import shlex
import sys
import tempfile
import unittest

from tests.native_process import native_link_flags, run_captured

ROOT = Path(__file__).resolve().parents[1]
PRODUCERS = os.environ.get('NANO_PRINT_STATEMENT_COMPILERS',
    'nanoc_c,nano_virt,nanoc_stage1,nanoc_stage2').split(',')


class LegacyPrintStatements(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        allowed = {'nanoc_c', 'nano_virt', 'nanoc_stage1', 'nanoc_stage2'}
        if not PRODUCERS or len(set(PRODUCERS)) != len(PRODUCERS) or set(PRODUCERS)-allowed:
            raise AssertionError('I require explicit unique actual source providers.')
        missing = [name for name in PRODUCERS if not (ROOT/'bin'/name).is_file()]
        if missing:
            raise AssertionError('I require actual source providers: '+', '.join(missing))

    def command(self, work, label, argv, *, build=False, accepted=True):
        result = run_captured(argv, cwd=ROOT, text=True, timeout=60, build=build)
        (work/(label+'.stdout')).write_text(result.stdout)
        (work/(label+'.stderr')).write_text(result.stderr)
        with (work/'commands.jsonl').open('a') as log:
            log.write(json.dumps({'label':label, 'argv':list(map(str,argv)),
                'returncode':result.returncode})+'\n')
        if accepted:
            self.assertEqual(result.returncode,0,result.stdout+result.stderr)
        return result

    def exercise(self, source, expected=None, *, refusal=None):
        output=ROOT/'.test_output/legacy-print-statements'
        output.mkdir(parents=True,exist_ok=True)
        for provider in PRODUCERS:
            with self.subTest(provider=provider):
                work=Path(tempfile.mkdtemp(prefix=self._testMethodName+'-'+provider+'-',dir=output))
                path=work/'main.nano'
                path.write_text(source)
                compiler=ROOT/'bin'/provider
                before=hashlib.sha256(compiler.read_bytes()).hexdigest()
                native=work/'native'
                product=native if provider=='nanoc_c' else work/'module.nvm'
                argv=[compiler,path,'-o',product]
                if provider!='nanoc_c':argv.append('--emit-nvm')
                if expected is None:
                    original=b'I retain my prior program.\n'
                    product.write_bytes(original)
                    result=self.command(work,'refusal',argv,build=True,accepted=False)
                    self.assertGreater(result.returncode,0,result.stdout+result.stderr)
                    self.assertIsNotNone(refusal)
                    self.assertRegex(result.stdout+result.stderr,refusal)
                    self.assertEqual(product.read_bytes(),original)
                else:
                    self.command(work,'source-build',argv,build=True)
                    if provider!='nanoc_c':
                        self.command(work,'verify',[ROOT/'bin/nano_vm','--verify-only',product])
                        vm=self.command(work,'vm',[ROOT/'bin/nano_vm',product])
                        self.assertEqual(vm.stdout,expected)
                        generated=work/'native.c'
                        self.command(work,'aot',[ROOT/'bin/nvm2c',product,'-o',generated],build=True)
                        self.command(work,'native-build',[
                            *shlex.split(os.environ.get('NANO_NATIVE_TEST_CC','cc')),
                            '-std=c11','-O1','-Wall','-Wextra','-Werror',
                            '-fsanitize=address,undefined','-fno-sanitize-recover=all',
                            generated,ROOT/'bin/nano_aot_runtime.o','-lm',*native_link_flags(),
                            *(['-Wl,--export-dynamic','-ldl'] if sys.platform.startswith('linux') else []),
                            '-o',native],build=True)
                    result=self.command(work,'native',[native])
                    self.assertEqual(result.stdout,expected)
                after=hashlib.sha256(compiler.read_bytes()).hexdigest()
                (work/'provider.json').write_text(json.dumps({'path':str(compiler),
                    'before':before,'after':after},indent=2)+'\n')
                self.assertEqual(before,after)

    def test_bare_statements_and_lexical_prefix_calls_keep_their_identity(self):
        self.exercise('''fn hundred(value:int)->int{return (+ value 100)}
shadow hundred { assert (== (hundred 7) 107) }
fn two_hundred(value:int)->int{return (+ value 200)}
shadow two_hundred { assert (== (two_hundred 7) 207) }
fn main()->int {
 let print:fn(int)->int=hundred
 let println:fn(int)->int=two_hundred
 print 1
 println 2
 println (print 3)
 println (println 4)
 return 0
}
shadow main { assert (== (main) 0) }
''','12\n103\n204\n')

    def test_print_arguments_run_once_in_source_order(self):
        self.exercise('''let mut count:int=0
fn next()->int{set count (+ count 1) return count}
shadow next {set count 0 assert (== (next) 1) set count 0}
fn main()->int{set count 0 print (next) println (next) assert (== count 2) return 0}
shadow main {assert (== (main) 0)}
''','12\n')

    def test_printed_lexical_capture_remains_visible(self):
        self.exercise('''fn outer(value:int)->fn()->int {
 return fn()->int {println value return value}
}
shadow outer {let call:fn()->int=(outer 3) assert (== (call) 3)}
fn main()->int{let call:fn()->int=(outer 7) assert (== (call) 7) return 0}
shadow main {assert (== (main) 0)}
''','7\n')

    def test_missing_values_unknown_names_and_false_purity_preserve_output(self):
        sources=[
            ('fn print(value:int)->int{return (+ value 100)} shadow print{assert (== (print 7) 107)} '
             'fn println(value:int)->int{return (+ value 200)} shadow println{assert (== (println 7) 207)} '
             'fn main()->int{print 1 println 2 println (print 3) println (println 4) return 0} '
             'shadow main{assert (== (main) 0)}', r"(?i)cannot redefine built-in function"),
            ('fn main()->int{println missing return 0} shadow main{assert true}', r'(?i)(undefined|unknown|unresolved|not defined|cannot find)'),
            ('fn main()->int{print} shadow main{assert true}', r'(?i)(parse|expected|unexpected|syntax)'),
            ('pure fn show()->int{println 7 return 0} shadow show{assert (== (show) 0)} '
             'fn main()->int{return 0} shadow main{assert (== (main) 0)}', r'(?i)(pure|effect summary)'),
        ]
        for source, pattern in sources:
            with self.subTest(source=source):self.exercise(source,refusal=pattern)
