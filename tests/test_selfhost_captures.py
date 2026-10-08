"""I execute lexical captures through source producers and owned native closures."""
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
COMPILER = Path(os.environ.get('NANOLANG_SELFHOST_COMPILER', ROOT / 'bin/nanoc_stage2')).resolve()


class SelfhostCaptures(unittest.TestCase):
    def checked(self, args, **kwargs):
        result = subprocess.run(list(map(str, args)), cwd=ROOT, capture_output=True,
                                text=True, timeout=180, **kwargs)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        return result

    def execute(self, source, work, expected, producers=('seed', 'selfhost')):
        for producer in producers:
            with self.subTest(producer=producer):
                module, native, binary = [work / (producer + suffix) for suffix in ('.nvm', '.c', '.exe')]
                compiler = ROOT / 'bin/nano_virt' if producer == 'seed' else COMPILER
                self.checked([compiler, source, '--emit-nvm', '-o', module])
                self.assertEqual(self.checked([ROOT / 'bin/nano_vm', module]).stdout, expected)
                self.checked([ROOT / 'bin/nvm2c', module, '-o', native])
                cc = os.environ.get('NANO_NATIVE_TEST_CC') or shutil.which('cc')
                self.assertTrue(cc)
                self.checked([cc, '-std=c11', '-O0', '-Wall', '-Wextra', '-Werror',
                              '-fsanitize=address,undefined', '-fno-sanitize-recover=all',
                              native, ROOT / 'bin/nano_aot_runtime.o', '-lm',
                              *(['-Wl,--export-dynamic', '-ldl'] if sys.platform.startswith('linux') else []),
                              '-o', binary])
                self.assertEqual(self.checked([binary], env={**os.environ, 'ASAN_OPTIONS': 'detect_leaks=1'}).stdout, expected)

    def test_unchanged_returned_closure_chain(self):
        with tempfile.TemporaryDirectory(prefix='nano-captured-chain-') as tmp:
            self.execute(ROOT / 'docs/evidence/captured-closure-baseline-20261008/capture.nano', Path(tmp), '42\n')

    def test_mutable_instances_containers_and_forwarded_callable(self):
        source = '''struct Box { callback: fn()->int, names: array<string> }
fn counter(start:int)->fn()->int {
    let mut value:int = start
    return fn()->int { set value (+ value 1) return value }
}
shadow counter { let f:fn()->int = (counter 40) assert (== (f) 41) assert (== (f) 42) }
fn relay(f:fn()->int)->fn()->fn()->int {
    return fn()->fn()->int { return fn()->int { return (f) } }
}
shadow relay { let middle:fn()->fn()->int = (relay (counter 40)) let f:fn()->int = (middle) assert (== (f) 41) }
fn named()->int { return 7 }
shadow named { assert (== (named) 7) }
fn managed(values:array<string>, box:Box)->fn()->string {
    return fn()->string { return (str_concat (at values 0) (at box.names 0)) }
}
shadow managed { let f:fn()->string = (managed ["a"] Box {callback: named, names: ["b"]}) assert (== (f) "ab") }
fn churn()->void {
    let mut i:int = 0
    while (< i 3000) { let temporary:string = (str_concat "allocated string" "temporary string") set i (+ i 1) }
}
shadow churn { (churn) }
fn main()->int {
    let one:fn()->int = (counter 0)
    let alias:fn()->int = one
    let two:fn()->int = (counter 0)
    assert (== (one) 1)
    assert (== (alias) 2)
    assert (== (two) 1)
    let values:array<fn()->int> = [one, named]
    let copies:array<fn()->int> = values
    (array_set copies 1 two)
    let projected:fn()->int = (at values 1)
    assert (== (projected) 2)
    let middle:fn()->fn()->int = (relay alias)
    let inner:fn()->int = (middle)
    let immediate:fn()->fn()->int = (relay fn()->int { return 99 })
    let immediate_inner:fn()->int = (immediate)
    assert (== (immediate_inner) 99)
    let box:Box = Box {callback: inner, names: [" retained"]}
    let text:fn()->string = (managed ["still"] box)
    let mut i:int = 0
    while (< i 40) { (array_push values (counter i)) set i (+ i 1) }
    (churn)
    let callback:fn()->int = box.callback
    assert (== (callback) 3)
    assert (== (text) "still retained")
    let last:fn()->int = (at values 41)
    assert (== (last) 40)
    (println "captures ok")
    return 0
}
shadow main { assert (== (main) 0) }
'''
        with tempfile.TemporaryDirectory(prefix='nano-captured-values-') as tmp:
            work = Path(tmp)
            path = work / 'input.nano'
            path.write_text(source)
            self.execute(path, work, 'captures ok\n')

    def test_imported_capture_and_shadow_local(self):
        with tempfile.TemporaryDirectory(prefix='nano-captured-import-') as tmp:
            work = Path(tmp)
            (work / 'offset.nano').write_text('''pub fn offset(n:int)->fn(int)->int { return fn(x:int)->int { return (+ n x) } }
shadow offset { let n:int = 40 let check:fn()->int = fn()->int { let f:fn(int)->int = (offset n) return (f 2) } assert (== (check) 42) }
''')
            source = work / 'input.nano'
            source.write_text('''module "offset.nano" as offsets
fn main()->int { let f:fn(int)->int = (offsets.offset 40) (println (f 2)) return 0 }
shadow main { assert (== (main) 0) }
''')
            self.execute(source, work, '42\n')

    def test_immediate_zero_capture_identity_and_scope_restore(self):
        source = '''fn fresh()->fn()->int { return fn()->int { return 7 } }
shadow fresh { let f:fn()->int = (fresh) assert (== (f) 7) }
fn main()->int {
    let first:fn()->int = (fresh)
    let second:fn()->int = (fresh)
    assert (!= first second)
    assert (== first first)
    let mut i:int = 0
    let mut total:int = 0
    while (< i 3) {
        let f:fn()->int = fn()->int { let mut j:int = 0 while (< j i) { set j (+ j 1) } return j }
        set total (+ total (f))
        set i (+ i 1)
    }
    assert (== total 3)
    let n:int = 42
    assert (== (fn()->int { return n }) 42)
    (println "scope ok")
    return 0
}
shadow main { assert (== (main) 0) }
'''
        with tempfile.TemporaryDirectory(prefix='nano-captured-scope-') as tmp:
            work = Path(tmp)
            path = work / 'input.nano'
            path.write_text(source)
            self.execute(path, work, 'scope ok\n')

    def test_invalid_capture_preserves_prior_output(self):
        cases = [
            'fn make()->fn()->int{return fn()->int{return absent}}',
            'fn make(n:int)->fn()->int{return fn()->int{set n 42 return n}}',
            'fn make()->fn()->int{if true {let hidden:int = 42} return fn()->int{return hidden}}',
            'fn make()->int{return (fn(x:int)->int{return x} "wrong")}',
            'fn make()->int{return (fn()->int{return 1} 2)}',
            'fn make()->int{return (fn(f:fn()->int)->int{return (f)} fn()->string{return "wrong"})}',
            'fn take(f:fn()->int)->int{return (f)} fn bad()->fn()->string{return fn()->string{return "wrong"}} fn make()->int{return (take (bad))}',
            'fn take(f:fn()->int)->int{return (f)} fn make()->int{return (take fn()->string{return "wrong"})}',

        ]
        with tempfile.TemporaryDirectory(prefix='nano-captured-refusal-') as tmp:
            work = Path(tmp)
            for case in cases:
                for compiler in (ROOT / 'bin/nano_virt', COMPILER):
                    with self.subTest(source=case, compiler=compiler):
                        source, output = work / 'bad.nano', work / 'prior.nvm'
                        source.write_text(case + '\nfn main()->int{return 0}\nshadow main {assert true}\n')
                        output.write_bytes(b'prior output\n')
                        result = subprocess.run([str(compiler), str(source), '--emit-nvm', '-o', str(output)],
                                                cwd=ROOT, capture_output=True, timeout=120)
                        self.assertNotEqual(result.returncode, 0, result.stdout + result.stderr)
                        self.assertEqual(output.read_bytes(), b'prior output\n')


if __name__ == '__main__':
    unittest.main()
