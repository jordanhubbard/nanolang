"""I preserve exact callable union payloads through checking and execution."""
import os
from pathlib import Path
import shlex
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
PRELUDE = 'fn identity(value:T)->T{return value}\nshadow identity {assert (== (identity 1) 1)}\n'


class CallableUnionStorage(unittest.TestCase):
    def command(self, command):
        result = subprocess.run(list(map(str, command)), cwd=ROOT,
                                capture_output=True, text=True, timeout=120)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        return result.stdout

    def execute(self, body):
        with tempfile.TemporaryDirectory(prefix='nano-callable-union-') as tmp:
            work = Path(tmp)
            source = work / 'source.nano'
            source.write_text(PRELUDE + body + '\nshadow main {assert (== (main) 0)}\n')
            if os.environ.get('NANO_UNION_CHECKER'):
                self.command([os.environ['NANO_UNION_CHECKER'], source])
            native = work / 'native'
            self.command([os.environ.get('NANO_CONTEXTUAL_NATIVE_COMPILER', ROOT/'bin/nanoc_c'), source, '-o', native])
            self.command([native])
            producers = ['c']
            if os.environ.get('NANO_UNION_EMITTER'):
                producers += ['whole', 'program', 'raw']
            for producer in producers:
                with self.subTest(producer=producer):
                    module, assembly = work / 'out.nvm', work / 'out.nasm'
                    if producer == 'c':
                        self.command([ROOT/'bin/nano_virt', source, '--emit-nvm', '-o', module])
                    else:
                        assembly.write_text(self.command([os.environ['NANO_UNION_EMITTER'], source, '0', producer]))
                        self.command([ROOT/'bin/nanoisa', 'asm', assembly, '-o', module])
                    self.command([ROOT/'bin/nano_vm', '--verify-only', module])
                    self.command([ROOT/'bin/nano_vm', module])
                    c, binary = work/'out.c', work/'out'
                    self.command([os.environ.get('NANO_UNION_TRANSLATOR', ROOT/'bin/nvm2c'), module, '-o', c])
                    cc = shlex.split(os.environ.get('NANO_NATIVE_TEST_CC', 'cc'))
                    self.command([*cc, '-std=c11', '-Wall', '-Wextra', '-Werror',
                        '-fsanitize=address,undefined', '-fno-sanitize-recover=all',
                        c, ROOT/'bin/nano_aot_runtime.o', '-lm', '-o', binary])
                    self.command([binary])

    def test_ordinary_payload(self):
        self.execute('''union Choice {Some {callback:fn(int)->int}}
fn main()->int {let c:Choice=Choice.Some {callback:identity}
match c {Some(payload)=>{let f:fn(int)->int=payload.callback assert (== (f 7) 7)}} return 0}''')

    def test_generic_payload(self):
        self.execute('''union Choice<T> {Some {value:T}}
fn main()->int {let c:Choice<fn(int)->int> = Choice.Some {value:identity}
match c {Some(payload)=>{let f:fn(int)->int=payload.value assert (== (f 7) 7)}} return 0}''')

    def test_generic_signature_field(self):
        self.execute('''union Choice<T> {Some {callback:fn(T)->T}}
fn main()->int {let c:Choice<int> = Choice.Some {callback:identity}
match c {Some(payload)=>{let f:fn(int)->int=payload.callback assert (== (f 7) 7)}} return 0}''')

    def test_nested_generic_signature(self):
        self.execute('''union Choice<T> {Some {callback:fn(T)->T}}
fn main()->int {let c:Choice<fn(int)->int> = Choice.Some {callback:identity}
match c {Some(payload)=>{
let select:fn(fn(int)->int)->fn(int)->int=payload.callback
let f:fn(int)->int=(select identity) assert (== (f 7) 7)}} return 0}''')

    def test_returned_union_and_multiple_signatures(self):
        self.execute('''union Choice {Number {callback:fn(int)->int}, Flag {callback:fn(bool)->bool}}
fn select(flag:bool)->Choice {if flag {return Choice.Flag {callback:identity}} return Choice.Number {callback:identity}}
shadow select {match (select true) {Flag(p)=>{let f:fn(bool)->bool=p.callback assert (f true)} Number(p)=>{assert false}}}
fn main()->int {match (select false) {Number(p)=>{let f:fn(int)->int=p.callback assert (== (f 7) 7)} Flag(p)=>{assert false}} return 0}''')

    def test_recursive_callable_signature(self):
        self.execute('''union Choice {Call {callback:fn(Choice)->int}, End {}}
fn read(value:Choice)->int {match value {Call(p)=>{return 1} End()=>{return 7}}}
shadow read {assert (== (read Choice.End {}) 7)}
fn main()->int {let c:Choice=Choice.Call {callback:read}
match c {Call(p)=>{let f:fn(Choice)->int=p.callback assert (== (f Choice.End {}) 7)} End()=>{assert false}} return 0}''')

    def test_incompatible_payload_preserves_output(self):
        declarations = [('union Choice {Some {callback:fn(int)->bool}}', 'Choice'),
                        ('union Choice<T> {Some {callback:T}}', 'Choice<fn(int)->bool>')]
        for declaration, annotation in declarations:
            bodies = [
                'fn unused()->void {let c:'+annotation+' = Choice.Some {callback:identity}}',
                'fn unused()->'+annotation+' {return Choice.Some {callback:identity}}',
                'fn unused(value:'+annotation+')->void {let mut c:'+annotation+' = value set c Choice.Some {callback:identity}}',
            ]
            for body in bodies:
                with self.subTest(annotation=annotation, body=body), tempfile.TemporaryDirectory() as tmp:
                    source, output = Path(tmp)/'source.nano', Path(tmp)/'out'
                    source.write_text(PRELUDE+declaration+'\n'+body+'\nshadow unused {assert true}\nfn main()->int{return 0}\nshadow main {assert true}\n')
                    for compiler in ('nanoc_c', 'nano_virt'):
                        output.write_bytes(b'prior')
                        mode = ['--emit-nvm'] if compiler == 'nano_virt' else []
                        command = os.environ.get('NANO_CONTEXTUAL_NATIVE_COMPILER', ROOT/'bin/nanoc_c') if compiler == 'nanoc_c' else ROOT/'bin/nano_virt'
                        result = subprocess.run([command, source, *mode, '-o', output],
                            cwd=ROOT, capture_output=True, text=True, timeout=120)
                        self.assertNotEqual(result.returncode, 0, result.stdout+result.stderr)
                        self.assertEqual(output.read_bytes(), b'prior')
                        self.assertRegex(result.stdout+result.stderr, r'complete declared callback storage signature|matching concrete types')
                    if os.environ.get('NANO_UNION_CHECKER'):
                        result = subprocess.run([os.environ['NANO_UNION_CHECKER'], source],
                            cwd=ROOT, capture_output=True, text=True, timeout=120)
                        self.assertEqual(result.returncode, 1, result.stdout+result.stderr)

    def test_unannotated_nested_and_global_checker_contexts(self):
        if not os.environ.get('NANO_UNION_CHECKER'):
            self.skipTest('I require the fresh self-hosted checker component.')
        contexts = [
            'fn unused()->void {let c=Choice.Some {callback:identity}}\nshadow unused {assert true}',
            'let c:Choice=Choice.Some {callback:identity}',
            'struct Holder {value:Choice}\nfn unused()->void {let h:Holder=Holder {value:Choice.Some {callback:identity}}}\nshadow unused {assert true}',
        ]
        for body in contexts:
            for result_type, status in [('int', 0), ('bool', 1)]:
                with self.subTest(body=body, result=result_type), tempfile.TemporaryDirectory() as tmp:
                    source = Path(tmp)/'source.nano'
                    source.write_text(PRELUDE+'union Choice {Some {callback:fn(int)->'+result_type+'}}\n'+body+'\nfn main()->int{return 0}\nshadow main {assert true}\n')
                    result = subprocess.run([os.environ['NANO_UNION_CHECKER'], source],
                        cwd=ROOT, capture_output=True, text=True, timeout=120)
                    self.assertEqual(result.returncode, status, result.stdout+result.stderr)
