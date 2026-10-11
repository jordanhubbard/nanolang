"""I require effect semantics and refusal parity through both NanoISA frontends."""
import os
from pathlib import Path
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
DRIVERS = os.environ.get('NANOLANG_EFFECT_COMPILERS', os.pathsep.join(
    str(ROOT/'bin'/name) for name in ('nano_virt', 'nanoc_stage1', 'nanoc_stage2'))).split(os.pathsep)


class SelfhostEffects(unittest.TestCase):
    def compile(self, driver, source, output):
        return subprocess.run([driver, str(source), '--emit-nvm', '-o', str(output)],
                              cwd=ROOT, capture_output=True, text=True, timeout=120)

    def test_dynamic_handlers_resume_and_lexical_returns(self):
        sources = [
            ('resume', '''effect Ask{ask:int->int}
fn send(x:int)->int {return (+ 10 perform Ask.ask(x))}
shadow send{assert true}
fn main()->int {return handle {(send 3)} with {ask value->{(+ value 1)}}}''', 14),
            ('lexical_return', '''effect Ask{ask:int->int}
let mut trace:int=0
fn send()->int{let x:int=perform Ask.ask(7) set trace 100 return x}
shadow send{assert true}
fn leave()->int{let x:int=handle {(send)} with {ask n->{return n}} return (+ x 1000)}
shadow leave{assert true}
fn main()->int{set trace 0 let x:int=(leave) assert (== trace 0) return x}''', 7),
            ('nested', '''effect Ask{ask:int->int}
fn main()->int{return handle {perform Ask.ask(4)} with {ask value->{let inner:int=handle {perform Ask.ask(2)} with {ask value->{(+ value 1)}} (+ value inner)}}}''', 7),
            ('ordered_arguments', '''effect Calc{mix:int int->int,tick:void->int}
let mut trace:int=0
fn argument(n:int)->int{set trace (+ (* trace 10) n) return n}
shadow argument{assert true}
fn main()->int{set trace 0 let left:int=handle {perform Calc.mix((argument 1) (argument 2))} with {mix first second->{(+ first second)}} let right:int=handle {perform Calc.tick()} with {tick ()->{4}} assert (== trace 12) return (+ left right)}''', 7),
            ('typed', '''effect Calc{mix:int string bool float->int}
fn main()->int{return handle {perform Calc.mix(3 "abc" true 2.5)} with {mix number text flag amount->{assert flag assert (== amount 2.5) (+ number (str_length text))}}}''', 6),
            ('void', '''effect Record{emit:int->void}
let mut recorded:int=0
fn send()->void{perform Record.emit(7)}
shadow send{assert true}
fn main()->int{set recorded 0 let ignored=handle {(send)} with {emit value->{set recorded value}} return recorded}''', 7),
            ('local_loop', '''effect Ask{ask:int->int}
fn main()->int{return handle {perform Ask.ask(3)} with {ask value->{let mut total:int=0 let mut i:int=0 while (< i value){set total (+ total i) set i (+ i 1)} total}}}''', 3),
        ]
        for driver in DRIVERS:
            for name, body, expected in sources:
                with self.subTest(driver=driver, case=name), tempfile.TemporaryDirectory() as directory:
                    work = Path(directory)
                    source, output = work/'source.nano', work/'source.nvm'
                    source.write_text(body+'\nshadow main{assert (== (main) '+str(expected)+')}\n')
                    built = self.compile(driver, source, output)
                    self.assertEqual(built.returncode, 0, built.stdout+built.stderr)
                    ran = subprocess.run([ROOT/'bin/nano_vm', output], capture_output=True, text=True, timeout=30)
                    self.assertEqual(ran.returncode, expected, ran.stdout+ran.stderr)

    def test_invalid_effects_preserve_previous_output(self):
        sources = [
            'fn main()->int{return perform Missing.ask(1)}',
            'effect Ask{ask:int->int} fn main()->int{return perform Ask.missing(1)}',
            'effect Ask{ask:int->int} fn main()->int{return perform Ask.ask(true)}',
            'effect Ask{ask:int->int} fn main()->int{return perform Ask.ask()}',
            'effect Ask{ask:int->int} fn main()->int{return handle {perform Ask.ask(1)} with {ask n->{true}}}',
            'effect Ask{ask:int->int} fn main()->int{return handle {perform Ask.ask(1)} with {ask n m->{n}}}',
            'effect Ask{ask:int->int} fn main()->int{return handle {perform Ask.ask(1)} with {ask n->{n} ask n->{n}}}',
            'effect A{ask:int->int} effect B{ask:int->int} fn main()->int{return handle {perform A.ask(1)} with {ask n->{n}}}',
            'effect Ask{ask:int->int} fn main()->int{let x:int=handle {perform Ask.ask(1)} with {ask hidden->{hidden}} return hidden}',
            'effect Ask{ask:int->int} fn main()->int{while true{let x:int=handle {perform Ask.ask(1)} with {ask n->{break}} return x} return 0}',
        ]
        for driver in DRIVERS:
            for body in sources:
                with self.subTest(driver=driver, source=body), tempfile.TemporaryDirectory() as directory:
                    work = Path(directory)
                    source, output = work/'source.nano', work/'source.nvm'
                    source.write_text(body+'\nshadow main{assert true}\n')
                    output.write_bytes(b'previous artifact')
                    built = self.compile(driver, source, output)
                    self.assertNotEqual(built.returncode, 0, built.stdout+built.stderr)
                    self.assertEqual(output.read_bytes(), b'previous artifact')


if __name__ == '__main__':
    unittest.main()
