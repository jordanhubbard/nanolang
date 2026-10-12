"""I check resource effect flow before backend ownership admission."""
import os
from pathlib import Path
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
HEADER = """resource struct Handle{value:int}
fn consume(h:Handle)->int{let Handle{value}=h return value}
shadow consume{assert (== (consume Handle{value:7}) 7)}
"""


class ResourceEffectFrontend(unittest.TestCase):
    def check(self, declaration, body, accepted):
        routes = os.environ.get('NANOLANG_RESOURCE_EFFECT_ROUTES', 'frontend').split(',')
        for route in routes:
            self.assertIn(route, ('frontend', 'native'))
            with self.subTest(route=route), tempfile.TemporaryDirectory() as directory:
                source = Path(directory)/'main.nano'
                output = Path(directory)/'main'
                source.write_text(HEADER+declaration+'\nfn main()->int{'+body+'}\nshadow main{assert true}\n')
                output.write_bytes(b'prior artifact')
                command = [ROOT/'obj/test_affine_c_frontend', source] if route == 'frontend' else [ROOT/'bin/nanoc_c', source, '-o', output]
                result = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, timeout=90)
                self.assertEqual(result.returncode, 0 if accepted else 1, result.stdout+result.stderr)
                if not accepted:
                    self.assertIn('ownership', result.stderr)
                    self.assertEqual(output.read_bytes(), b'prior artifact')
                elif route == 'native':
                    ran = subprocess.run([output], capture_output=True, text=True, timeout=10)
                    self.assertEqual(ran.returncode, 7, ran.stdout+ran.stderr)

    def test_transfer(self):
        self.check('effect Transfer{pass:Handle->Handle}', 'let h:Handle=Handle{value:7} let k:Handle=handle {perform Transfer.pass(h)} with {pass value->{value}} return (consume k)', True)

    def test_consumed_handler_argument(self):
        self.check('effect Transfer{take:Handle->int}', 'let h:Handle=Handle{value:7} return handle {perform Transfer.take(h)} with {take value->{(consume value)}}', True)

    def test_forwarded_resource(self):
        self.check('effect Transfer{pass:Handle->Handle} fn send(h:Handle)->Handle{return perform Transfer.pass(h)} shadow send{assert true}', 'let h:Handle=Handle{value:7} let k=handle {(send h)} with {pass value->{value}} return (consume k)', True)

    def test_lexical_exit_consumes_argument(self):
        self.check('effect Transfer{pass:Handle->Handle}', 'let h:Handle=Handle{value:7} let k:Handle=handle {perform Transfer.pass(h)} with {pass value->{return (consume value)}} return (consume k)', True)

    def test_invalid_ownership(self):
        declaration = 'effect Transfer{pass:Handle->Handle}'
        cases = {
            'reuse_input': 'let h:Handle=Handle{value:7} let k:Handle=handle {perform Transfer.pass(h)} with {pass value->{value}} let a:int=(consume h) return (consume k)',
            'drop_result': 'let h:Handle=Handle{value:7} let k:Handle=handle {perform Transfer.pass(h)} with {pass value->{value}} return 7',
            'drop_parameter': 'let h:Handle=Handle{value:7} let k:Handle=handle {perform Transfer.pass(h)} with {pass value->{Handle{value:7}}} return (consume k)',
            'reuse_parameter': 'let h:Handle=Handle{value:7} let k:Handle=handle {perform Transfer.pass(h)} with {pass value->{let n:int=(consume value) value}} return (consume k)',
            'capture_owner': 'let h:Handle=Handle{value:7} let k:Handle=handle {perform Transfer.pass(Handle{value:1})} with {pass value->{let n:int=(consume value) h}} return (consume k)',
            'unrelated_live_owner': 'let h:Handle=Handle{value:7} let other:Handle=Handle{value:1} let k:Handle=handle {perform Transfer.pass(h)} with {pass value->{value}} let n:int=(consume other) return (consume k)',
            'discard_resume': 'let h:Handle=Handle{value:7} let ignored=handle {perform Transfer.pass(h) 7} with {pass value->{value}} return ignored',
        }
        for name, body in cases.items():
            with self.subTest(case=name):
                self.check(declaration, body, False)

    def test_nested_perform_preserves_staged_owners(self):
        self.check('effect Transfer{pass:Handle int->Handle,tick:int->int}', 'let h:Handle=Handle{value:7} let k:Handle=handle {perform Transfer.pass(h perform Transfer.tick(0))} with {pass value number->{value} tick number->{return 7}} return (consume k)', False)

    def test_cannot_reinitialize_captured_owner(self):
        self.check('effect Transfer{pass:Handle->Handle}', 'let mut h:Handle=Handle{value:7} let n:int=(consume h) let k:Handle=handle {perform Transfer.pass(Handle{value:1})} with {pass value->{set h Handle{value:2} value}} return (consume k)', False)

    def test_effectful_call_preserves_staged_owner(self):
        self.check('effect Transfer{pass:Handle int->Handle,tick:int->int} fn tick()->int{return perform Transfer.tick(0)} shadow tick{assert true}', 'let h:Handle=Handle{value:7} let k:Handle=handle {perform Transfer.pass(h (tick))} with {pass value number->{value} tick number->{return 7}} return (consume k)', False)

    def test_effectful_call_preserves_live_owner(self):
        self.check('effect Transfer{tick:int->int} fn tick()->int{return perform Transfer.tick(0)} shadow tick{assert true}', 'let h:Handle=Handle{value:7} let n:int=handle {(tick)} with {tick number->{return 7}} return (consume h)', False)

    def test_callee_contains_its_effect(self):
        self.check('effect Transfer{tick:int->int} fn tick()->int{return handle {perform Transfer.tick(7)} with {tick n->{n}}} shadow tick{assert true}', 'let h:Handle=Handle{value:7} let n:int=(tick) return (consume h)', True)

    def test_transitive_effectful_call(self):
        self.check('effect Transfer{tick:int->int} fn tick()->int{return perform Transfer.tick(0)} shadow tick{assert true} fn forward()->int{return (tick)} shadow forward{assert true}', 'let h:Handle=Handle{value:7} let n:int=handle {(forward)} with {tick number->{return 7}} return (consume h)', False)

    def test_partial_handler_does_not_hide_other_operation(self):
        self.check('effect Transfer{tick:int->int,escape:int->int} fn tick()->int{return handle {perform Transfer.escape(0)} with {tick n->{n}}} shadow tick{assert true}', 'let h:Handle=Handle{value:7} let n:int=handle {(tick)} with {escape number->{return 7}} return (consume h)', False)

    def test_forward_declared_callee_contains_its_effect(self):
        self.check('effect Transfer{tick:int->int} fn carrier()->int{let h:Handle=Handle{value:7} let n:int=(tick) return (consume h)} shadow carrier{assert true} fn tick()->int{return handle {perform Transfer.tick(7)} with {tick n->{n}}} shadow tick{assert true}', 'return (carrier)', True)

    def test_mutual_recursion_propagates_effect(self):
        self.check('effect Transfer{tick:int->int} fn first(n:int)->int{if (> n 0){return (second (- n 1))} return 0} shadow first{assert true} fn second(n:int)->int{if (> n 0){return (first (- n 1))} return perform Transfer.tick(0)} shadow second{assert true}', 'let h:Handle=Handle{value:7} let n:int=handle {(first 1)} with {tick number->{return 7}} return (consume h)', False)

    def test_effect_preserves_staged_spread_owner(self):
        self.check('effect Transfer{tick:int->int}', 'let h:Handle=Handle{value:7} let k:Handle=handle {{..h,value:perform Transfer.tick(0)}} with {tick number->{return 7}} return (consume k)', False)

    def test_callback_effect_preserves_live_owner(self):
        self.check('effect Transfer{tick:int->int} fn tick(n:int)->int{return perform Transfer.tick(n)} shadow tick{assert true}', 'let h:Handle=Handle{value:7} let values:array<int> = handle {(array_map [1] tick)} with {tick number->{return 7}} return (consume h)', False)

    def test_effect_free_callback_preserves_live_owner(self):
        self.check('effect Transfer{tick:int->int} fn plus(n:int)->int{return (+ n 1)} shadow plus{assert true}', 'let h:Handle=Handle{value:7} let values:array<int> = (array_map [1] plus) return (consume h)', True)

    def test_callback_shadow_does_not_use_global_summary(self):
        self.check('effect Transfer{tick:int->int} fn plus(n:int)->int{return (+ n 1)} shadow plus{assert true} fn tick(n:int)->int{return perform Transfer.tick(n)} shadow tick{assert true}', 'let h:Handle=Handle{value:7} let plus:fn(int)->int=tick let values:array<int> = handle {(array_map [1] plus)} with {tick number->{return 7}} return (consume h)', False)

    def test_array_alias_pipeline(self):
        self.check('fn plus(n:int)->int{return (+ n 1)} shadow plus{assert true} fn keep(n:int)->bool{return (> n 2)} shadow keep{assert true} fn add(a:int,b:int)->int{return (+ a b)} shadow add{assert true}', 'let values:array<int> = (array_map [1,2] plus) let selected:array<int> = (array_filter values keep) return (array_fold selected 4 add)', True)

    def test_array_alias_callable_shadow(self):
        self.check('fn plus(n:int)->int{return (+ n 1)} shadow plus{assert true} fn unchanged(values:array<int>, callback:fn(int)->int)->array<int>{return values} shadow unchanged{assert true}', 'let array_map:fn(array<int>,fn(int)->int)->array<int> = unchanged let values:array<int> = (array_map [7] plus) return (at values 0)', True)

    def test_declared_alias_effect_preserves_live_owner(self):
        self.check('effect Transfer{tick:int->int} fn array_map(n:int)->int{return perform Transfer.tick(n)} shadow array_map{assert true}', 'let h:Handle=Handle{value:7} let n:int=handle {(array_map 0)} with {tick number->{return 7}} return (consume h)', False)

    def test_declared_alias_contains_its_effect(self):
        self.check('effect Transfer{tick:int->int} fn array_map(n:int)->int{return handle {perform Transfer.tick(n)} with {tick number->{number}}} shadow array_map{assert true}', 'let h:Handle=Handle{value:7} let n:int=(array_map 0) return (consume h)', True)
