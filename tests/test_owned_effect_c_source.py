"""I publish checked resource effects through the C NanoISA producer."""
import os
from pathlib import Path
import subprocess
import tempfile
import unittest
from tests import test_resource_effect_frontend as frontend

ROOT = Path(__file__).resolve().parents[1]


class OwnedEffectCSource(unittest.TestCase):
    def check(self, declaration, body, accepted, *, files=None, diagnostic="ownership"):
        with tempfile.TemporaryDirectory(prefix='nano-owned-effect-source-') as directory:
            work=Path(directory);source=work/'main.nano';module=work/'main.nvm'
            for name,content in (files or {}).items():(work/name).write_text(content)
            source.write_text(frontend.HEADER+declaration+'\nfn main()->int{'+body+'}\nshadow main{assert true}\n')
            module.write_bytes(b'prior artifact')
            result=subprocess.run([os.environ.get('NANO_OWNED_EFFECT_DRIVER',str(ROOT/'bin/nano_virt')),source,'--emit-nvm','-o',module],cwd=ROOT,
                                  capture_output=True,text=True,timeout=120)
            self.assertEqual(result.returncode,0 if accepted else 1,result.stdout+result.stderr)
            if not accepted:
                messages=result.stdout+result.stderr
                alternatives=(diagnostic,) if isinstance(diagnostic,str) else diagnostic
                self.assertTrue(any(text in messages for text in alternatives),messages)
                self.assertEqual(module.read_bytes(),b'prior artifact');return
            for command,expected in (([ROOT/'bin/nano_vm','--verify-only',module],0),
                                     ([ROOT/'bin/nano_vm',module],7)):
                ran=subprocess.run(command,cwd=ROOT,capture_output=True,text=True,timeout=30)
                self.assertEqual(ran.returncode,expected,ran.stdout+ran.stderr)
            native_source=work/'main.c';native=work/'main'
            emitted=subprocess.run([ROOT/'bin/nvm2c',module,'-o',native_source],cwd=ROOT,
                                   capture_output=True,text=True,timeout=60)
            self.assertEqual(emitted.returncode,0,emitted.stdout+emitted.stderr)
            compiler=os.environ.get('NANO_OWNED_EFFECT_CC','/opt/homebrew/opt/llvm/bin/clang'
                if Path('/opt/homebrew/opt/llvm/bin/clang').exists() else 'cc')
            compiled=subprocess.run([compiler,'-std=c11','-Wall','-Wextra','-Werror','-O1',
                '-fsanitize=address,undefined','-fno-omit-frame-pointer',native_source,'-o',native],
                capture_output=True,text=True,timeout=60)
            self.assertEqual(compiled.returncode,0,compiled.stdout+compiled.stderr)
            ran=subprocess.run([native],capture_output=True,text=True,timeout=30,
                env={**os.environ,'ASAN_OPTIONS':'halt_on_error=1','UBSAN_OPTIONS':'halt_on_error=1'})
            self.assertEqual(ran.returncode,7,ran.stdout+ran.stderr)

    def test_source_resource_effects(self):
        for name in ('test_transfer','test_consumed_handler_argument','test_forwarded_resource',
                     'test_lexical_exit_consumes_argument','test_callee_contains_its_effect',
                     'test_forward_declared_callee_contains_its_effect','test_invalid_ownership',
                     'test_nested_perform_preserves_staged_owners','test_cannot_reinitialize_captured_owner',
                     'test_effectful_call_preserves_staged_owner','test_effectful_call_preserves_live_owner',
                     'test_transitive_effectful_call','test_partial_handler_does_not_hide_other_operation'):
            with self.subTest(case=name):getattr(frontend.ResourceEffectFrontend,name)(self)

    def test_handler_execution(self):
        cases = {
            'ordered_arguments': ('effect Transfer{pass:Handle int->Handle}',
                'let k:Handle=handle {perform Transfer.pass(Handle{value:4} 3)} with {pass value number->{let n:int=(consume value) Handle{value:(+ n number)}}} return (consume k)'),
            'nested_transfer': ('effect Transfer{pass:Handle->Handle}',
                'let k:Handle=handle {perform Transfer.pass(Handle{value:7})} with {pass value->{let next:Handle=handle {perform Transfer.pass(value)} with {pass inner->{inner}} next}} return (consume k)'),
            'capture_mutation': ('effect Transfer{pass:Handle->Handle}',
                'let mut number:int=0 let k:Handle=handle {perform Transfer.pass(Handle{value:7})} with {pass value->{set number 7 value}} assert (== number 7) return (consume k)'),
            'parameter_shadow': ('effect Transfer{pass:Handle->Handle}',
                'let value:Handle=Handle{value:7} let k:Handle=handle {perform Transfer.pass(value)} with {pass value->{value}} return (consume k)'),
            'arm_prefix_destructure': ('effect Transfer{take:Handle->int}',
                'return handle {perform Transfer.take(Handle{value:7})} with {take h->{let Handle{value}=h value}}'),
            'destructure_before_perform': ('effect Transfer{tick:int->int}',
                'let h:Handle=Handle{value:7} let Handle{value}=h return handle {perform Transfer.tick(value)} with {tick n->{n}}'),
            'void_resume': ('effect Transfer{tick:void->void}',
                'handle {perform Transfer.tick()} with {tick ->{}} return 7'),
            'void_perform_return': ('effect Transfer{emit:int->void} fn send(h:Handle)->void{let n:int=(consume h) return perform Transfer.emit(n)} shadow send{assert true}',
                'handle {(send Handle{value:7})} with {emit n->{assert (== n 7)}} return 7'),
            'void_handle_return': ('effect Transfer{emit:int->void} fn send(h:Handle)->void{let n:int=(consume h) return handle {perform Transfer.emit(n)} with {emit n->{assert (== n 7)}}} shadow send{assert true}',
                '(send Handle{value:7}) return 7'),
            'float_resume': ('effect Transfer{tick:float->float}',
                'let n:float=handle {perform Transfer.tick(7.5)} with {tick n->{n}} assert (== n 7.5) return 7'),
            'string_resume': ('effect Transfer{tick:string->string}',
                'let text:string=handle {perform Transfer.tick("seven")} with {tick text->{text}} assert (== text "seven") return 7'),
            'body_return': ('effect Transfer{tick:int->int}',
                'handle {return 7} with {tick n->{n}} return 0'),
            'body_break': ('effect Transfer{tick:int->int}',
                'let h:Handle=Handle{value:7} while true {handle {break} with {tick n->{n}}} return (consume h)'),
            'owned_union': ('union Choice{Some{item:Handle},None{}} effect Transfer{pass:Choice->Choice} fn take(c:Choice)->int{match c{Some(value)=>{let Choice.Some{item}=value return (consume item)} None(value)=>{return 0}}} shadow take{assert true}',
                'let result:Choice=handle {perform Transfer.pass(Choice.Some{item:Handle{value:7}})} with {pass value->{value}} return (take result)'),
        }
        for name,(declaration,body) in cases.items():
            with self.subTest(case=name):self.check(declaration,body,True)

    def test_global_effects(self):
        cases = {
            'handler_write': ('let mut total:int=0 effect E{tick:int->int}',
                'let n:int=handle {perform E.tick(7)} with {tick value->{set total value value}} assert (== n total) return total'),
            'helper_write': ('let mut total:int=0 effect E{tick:int->int} fn send()->int{return perform E.tick(7)} shadow send{assert true}',
                'let n:int=handle {(send)} with {tick value->{set total value value}} assert (== n total) return total'),
            'lexical_return': ('let mut total:int=0 effect E{tick:int->int}',
                'handle {perform E.tick(3)} with {tick value->{set total (+ value 4) return total}} return 0'),
            'initializer': ('effect E{tick:int->int} let total:int=handle {perform E.tick(7)} with {tick value->{value}}',
                'return total'),
            'string_write': ('let mut text:string="before" effect E{tick:string->string}',
                'let got:string=handle {perform E.tick("after")} with {tick value->{set text value value}} assert (== got text) assert (== text "after") return 7'),
            'owned_argument': ('let mut total:int=0 effect Transfer{pass:Handle->Handle}',
                'let k:Handle=handle {perform Transfer.pass(Handle{value:7})} with {pass h->{set total 7 h}} assert (== total 7) return (consume k)'),
            'nested_handlers': ('let mut total:int=0 effect E{tick:int->int}',
                'let n:int=handle {handle {perform E.tick(7)} with {tick value->{set total value value}}} with {tick value->{set total 99 value}} assert (== n total) return total'),
            'union_write': ('union Flag{Yes{value:int},No{}} let mut flag:Flag=Flag.No{} effect E{tick:int->int}',
                'let n:int=handle {perform E.tick(7)} with {tick value->{set flag Flag.Yes{value:value} value}} match flag {Yes(v)=>{return v.value} No(v)=>{return 0}}'),
            'scalar_kinds': ('let mut enabled:bool=false let mut ratio:float=1.25 effect E{tick:int->int}',
                'let n:int=handle {perform E.tick(7)} with {tick value->{set enabled true set ratio 2.0 value}} assert enabled assert (== ratio 2.0) return n'),
            'initializer_order': ('effect E{tick:int->int} let first:int=handle {perform E.tick(7)} with {tick value->{value}} let second:int=handle {perform E.tick(first)} with {tick value->{value}}',
                'return (- (+ first second) 7)'),
        }
        for name,(declaration,body) in cases.items():
            with self.subTest(case=name):self.check(declaration,body,True)

    def test_imported_resource_effects(self):
        files={name+'.nano': 'pub resource struct Handle{value:int} pub effect E{'+name+'_pass:Handle->Handle} fn unwrap(h:Handle)->int{let Handle{value}=h return value} shadow unwrap{assert (== (unwrap Handle{value:7}) 7)} pub fn take(h:Handle)->int{return (unwrap h)} shadow take{assert (== (take Handle{value:7}) 7)}' for name in ('left','right')}
        self.check('module "left.nano" as left\nmodule "right.nano" as right',
            'let first:left.Handle=handle {perform left.E.left_pass(left.Handle{value:3})} with {left_pass value->{value}} let a:int=(left.take first) '
            'let second:right.Handle=handle {perform right.E.right_pass(right.Handle{value:4})} with {right_pass value->{value}} return (+ a (right.take second))',True,files=files)

    def test_ambiguous_imported_operations(self):
        files={name+'.nano': 'pub resource struct Handle{value:int} pub effect E{pass:Handle->Handle}' for name in ('left','right')}
        self.check('module "left.nano" as left\nmodule "right.nano" as right',
            'let h:left.Handle=handle {perform left.E.pass(left.Handle{value:7})} with {pass value->{value}} let left.Handle{value}=h return value',
            False,files=files,diagnostic=('cannot infer a unique effect','I require handler operations to identify one declared effect.'))
