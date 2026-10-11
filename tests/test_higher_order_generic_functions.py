"""I infer concrete higher-order calls independently of argument order."""
import subprocess
import tempfile
import unittest
from pathlib import Path
from tests import test_contextual_generic_functions as context
from tests import test_cseed_generic_functions as shared


class HigherOrderGenericFunctions(unittest.TestCase):
    def execute(self, body, files=None):
        context.ContextualGenericFunctions().execute(context.PRELUDE+body+context.SHADOW, files=files)

    def test_argument_order(self):
        for params,args in [('value:T,callback:fn(T)->T','7 identity'),
                            ('callback:fn(T)->T,value:T','identity 7')]:
            with self.subTest(params=params):
                self.execute(f'''fn apply({params})->T{{return (callback value)}}
shadow apply {{assert (== (apply {args}) 7)}}
fn main()->int{{assert (== (apply {args}) 7) return 0}}''')

    def test_callback_result_inference(self):
        self.execute('''fn apply(value:T,callback:fn(T)->E)->E{return (callback value)}
shadow apply {assert (== (apply 1 identity) 1)}
fn main()->int {
 assert (== (apply 7 identity) 7)
 assert (== (apply "kept" identity) "kept")
 return 0
}''')

    def test_structural_callback_result(self):
        self.execute('''struct Box {value:int}
fn wrap(value:T)->array<T>{return [value]}
shadow wrap {assert (== (at (wrap 1) 0) 1)}
fn apply(value:T,callback:fn(T)->E)->E{return (callback value)}
shadow apply {assert (== (at (apply 1 wrap) 0) 1)}
fn main()->int {
 let values = (apply 7 wrap)
 assert (== (at values 0) 7)
 assert (== (at (apply "kept" wrap) 0) "kept")
 assert (at (apply true wrap) 0)
 assert (== (at (at (apply [9] wrap) 0) 0) 9)
 let record:Box = (at (apply Box {value:11} wrap) 0)
 assert (== record.value 11)
 return 0
}''')

    def test_dependent_callbacks_reach_fixed_point(self):
        self.execute('''fn chain(second:fn(E)->R,first:fn(T)->E,value:T)->R{return (second (first value))}
shadow chain {assert (== (chain identity identity 1) 1)}
fn main()->int {
 assert (== (chain identity identity 7) 7)
 assert (chain identity identity true)
 return 0
}''')

    def test_imported_callback_alias(self):
        self.execute('''from "callback.nano" import keep as retained
fn apply(callback:fn(T)->E,value:T)->E{return (callback value)}
shadow apply {assert (== (apply identity 1) 1)}
fn main()->int{assert (== (apply retained 7) 7) return 0}''', files={'callback.nano':
            'pub fn keep(value:U)->U{return value}\nshadow keep {assert (== (keep 1) 1)}\n'})

    def test_nested_array_and_record_binding(self):
        self.execute('''struct Box {value:int}
fn keep(values:array<U>)->array<U>{return values}
shadow keep {assert (== (at (keep [1]) 0) 1)}
fn apply(callback:fn(T)->E,value:T)->E{return (callback value)}
shadow apply {assert (== (apply identity 1) 1)}
fn main()->int {
 let values = (apply keep [[Box {value:7}]])
 let record:Box = (at (at values 0) 0)
 assert (== record.value 7)
 return 0
}''')

    def test_refusals_preserve_output(self):
        sources=[
            'fn apply(value:T,f:fn(T)->bool)->bool{return (f value)}\nshadow apply {assert true}\nfn main()->int{let x=(apply 7 identity) return 0}',
            'fn apply(f:fn(T)->E)->int{return 0}\nshadow apply {assert true}\nfn main()->int{return (apply identity)}',
            'fn apply(a:T,b:T,f:fn(T)->T)->T{return (f a)}\nshadow apply {assert true}\nfn main()->int{let x=(apply 7 true identity) return 0}',
        ]
        for body in sources:
            with self.subTest(body=body),tempfile.TemporaryDirectory() as tmp:
                source,output=Path(tmp)/'main.nano',Path(tmp)/'output'
                source.write_text(context.PRELUDE+body+'\nshadow main {assert true}\n')
                output.write_bytes(b'prior')
                mode=[] if context.NATIVE else ['--emit-nvm']
                result=subprocess.run([context.NATIVE or str(shared.COMPILER),str(source),*mode,'-o',str(output)],
                    cwd=shared.ROOT,capture_output=True,text=True,timeout=120)
                self.assertGreater(result.returncode,0,result.stdout+result.stderr)
                self.assertEqual(output.read_bytes(),b'prior')
