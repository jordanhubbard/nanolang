"""I retain contextual generic callbacks inside declared aggregate storage."""
import subprocess
import tempfile
import unittest
from pathlib import Path
from tests import test_contextual_generic_functions as context
from tests import test_cseed_generic_functions as shared


class GenericCallbackStorage(unittest.TestCase):
    def execute(self, body):
        context.ContextualGenericFunctions().execute(context.PRELUDE+body+context.SHADOW)

    def test_record_field(self):
        self.execute('struct Holder {callback:fn(int)->int}\n'
            'fn main()->int {let holder=Holder {callback:identity} '
            'let callback:fn(int)->int=holder.callback assert (== (callback 7) 7) return 0}')

    def test_array_element(self):
        self.execute('fn main()->int {let callbacks:array<fn(int)->int> = [identity] '
            'let callback:fn(int)->int=(at callbacks 0) assert (== (callback 7) 7) return 0}')

    def test_tuple_element(self):
        self.execute('fn main()->int {let pair:(fn(int)->int,bool)=(identity,true) '
            'let callback:fn(int)->int=pair.0 assert (== (callback 7) 7) return 0}')

    def test_nested_tuple_and_array(self):
        self.execute('fn main()->int {let pair:((fn(int)->int,bool),array<fn(bool)->bool>)=((identity,true),[identity]) '
            'let number:fn(int)->int=(pair.0).0 let flag:fn(bool)->bool=(at pair.1 0) '
            'assert (== (number 7) 7) assert (flag true) return 0}')

    def test_aggregate_parameter_and_return(self):
        self.execute('fn consume(pair:(fn(int)->int,bool))->int {'
            'let callback:fn(int)->int=pair.0 return (callback 7)}\n'
            'shadow consume {assert (== (consume (identity,true)) 7)}\n'
            'fn produce()->(fn(bool)->bool,int){return (identity,9)}\n'
            'shadow produce {let pair=(produce) assert (== pair.1 9)}\n'
            'fn main()->int{let pair=(produce) let callback:fn(bool)->bool=pair.0 '
            'assert (callback true) assert (== (consume (identity,true)) 7) return 0}')

    def test_reassignment_and_global_storage(self):
        self.execute('struct Holder {callback:fn(int)->int}\n'
            'let global:Holder=Holder {callback:identity}\n'
            'fn main()->int{let mut pair:(fn(int)->int,bool)=(identity,false) '
            'set pair (identity,true) let callback:fn(int)->int=pair.0 '
            'let saved:fn(int)->int=global.callback assert (== (callback 7) 7) '
            'assert (== (saved 9) 9) assert pair.1 return 0}')

    def test_incompatible_storage_preserves_output(self):
        for declaration,unused in [(declaration,unused) for unused in (False,True) for declaration in ['let value:array<fn(int)->bool> = [identity]',
                            'let value:(fn(int)->bool,int)=(identity,7)',
                            'let value=Holder {callback:identity}']]:
            with self.subTest(declaration=declaration,unused=unused),tempfile.TemporaryDirectory() as tmp:
                path,output=Path(tmp)/'main.nano',Path(tmp)/'prior'
                path.write_text(context.PRELUDE+'struct Holder {callback:fn(int)->bool}\n'
                    +('fn unused()->int{'+declaration+' return 0}\nshadow unused {assert true}\nfn main()->int{return 0}' if unused else 'fn main()->int{'+declaration+' return 0}')+context.SHADOW)
                output.write_bytes(b'prior artifact')
                mode=[] if context.NATIVE else ['--emit-nvm']
                result=subprocess.run([context.NATIVE or str(shared.COMPILER),str(path),*mode,'-o',str(output)],
                    cwd=shared.ROOT,capture_output=True,text=True,timeout=120)
                self.assertNotEqual(result.returncode,0,result.stdout+result.stderr)
                self.assertEqual(output.read_bytes(),b'prior artifact')
