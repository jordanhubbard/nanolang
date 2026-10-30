"""I apply checked byte contexts at ordered ordinary record field boundaries."""
import os
from pathlib import Path
import shlex
import unittest
from tests import test_optional_record_join as harness


def ordered_source(spread=False):
    definitions = '''struct Bytes {first:u8, middle:int, last:u8}
let mut trace:int=0
fn tick(mark:int,value:int)->int {set trace (+ (* trace 10) mark) return value}
shadow tick {set trace 0 assert (== (tick 1 7) 7) assert (== trace 1)}
'''
    if spread:
        definitions += '''fn base()->Bytes {let ignored:int=(tick 1 0) return Bytes{first:7,middle:8,last:9}}
shadow base {set trace 0 assert (== (base).middle 8) assert (== trace 1)}
'''
        expression='{..(base),last:(tick 2 (- 0 1)),first:(tick 3 256)}'
        middle=8
    else:
        expression='Bytes{last:(tick 1 (- 0 1)),middle:(tick 2 17),first:(tick 3 256)}'
        middle=17
    return definitions+f'''fn main()->int {{
 set trace 0
 let value:Bytes={expression}
 assert (== trace 123)
 assert (== (cast_int value.first) 0)
 assert (== value.middle {middle})
 assert (== (cast_int value.last) 255)
 let literal:Bytes=Bytes{{first:0,middle:0,last:255}}
 assert (== (cast_int literal.first) 0)
 assert (== (cast_int literal.last) 255)
 let identity:Bytes=Bytes{{first:value.last,middle:0,last:value.first}}
 assert (== (cast_int identity.first) 255)
 assert (== (cast_int identity.last) 0)
 let limits:Bytes=Bytes{{first:(+ 9223372036854775807 0),middle:0,last:(- (- 0 9223372036854775807) 1)}}
 assert (== (cast_int limits.first) 255)
 assert (== (cast_int limits.last) 0)
 return 0
}}
shadow main {{assert (== (main) 0)}}
'''


class U8FieldContext(unittest.TestCase):
    setUpClass=classmethod(harness.OptionalRecordJoin.setUpClass.__func__)
    run_retained=harness.OptionalRecordJoin.run_retained
    paired=harness.OptionalRecordJoin.paired

    def test_ordered_exact_field_context(self):
        for spread in (False,True):
            case='spread' if spread else 'direct'
            self.paired(case,ordered_source(spread),0)
            for number,compiler in enumerate(self.compilers):
                label=f'{case}-{number}-{compiler.name}'
                dump=self.run_retained(label+'-dump',[harness.ROOT/'bin/nanoisa','dump',self.work/(label+'.nvm')]).stdout
                self.assertIn('PUSH_U8 255',dump)
                self.assertIn('PUSH_U8 0',dump)
                self.assertIn('CAST_U8',dump)

    def test_nested_and_call_argument_contexts(self):
        direct=ordered_source()
        nested=direct.replace('let mut trace:int=0','struct Box {item:Bytes}\nlet mut trace:int=0')
        nested=nested.replace('let value:Bytes=Bytes{last:', 'let box:Box=Box{item:Bytes{last:')
        nested=nested.replace('first:(tick 3 256)}', 'first:(tick 3 256)}}\n let value:Bytes=box.item')
        self.paired('nested-order',nested,0)
        call=direct.replace('fn main()->int {',
            'fn keep(value:Bytes)->Bytes {return value}\n'
            'shadow keep {let b:Bytes=Bytes{first:0,middle:7,last:255} assert (== (keep b).middle 7)}\n'
            'fn main()->int {')
        call=call.replace('let value:Bytes=Bytes{last:', 'let value:Bytes=(keep Bytes{last:')
        call=call.replace('first:(tick 3 256)}', 'first:(tick 3 256)})')
        self.paired('call-order',call,0)

    def test_checked_imports_keep_distinct_record_identity(self):
        for file,field in (('u8-a.nano','left'),('u8-b.nano','right')):
            (self.work/file).write_text('struct Byte {'+field+':u8} '
                'pub fn make(value:int)->int {let b:Byte=Byte{'+field+':value} return (cast_int b.'+field+')} '
                'shadow make {assert (== (make 256) 0)}')
        self.paired('imported', 'module "u8-a.nano" as first\nmodule "u8-b.nano" as second\n'
            'fn main()->int {assert (== (first.make 256) 0) assert (== (second.make (- 0 1)) 255) return 0} '
            'shadow main {assert (== (main) 0)}',0)
        compiler=Path(os.environ.get('NANO_U8_NATIVE_COMPILER',harness.ROOT/'bin/nanoc_c'))
        binary=self.work/'imported-legacy'
        self.run_retained('imported-legacy-build',[compiler,self.work/'imported.nano','-o',binary],build=True)
        self.run_retained('imported-legacy-run',[binary])

    def test_field_refusals_preserve_output(self):
        for spread in (False,True):
            for case,expression in enumerate(('256','-1','true','1.5','"bad"')):
                label=('spread' if spread else 'direct')+'-bad-'+str(case)
                source=self.work/(label+'.nano')
                value=('{..base,value:'+expression+'}' if spread else 'Byte{value:'+expression+'}')
                source.write_text('struct Byte{value:u8} fn main()->int {let base:Byte=Byte{value:0} let bad:Byte='+value+' return 0} shadow main {assert true}')
                for number,compiler in enumerate(self.compilers):
                    output=self.work/(label+'-'+str(number)+'.nvm');output.write_bytes(b'prior')
                    result=self.run_retained(label+'-'+str(number),[compiler,source,'--emit-nvm','-o',output],None)
                    self.assertNotEqual(result.returncode,0)
                    self.assertEqual(output.read_bytes(),b'prior')
        for expression in ('Byte{value:1,value:2}','Byte{unknown:1}'):
            source=self.work/'bad-name.nano';source.write_text('struct Byte{value:u8} fn main()->int {let bad:Byte='+expression+' return 0} shadow main {assert true}')
            for number,compiler in enumerate(self.compilers):
                label='bad-name-'+str(number)+'-'+str(len(expression));output=self.work/(label+'.nvm');output.write_bytes(b'prior')
                result=self.run_retained(label,[compiler,source,'--emit-nvm','-o',output],None)
                self.assertNotEqual(result.returncode,0);self.assertEqual(output.read_bytes(),b'prior')

    def test_checked_registry_identity(self):
        binary=self.work/'identity'
        self.run_retained('identity-build',[*self.cc,'-std=c11','-Wall','-Wextra','-Werror','-I',harness.ROOT/'src',harness.ROOT/'tests/nanovirt/test_checked_record_identity.c','-o',binary],build=True)
        self.run_retained('identity-run',[binary])

    def test_native_c_profiles(self):
        compiler=Path(os.environ.get('NANO_U8_NATIVE_COMPILER',harness.ROOT/'bin/nanoc_c'))
        for spread in (False,True):
            label='legacy-'+('spread' if spread else 'direct');source=self.work/(label+'.nano');source.write_text(ordered_source(spread));binary=self.work/label
            self.run_retained(label+'-build',[compiler,source,'--keep-c','-o',binary],build=True)
            self.run_retained(label+'-run',[binary])
            # I qualify strict C's pure numeric boundary separately; its record
            # operand order and cast_int call support remain open profile gaps.
            if not spread:
                source=self.work/'strict-pure.nano'
                source.write_text('struct Byte{value:u8} fn input(value:int)->int {return value} '
                    'shadow input {assert (== (input 256) 256)} fn main()->int {'
                    'let zero:Byte=Byte{value:0} let top:Byte=Byte{value:255} '
                    'let wrapped:Byte=Byte{value:(input 256)} '
                    'let negative:Byte=Byte{value:(input (- 0 1))} '
                    'assert (== zero.value wrapped.value) '
                    'assert (== top.value negative.value) return 0} '
                    'shadow main {assert (== (main) 0)}')
            selected=self.work/(label+'-strict.c');selected.write_text('prior')
            emitted=self.run_retained(label+'-strict-emit',[compiler,source,'--target','c','-o',selected],None)
            if spread:
                self.assertNotEqual(emitted.returncode,0);self.assertEqual(selected.read_text(),'prior')
                self.assertIn('spread',emitted.stdout+emitted.stderr)
            else:
                self.assertEqual(emitted.returncode,0,emitted.stdout+emitted.stderr)
                for opt in ('-O0','-O2'):
                    binary=self.work/(label+opt)
                    self.run_retained(label+opt+'-strict-build',[*self.cc,'-std=c11',opt,'-Wall','-Wextra','-Werror','-Wno-unused-function','-fsanitize=address,undefined','-fno-sanitize-recover=all',selected,'-lm','-o',binary],build=True)
                    self.run_retained(label+opt+'-strict-run',[binary])
        for case,expression in enumerate(('256','-1','true','1.5','"bad"')):
            source=self.work/'native-bad.nano';source.write_text('struct Byte{value:u8} fn main()->int {let b:Byte=Byte{value:'+expression+'} return 0} shadow main {assert true}')
            for flags in ([],['--target','c']):
                label='native-bad-'+str(case)+'-'+str(len(flags));output=self.work/label;output.write_bytes(b'prior')
                result=self.run_retained(label,[compiler,source,*flags,'-o',output],None,build=True)
                self.assertNotEqual(result.returncode,0);self.assertEqual(output.read_bytes(),b'prior')

if __name__=='__main__':unittest.main()
