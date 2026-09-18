"""I qualify supported C storage and refuse legacy fallback representations."""
import os
from pathlib import Path
import unittest
from tests import test_public_c_nonfinite_format as base
ROOT=Path(__file__).resolve().parents[1]
class SupportedTypes(unittest.TestCase):
    setUp=base.PublicCFormat.setUp
    run_cmd=base.PublicCFormat.run_cmd
    compile=base.PublicCFormat.compile
    emit=base.PublicCFormat.emit
    def test_api_kind_inventory_order_and_publication_recovery(self):
        api=self.work/'api'
        self.run_cmd([os.environ.get('CC','cc'),'-std=c99','-D_POSIX_C_SOURCE=200809L',
            '-Wall','-Wextra','-Werror','-O1','-fsanitize=address,undefined','-fno-sanitize-recover=all',
            '-I',ROOT/'src',ROOT/'tests/test_public_c_supported_types_api.c','-o',api])
        for mode in ('record','int','u8','enum','union'):
            output=self.work/(mode+'.c');self.run_cmd([api,output,mode])
            # Carrier API modes separately retain and observe their exact value.
            if mode in ('int','u8','enum'):
                with output.open('a') as f:f.write('\nint carrier_observer(void) { return helper(7) == 7 ? 0 : 1; }\n')
                text=output.read_text();text=text.replace('int main(void) {','int carrier_observer(void);\nint main(void) { if (carrier_observer()) return 1;')
                output.write_text(text)
            for standard in ('c99','c11'):
                for optimization in ('-O0','-O2'):
                    exe=self.work/(mode+standard+optimization);self.compile(output,exe,standard,optimization);self.run_cmd([exe])
    def test_prior_record_order_source_and_scalar_fields(self):
        source='''struct Inner{number:int,text:string,value:float}
struct Outer{inner:Inner,ready:bool}
fn make()->Outer{return Outer{inner:Inner{number:7,text:"kept",value:2.5},ready:true}}
shadow make{let x:Outer=(make) assert (== x.inner.number 7)}
fn main()->int{let x:Outer=(make) assert x.ready assert (== x.inner.number 7) assert (== x.inner.text "kept") assert (== x.inner.value 2.5) return 0}
shadow main{assert true}
'''
        output,_=self.emit(source)
        for standard in ('c99','c11'):
            for optimization in ('-O0','-O2'):
                exe=self.work/(standard+optimization);self.compile(output,exe,standard,optimization);self.run_cmd([exe])
    def test_dotted_union_scalar_and_empty_source_constructors(self):
        source='''union Choice{Some{number:int},None{}}
fn value()->Choice{return Choice.Some{number:7}}
shadow value{assert true}
fn main()->int{let x:Choice=(value) let empty:Choice=Choice.None{}
match x{Some(data)=>{assert (== data.number 7)},None=>{assert false}}
match empty{Some(data)=>{assert false},None=>{assert true}}
return 0}
shadow main{assert true}
'''
        output,_=self.emit(source)
        for standard in ('c99','c11'):
            for optimization in ('-O0','-O2'):
                exe=self.work/(standard+optimization);self.compile(output,exe,standard,optimization);self.run_cmd([exe])
    def test_source_opaque_and_record_order_refusals(self):
        sources=[
            'extern fn pointer_input(value:opaque)->int fn main()->int{return 0}',
            'struct Outer{inner:Inner} struct Inner{number:int} fn main()->int{return 0}',
            'struct Self{next:Self} fn main()->int{return 0}',
            'struct First{next:Second} struct Second{next:First} fn main()->int{return 0}',
        ]
        for index,source in enumerate(sources):
            path=self.work/f'ordinary{index}.nano';path.write_text(source+'\nshadow main{assert true}\n')
            output=self.work/'previous.c';output.write_text('previous')
            result=self.run_cmd([ROOT/'bin/nanoc_c','--target','c',path,'-o',output],expected=1)
            self.assertRegex(result.stderr,'supported exact C value representation|exact complete local C nominal declaration')
            self.assertEqual(output.read_text(),'previous')
if __name__=='__main__':unittest.main()
