"""I qualify exact byte record fields without expanding integer coercion."""
import re
import struct
import unittest
from tests import test_optional_record_join as harness


def list_source(case):
    values = '[]' if case == 'empty' else '[255]'
    checked = 0 if case == 'unchecked' else 1
    index = 1 if case == 'raw-missing' else 0
    condition = ('(== checked 1)' if case == 'raw-missing' else
                 '(and (== checked 1) (== (array_length f.owner) 1))')
    expected = 0 if case in ('empty', 'unchecked') else 255
    return f'''struct Fact {{ owner: array<u8>, completed: u8 }}
struct Parser {{ functions: List<Fact> }}
fn update(p: Parser, checked: int) -> void {{
    let fallback: u8 = 0
    let f: Fact = (list_Fact_get p.functions 0)
    (list_Fact_set p.functions 0 Fact {{ owner: f.owner,
        completed: (cond ({condition} (at f.owner {index})) (else fallback)) }})
}}
shadow update {{
    let octet: u8 = 255
    let zero: u8 = 0
    let functions: List<Fact> = (list_Fact_new)
    (list_Fact_push functions Fact {{ owner: [255], completed: octet }})
    (update Parser {{ functions: functions }} 0)
    assert (== (list_Fact_get functions 0).completed zero)
}}
fn main() -> int {{
    let initial: u8 = 0
    let expected: u8 = {expected}
    let functions: List<Fact> = (list_Fact_new)
    (list_Fact_push functions Fact {{ owner: {values}, completed: initial }})
    (update Parser {{ functions: functions }} {checked})
    assert (== (list_Fact_get functions 0).completed expected)
    return 0
}}
# I exercise main in the VM/native matrix; raw-missing intentionally fails.
shadow main {{ assert true }}
'''


class U8RecordFields(unittest.TestCase):
    # I reuse the retained source/VM/native harness without inheriting its suite.
    setUpClass = classmethod(harness.OptionalRecordJoin.setUpClass.__func__)
    run_retained = harness.OptionalRecordJoin.run_retained
    paired = harness.OptionalRecordJoin.paired

    def test_direct_nested_and_list_fields(self):
        source = '''struct Byte { low:u8, high:u8 }
struct Box { item:Byte }
fn main()->int {
    let zero:u8=0 let high:u8=255
    let value:Byte=Byte{low:zero,high:high}
    let nested:Box=Box{item:value}
    assert (== nested.item.low zero) assert (== nested.item.high high)
    let values:List<Byte> = (list_Byte_new)
    (list_Byte_push values value)
    assert (== (list_Byte_get values 0).high high)
    (list_Byte_set values 0 Byte{low:high,high:zero})
    assert (== (list_Byte_get values 0).high zero)
    assert (== nested.item.high high)
    return 0
}
shadow main { assert (== (main) 0) }
'''
        self.paired('direct-nested-list', source, 0)

    def test_exact_u8_layout(self):
        self.paired('direct-layout', 'struct Byte {low:u8,high:u8} fn main()->int {'
                    'let low:u8=0 let high:u8=255 let b:Byte=Byte{low:low,high:high} '
                    'assert (== b.low low) assert (== b.high high) return 0} '
                    'shadow main {assert (== (main) 0)}', 0)
        for number, compiler in enumerate(self.compilers):
            label = f'direct-layout-{number}-{compiler.name}'
            dump = self.run_retained(label+'-dump',
                [harness.ROOT/'bin/nanoisa', 'dump', self.work/(label+'.nvm')]).stdout
            layouts = bytes.fromhex(''.join(re.findall(r'^\.layouts "([0-9a-f]*)"$',dump,re.M)))
            self.assertEqual(len(layouts),36)
            self.assertEqual(struct.unpack_from('<I',layouts)[0],1)
            self.assertEqual(struct.unpack_from('<BBH',layouts,4),(0,0,2))
            for offset in (12,24):
                tag, nested, name = struct.unpack_from('<B3xII',layouts,offset)
                self.assertEqual((tag,nested),(2,0xffffffff))

    def test_conditional_byte_fields(self):
        for case in ('present', 'empty', 'unchecked', 'raw-missing'):
            self.paired('u8-'+case, list_source(case),
                        None if case == 'raw-missing' else 0, 'Assertion failed')

    def test_wrong_field_types_preserve_output(self):
        for value in ('true', '1.5', '"bad"'):
            source = self.work/('wrong-'+str(len(value))+'.nano')
            source.write_text('struct Byte {value:u8} fn main()->int {let b:Byte=Byte{value:'+
                              value+'} return 0} shadow main {assert true}')
            for number, compiler in enumerate(self.compilers):
                label = 'wrong-'+str(len(value))+'-'+str(number)
                output = self.work/(label+'.nvm')
                output.write_bytes(b'prior qualified output')
                result = self.run_retained(label, [compiler,source,'--emit-nvm','-o',output], None)
                self.assertNotEqual(result.returncode,0)
                self.assertEqual(output.read_bytes(),b'prior qualified output')

    def test_unconsumed_resource_stays_rejected(self):
        source=self.work/'resource.nano'
        source.write_text('resource struct Byte {value:u8} fn main()->int {'
                          'let v:u8=7 let owner:Byte=Byte{value:v} return 0} '
                          'shadow main {assert true}')
        for number, compiler in enumerate(self.compilers):
            label='unconsumed-'+str(number);output=self.work/(label+'.nvm')
            output.write_bytes(b'prior qualified output')
            result=self.run_retained(label,[compiler,source,'--emit-nvm','-o',output],None)
            self.assertNotEqual(result.returncode,0)
            self.assertEqual(output.read_bytes(),b'prior qualified output')

    def test_out_of_range_byte_literal_preserves_output(self):
        for context, body in (('local', 'let bad:u8=256 let b:Byte=Byte{value:bad}'),):
            source = self.work/('byte-256-'+context+'.nano')
            source.write_text('struct Byte {value:u8} fn main()->int {'+body+
                              ' return 0} shadow main {assert true}')
            for number, compiler in enumerate(self.compilers):
                label = 'byte-256-'+context+'-'+str(number); output=self.work/(label+'.nvm')
                output.write_bytes(b'prior qualified output')
                result=self.run_retained(label,[compiler,source,'--emit-nvm','-o',output],None)
                self.assertNotEqual(result.returncode,0)
                self.assertEqual(output.read_bytes(),b'prior qualified output')

    def test_literal_field_range_refusal_is_preserved(self):
        for compiler in self.compilers:
            label=compiler.name+'-raw-literal-256'
            source=self.work/(label+'.nano');output=self.work/(label+'.nvm')
            source.write_text('struct Byte {value:u8} fn main()->int {let b:Byte=Byte{value:256} return 0} shadow main {assert true}')
            output.write_bytes(b'prior qualified output')
            result=self.run_retained(label,[compiler,source,'--emit-nvm','-o',output],None)
            self.assertNotEqual(result.returncode,0)
            self.assertEqual(output.read_bytes(),b'prior qualified output')


if __name__ == '__main__':
    unittest.main()
