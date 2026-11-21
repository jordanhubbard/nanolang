"""I retain byte storage through computed writes and returned handles."""
from pathlib import Path
import os,tempfile,unittest
from tests.nanolang_translator_routes import build_routes,checked
from tests.native_process import run_captured
ROOT=Path(__file__).resolve().parents[1]


def source(reverse=False,direct=False):
    value='number' if direct else '(+ number 0)'
    helpers=[f'''fn append_byte(bytes: array<u8>, number: int) -> array<u8> {{
    let mut result: array<u8> = bytes
    set result (array_push result {value})
    return result
}}
shadow append_byte {{
    let initial: array<u8> = []
    let first: array<u8> = (append_byte initial 300)
    let second: array<u8> = (append_byte first -1)
    assert (== (at second 0) 44)
    assert (== (at second 1) 255)
}}
''',f'''fn replace_byte(bytes: array<u8>, index: int, number: int) -> array<u8> {{
    (array_set bytes index {value})
    return bytes
}}
shadow replace_byte {{
    let initial: array<u8> = [9]
    assert (== (at (replace_byte initial 0 511) 0) 255)
}}
''','''fn forward(bytes: array<u8>, number: int) -> array<u8> {
    return (append_byte bytes number)
}
shadow forward {
    let initial: array<u8> = []
    assert (== (at (forward initial 257) 0) 1)
}
''','''fn encode(initial: array<u8>) -> Packet {
    let start: int = (array_length initial)
    let first: array<u8> = (forward initial 300)
    let second: array<u8> = (append_byte first -1)
    let alias: array<u8> = second
    let changed: array<u8> = (replace_byte alias start 511)
    assert (== (array_length changed) (+ start 2))
    assert (== (at second start) 255)
    let snapshot: array<u8> = (array_slice changed start 2)
    let last: array<u8> = (replace_byte changed start 1)
    assert (== (at snapshot 0) 255)
    assert (== (at snapshot 1) 255)
    assert (== (at last start) 1)
    return Packet { bytes: last, snapshot: snapshot }
}
shadow encode {
    let initial: array<u8> = []
    let packet: Packet = (encode initial)
    assert (== (at packet.bytes 0) 1)
    assert (== (at packet.snapshot 0) 255)
}
''']
    if reverse:helpers.reverse()
    return 'struct Packet { bytes: array<u8>, snapshot: array<u8> }\n'+''.join(helpers)+'''fn main() -> int {
    let empty: array<u8> = []
    let first: Packet = (encode empty)
    assert (== (array_length first.bytes) 2)
    let nonempty: array<u8> = [9]
    let second: Packet = (encode nonempty)
    assert (== (at second.bytes 0) 9)
    assert (== (array_length second.bytes) 3)
    let dynamic: array<u8> = (bytes_from_string "Z")
    let third: Packet = (encode dynamic)
    assert (== (at third.bytes 0) 90)
    assert (== (array_length third.bytes) 3)
    (println (at first.snapshot 0))
    (println (at second.snapshot 1))
    return 0
}
'''


class ByteArrayResults(unittest.TestCase):
    def test_computed_and_direct_writes_both_function_orders(self):
        with tempfile.TemporaryDirectory(prefix='nano-byte-return-') as directory:
            work=Path(directory)
            for reverse in (False,True):
                for direct in (False,True):
                    fixture=work/f'bytes-{reverse}-{direct}.nano';fixture.write_text(source(reverse,direct))
                    for route,command in build_routes(fixture,work):
                        with self.subTest(reverse=reverse,direct=direct,route=route):self.assertEqual(checked(command),'255\n255\n')

    def test_original_generic_and_byte_fixtures_available_producers(self):
        # I execute the original fixture with actual available images. This
        # does not replace its separate installed Stage1/Stage2 gate.
        with tempfile.TemporaryDirectory(prefix='nano-byte-return-generic-') as directory:
            for fixture in ('tests/nanoisa/fixtures/generic_scalar_array.nano','tests/test_u8_basic.nano','tests/test_bytes_conversion.nano'):
                for route,command in build_routes(ROOT/fixture,Path(directory)):
                    with self.subTest(fixture=fixture,route=route):self.assertEqual(checked(command),'')

    def test_invalid_source_preserves_prior_products(self):
        producers=[(ROOT/'bin/nanoc_c',[]),(ROOT/'bin/nano_virt',[])]
        if os.environ.get('NANOLANG_TRANSLATOR_SOURCE_COMPILER'):producers.append((Path(os.environ['NANOLANG_TRANSLATOR_SOURCE_COMPILER']),['--emit-nvm']))
        with tempfile.TemporaryDirectory(prefix='nano-byte-return-refusal-') as directory:
            work=Path(directory);fixture=work/'bad.nano'
            for expression in ('true','"wrong"','[1]'):
                for operation in ('push','set'):
                    for initial in ('[]','[7]'):
                        statement='let result:array<u8> = (array_push bytes '+expression+')' if operation=='push' else '(array_set bytes 0 '+expression+')'
                        fixture.write_text('fn main()->int { let bytes:array<u8> = '+initial+' '+statement+' return 0 }\n')
                        for index,(producer,flags) in enumerate(producers):
                            output=work/f'prior-{index}.nvm';prior=b'I preserve this prior product.';output.write_bytes(prior)
                            result=run_captured([producer,fixture,*flags,'-o',output],cwd=ROOT,build=True,timeout=60)
                            with self.subTest(operation=operation,initial=initial,expression=expression,producer=str(producer)):
                                self.assertNotEqual(result.returncode,0)
                                self.assertEqual(output.read_bytes(),prior)

    def test_incompatible_native_element_preserves_prior_output(self):
        assembly='''.entry main
.function append 2 2 0 array 1
.parameters append array int
LOAD_LOCAL 0
LOAD_LOCAL 1
PUSH_I64 0
I64_ADD
ARR_PUSH
RET
.end
.function main 0 0 0 int 1
PUSH_BOOL 1
ARR_LITERAL 4 1
PUSH_I64 7
CALL append
POP
PUSH_I64 0
RET
.end
'''
        with tempfile.TemporaryDirectory(prefix='nano-byte-return-native-refusal-') as directory:
            work=Path(directory);asm=work/'bad.nasm';module=work/'bad.nvm';output=work/'prior.c'
            asm.write_text(assembly);checked([ROOT/'bin/nanoisa','asm',asm,'-o',module])
            prior=b'I preserve this prior native source.';output.write_bytes(prior)
            result=run_captured([ROOT/'bin/nvm2c',module,'-o',output],cwd=ROOT,build=True,timeout=60)
            self.assertNotEqual(result.returncode,0)
            self.assertEqual(output.read_bytes(),prior)
            self.assertIn('matching scalar shape',result.stderr.decode())


if __name__=='__main__':unittest.main()
