"""I execute interpreter binary-string shadows and actual generated native programs."""
import hashlib
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from tests import test_numeric_builtin_selection as numeric

ROOT = numeric.ROOT


class BstringCContracts(unittest.TestCase):
    command = numeric.NumericBuiltinSelection.command

    def test_extreme_generated_c_bounds_under_sanitizers(self):
        source = '''fn main()->int{
 let value:bstring=(bstr_new "abc")
 let tail:bstring=(bstr_substring value 1 9223372036854775807)
 let beyond:bstring=(bstr_substring value 9223372036854775807 9223372036854775807)
 let negative:bstring=(bstr_substring value -1 9223372036854775807)
 let negative_length:bstring=(bstr_substring value 1 -1)
 let empty:bstring=(bstr_substring value 3 9223372036854775807)
 assert (== (bstr_length tail) 2)
 assert (== (bstr_to_cstr tail) "bc")
 assert (== (bstr_length beyond) 0)
 assert (== (bstr_length negative) 0)
 assert (== (bstr_length negative_length) 0)
 assert (== (bstr_length empty) 0)
 assert (== (bstr_byte_at value -1) 0)
 assert (== (bstr_byte_at value 9223372036854775807) 0)
 (bstr_free tail) (bstr_free beyond) (bstr_free negative)
 (bstr_free negative_length) (bstr_free empty) (bstr_free value)
 return 0
}
# I check the production native body; useful interpreter builtin shadows remain open.
shadow main{assert true}
'''
        self.exercise(source)

    def exercise(self, source):
        output = ROOT/'.test_output/bstring-c-contracts'
        output.mkdir(parents=True, exist_ok=True)
        for optimization in ['-O0', '-O2']:
            with self.subTest(optimization=optimization):
                work = Path(tempfile.mkdtemp(prefix=optimization[1:]+'-', dir=output))
                path, product = work/'main.nano', work/'native'
                path.write_text(source)
                compiler = ROOT/'bin/nanoc_c'
                before = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in [path, compiler]}
                cflags = os.environ.get('NANO_CFLAGS', '')+' '+optimization+' -fsanitize=address,undefined -fno-sanitize-recover=all'
                product_before = None
                try:
                    with patch.dict(os.environ, {'NANO_CFLAGS': cflags}):
                        self.command(work, 'build', [compiler, path, '--keep-c', '-o', product], build=True)
                    self.assertIn(product.read_bytes()[:4], [b'\x7fELF', bytes.fromhex('cffaedfe'), bytes.fromhex('feedfacf')])
                    product_before = hashlib.sha256(product.read_bytes()).hexdigest()
                    self.assertEqual(self.command(work, 'run', [product]).stdout, '')
                finally:
                    after = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in [path, compiler]}
                    product_after = hashlib.sha256(product.read_bytes()).hexdigest() if product.exists() else None
                    (work/'inputs.json').write_text(json.dumps({'scope': 'I record actual C-seed shadow creation and native production execution; selected source shadows determine interpreter coverage.',
                        'native_build_cflags': cflags, 'before': before, 'after': after,
                        'product_before_run': product_before, 'product_after_run': product_after}, indent=2)+'\n')
                    self.assertEqual(before, after)
                    if product_before:
                        self.assertEqual(product_before, product_after)

    def test_all_builtins_useful_shadows_and_u8_storage(self):
        self.exercise('''let mut trace:int=0
fn mark(value:bstring,digit:int)->bstring{
 set trace (+ (* trace 10) digit)
 return value
}
shadow mark{
 set trace 0
 let value:bstring=(bstr_new "mark")
 assert (== (bstr_length (mark value 1)) 4)
 assert (== trace 1)
 (bstr_free value)
}
fn echo_octet(value:u8)->u8{return value}
shadow echo_octet{let value:u8=65 assert (== (echo_octet value) 65)}
fn keep_octet(value:u8)->bool{return (>= value 65)}
shadow keep_octet{let value:u8=65 assert (keep_octet value)}
fn keep_owner(value:bstring)->bstring{return value}
shadow keep_owner{
 let value:bstring=(bstr_new "owner")
 assert (bstr_equals (keep_owner value) value)
 (bstr_free value)
}
fn owner_present(value:bstring)->bool{return (> (bstr_length value) 0)}
shadow owner_present{
 let value:bstring=(bstr_new "owner")
 assert (owner_present value)
 (bstr_free value)
}
fn exercise()->int{
 let left:bstring=(bstr_new "Aé茶")
 assert (== (bstr_utf8_length left) -1)
 assert (bstr_validate_utf8 left)
 assert (== (bstr_utf8_length left) 3)
 assert (== (bstr_utf8_char_at left 1) 233)
 assert (== (bstr_utf8_char_at left 2) 33590)
 let right:bstring=(bstr_new "!")
 set trace 0
 let joined:bstring=(bstr_concat (mark left 1) (mark right 2))
 assert (== trace 12)
 let snapshot:string=(str_concat (bstr_to_cstr joined) "")
 (bstr_free left) (bstr_free right) (bstr_free joined)
 assert (== snapshot "Aé茶!")
 let first:u8=65 let zero:u8=0 let last:u8=66
 let bytes:array<u8> = [first,zero,last]
 let binary:bstring=(bstr_new_binary bytes)
 assert (== (bstr_length binary) 3)
 assert (== (bstr_byte_at binary 0) 65)
 assert (== (bstr_byte_at binary 1) 0)
 assert (== (bstr_byte_at binary 2) 66)
 let tail:bstring=(bstr_substring binary 1 9223372036854775807)
 assert (== (bstr_length tail) 2)
 assert (== (bstr_byte_at tail 0) 0)
 assert (== (bstr_byte_at tail 1) 66)
 let same:bstring=(bstr_new_binary bytes)
 assert (bstr_equals binary same)
 assert (== binary binary)
 assert (not (== binary same))
 let sliced:array<u8> = (array_slice bytes 1 2)
 let sliced_binary:bstring=(bstr_new_binary sliced)
 assert (bstr_equals tail sliced_binary)
 let integers:array<int> = [65,0,66]
 let ignored:bstring=(bstr_new_binary integers)
 assert (== (bstr_length ignored) 0)
 let converted:array<u8> = (bytes_from_string "OK")
 let from_text:bstring=(bstr_new_binary converted)
 assert (== (bstr_to_cstr from_text) "OK")
 (array_push converted first)
 assert (== (array_pop converted) 65)
 (array_set converted 0 last)
 let mut total:int=0
 for octet in converted {set total (+ total octet)}
 assert (== total 141)
 let reversed:array<u8> = (array_reverse converted)
 let reversed_owner:bstring=(bstr_new_binary reversed)
 assert (== (bstr_to_cstr reversed_owner) "KB")
 assert (array_contains converted last)
 assert (== (array_index_of converted last) 0)
 let mapped:array<u8> = (map converted echo_octet)
 let mapped_owner:bstring=(bstr_new_binary mapped)
 assert (== (bstr_to_cstr mapped_owner) "BK")
 let filtered:array<u8> = (filter converted keep_octet)
 let filtered_owner:bstring=(bstr_new_binary filtered)
 assert (bstr_equals mapped_owner filtered_owner)
 (bstr_free reversed_owner) (bstr_free mapped_owner) (bstr_free filtered_owner)
 let bad:u8=255 let invalid_bytes:array<u8> = [bad]
 let invalid:bstring=(bstr_new_binary invalid_bytes)
 assert (not (bstr_validate_utf8 invalid))
 assert (== (bstr_utf8_length invalid) -1)
 assert (== (bstr_utf8_char_at invalid 0) -1)
 (bstr_free invalid) (bstr_free from_text) (bstr_free ignored)
 (bstr_free sliced_binary) (bstr_free same) (bstr_free tail) (bstr_free binary)
 return 0
}
shadow exercise{
 assert (== (exercise) 0)
 let value:bstring=(bstr_new "array")
 let other:bstring=(bstr_new "other")
 let mut values:array<bstring> = [value]
 (array_push values other)
 assert (== (bstr_length (at values 0)) 5)
 let copied:array<bstring> = (array_slice values 0 1)
 assert (bstr_equals (at copied 0) value)
 let popped:bstring=(array_pop values)
 assert (bstr_equals popped other)
 (array_set values 0 other)
 assert (bstr_equals (at values 0) other)
 let mapped:array<bstring> = (map values keep_owner)
 assert (bstr_equals (at mapped 0) other)
 let filtered:array<bstring> = (filter values owner_present)
 assert (bstr_equals (at filtered 0) other)
 (bstr_free value) (bstr_free other)
}
fn main()->int{return (exercise)}
shadow main{assert (== (main) 0)}
''')

    def test_ordinary_lexical_callee_keeps_its_identity(self):
        self.exercise('''fn add_two(value:int)->int{return (+ value 2)}
shadow add_two{assert (== (add_two 3) 5)}
fn apply(bstr_length:fn(int)->int,value:int)->int{return (bstr_length value)}
shadow apply{assert (== (apply add_two 40) 42)}
fn main()->int{assert (== (apply add_two 40) 42) return 0}
shadow main{assert (== (main) 0)}
''')

    def test_ignored_invalid_calls_refuse_shadow_publication(self):
        bodies = ['let value:bstring=(bstr_new "x") (bstr_free value) (bstr_length value)',
                  'let value:bstring=(bstr_new "x") (bstr_free value) (bstr_free value)',
                  '(bstr_length "x")', '(bstr_new)']
        output = ROOT/'.test_output/bstring-c-contracts'
        output.mkdir(parents=True, exist_ok=True)
        for index, body in enumerate(bodies):
            with self.subTest(body=body):
                work = Path(tempfile.mkdtemp(prefix='refused-', dir=output))
                source, product = work/'main.nano', work/'native'
                source.write_text('fn main()->int{return 0}\nshadow main{'+body+' assert true}\n')
                prior = b'I retain my prior program after failed binary-string shadows.\n'
                product.write_bytes(prior)
                result = self.command(work, 'refusal', [ROOT/'bin/nanoc_c', source, '-o', product], build=True, accepted=False)
                self.assertNotEqual(result.returncode, 0)
                self.assertRegex(result.stdout+result.stderr, r'(?i)(binary.string|type|shadow|argument|arity)')
                self.assertEqual(product.read_bytes(), prior)
