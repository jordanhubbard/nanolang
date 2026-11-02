"""I retain exact literal element context at array mutation boundaries."""
import unittest
from tests import test_optional_record_join as harness


class NestedArrayMutationContext(unittest.TestCase):
    setUpClass = classmethod(harness.OptionalRecordJoin.setUpClass.__func__)
    run_retained = harness.OptionalRecordJoin.run_retained
    paired = harness.OptionalRecordJoin.paired

    def test_empty_replacement_and_append_retain_storage(self):
        for kind, value, expected in [('string', '"after"', '"after"'),
                                      ('bool', 'true', 'true'), ('float', '1.25', '1.25'),
                                      ('int', '7', '7'), ('u8', '300', '44')]:
            text = f'''struct Box {{ rows:array<array<{kind}>> }}
fn main()->int {{
 let box:Box=Box{{rows:[[]]}}
 let alias:array<array<{kind}>> =box.rows
 (array_set box.rows 0 [])
 let row:array<{kind}> =(at alias 0)
 let filled:array<{kind}> =(array_push row {value})
 assert (== (at filled 0) {expected})
 let added:array<array<{kind}>> =(array_push box.rows [])
 let last:array<{kind}> =(array_push (at added 1) {value})
 assert (== (at last 0) {expected})
 (array_set added 0 [{value}])
 assert (== (at (at added 0) 0) {expected})
 let literal:array<array<{kind}>> =(array_push added [{value}])
 assert (== (at (at literal 2) 0) {expected})
 return 0
}}
shadow main {{assert (== (main) 0)}}
'''
            if kind == 'u8':
                for value_read in ('(at filled 0)', '(at last 0)', '(at (at added 0) 0)', '(at (at literal 2) 0)'):
                    text = text.replace('(== '+value_read+' 44)', '(== (cast_int '+value_read+') 44)')
            self.paired('storage-'+kind, text, 0)

    def test_receiver_index_and_value_execute_once_in_order(self):
        self.paired('ordered', '''struct Box {rows:array<array<string>>}
let mut trace:int=0
fn receiver(box:Box)->array<array<string>> {set trace (+ (* trace 10) 1) return box.rows}
shadow receiver {let b:Box=Box{rows:[[]]} assert (== (array_length (receiver b)) 1)}
fn index()->int {set trace (+ (* trace 10) 2) return 0}
shadow index {assert (== (index) 0)}
fn value()->string {set trace (+ (* trace 10) 3) return "kept"}
shadow value {assert (== (value) "kept")}
fn main()->int {
 let box:Box=Box{rows:[[]]}
 set trace 0
 (array_set (receiver box) (index) [(value)])
 assert (== trace 123)
 assert (== (at (at box.rows 0) 0) "kept")
 set trace 0
 let result:array<array<string>> = (array_push (receiver box) [(value)])
 assert (== trace 13)
 assert (== (at (at result 1) 0) "kept")
 return 0
}
shadow main {assert (== (main) 0)}
''', 0)

    def test_declared_mutation_names_keep_their_own_signatures(self):
        self.paired('declared-names', '''fn array_push(rows:array<array<string>>, value:array<bool>)->int {return (array_length value)}
shadow array_push {assert (== (array_push [["kept"]] [false]) 1)}
fn main()->int {
 let rows:array<array<string>> = [["kept"]]
 assert (== (array_push rows [false]) 1)
 assert (== (at (at rows 0) 0) "kept")
 return 0
}
shadow main {assert (== (main) 0)}
''', 0)

    def test_wrong_literal_and_nonliteral_storage_preserve_output(self):
        for operation in ('array_set rows 0', 'array_push rows'):
            for expression in ('[true]', 'wrong'):
                source = self.work/'wrong.nano'
                source.write_text('fn main()->int {let rows:array<array<string>> = [[]] '
                                  'let wrong:array<int> = [] ('+operation+' '+expression+') return 0}\n'
                                  'shadow main {assert true}\n')
                for index, compiler in enumerate(self.compilers):
                    output = self.work/'prior.nvm'; output.write_bytes(b'previous')
                    result = self.run_retained('refusal-'+str(index)+'-'+operation.split()[0]+'-'+str(expression=='wrong'),
                                               [compiler, source, '--emit-nvm', '-o', output], expected=None)
                    self.assertNotEqual(result.returncode, 0)
                    self.assertEqual(output.read_bytes(), b'previous')


if __name__ == '__main__':
    unittest.main()
