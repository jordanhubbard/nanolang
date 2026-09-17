"""I retain map result annotations and evaluate operation arguments once."""
from pathlib import Path
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]


class NativeReturnedMaps(unittest.TestCase):
    def checked(self, command):
        result = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, timeout=120)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        return result

    def execute(self, source, vm=True):
        with tempfile.TemporaryDirectory(prefix='nano-returned-map-') as directory:
            work = Path(directory)
            path = work/'case.nano'
            path.write_text(source)
            self.checked([ROOT/'bin/nanoc_c', path, '-o', work/'native'])
            self.checked([work/'native'])
            if vm:
                self.checked([ROOT/'bin/nano_virt', path, '--emit-nvm', '-o', work/'case.nvm'])
                self.checked([ROOT/'bin/nano_vm', work/'case.nvm'])

    def source(self, key, value):
        k = '7' if key == 'int' else '"key"'
        v = '42' if value == 'int' else '"kept"'
        typ = f'HashMap<{key},{value}>'
        return f'''let mut trace: int = 0
let mut made: int = 0
fn receiver(values: {typ}) -> {typ} {{ set trace (+ (* trace 10) 1) return values }}
shadow receiver {{ let values: {typ} = (map_new) set trace 0 assert (== (map_size (receiver values)) 0) assert (== trace 1) }}
fn read_key() -> {key} {{ set trace (+ (* trace 10) 2) return {k} }}
shadow read_key {{ set trace 0 assert (== (read_key) {k}) assert (== trace 2) }}
fn read_value() -> {value} {{ set trace (+ (* trace 10) 3) return {v} }}
shadow read_value {{ set trace 0 assert (== (read_value) {v}) assert (== trace 3) }}
fn fresh() -> {typ} {{ set made (+ made 1) let values: {typ} = (map_new) (map_put values {k} {v}) return values }}
shadow fresh {{ let values: {typ} = (fresh) assert (== (map_get values {k}) {v}) }}
fn returned_value() -> {value} {{ return (map_get (fresh) {k}) }}
shadow returned_value {{ assert (== (returned_value) {v}) }}
fn main() -> int {{
 let values: {typ} = (map_new)
 set trace 0
 (map_set (receiver values) (read_key) (read_value))
 assert (== trace 123)
 set trace 0
 assert (== (map_get (receiver values) (read_key)) {v})
 assert (== trace 12)
 set trace 0
 assert (map_has (receiver values) (read_key))
 assert (== trace 12)
 set trace 0
 assert (== (map_size (receiver values)) 1)
 assert (== trace 1)
 set trace 0
 assert (== (map_length (receiver values)) 1)
 assert (== trace 1)
 set trace 0
 let keys: array<{key}> = (map_keys (receiver values))
 assert (== (array_length keys) 1)
 assert (== (at keys 0) {k})
 assert (== trace 1)
 set trace 0
 let items: array<{value}> = (map_values (receiver values))
 assert (== (array_length items) 1)
 assert (== (at items 0) {v})
 assert (== trace 1)
 set trace 0
 (map_remove (receiver values) (read_key))
 assert (== trace 12)
 assert (== (map_size values) 0)
 set trace 0
 (map_put (receiver values) (read_key) (read_value))
 assert (== trace 123)
 set trace 0
 (map_clear (receiver values))
 assert (== trace 1)
 assert (== (map_size values) 0)
 set made 0
 assert (== (returned_value) {v})
 assert (== made 1)
 return 0
}}
shadow main {{ assert (== (main) 0) }}
'''

    def test_all_scalar_pairs_and_operation_order(self):
        for key in ('int', 'string'):
            for value in ('int', 'string'):
                with self.subTest(key=key, value=value):
                    self.execute(self.source(key, value))

    def test_nested_receiver_calls(self):
        source = self.source('string', 'string').replace('(receiver values)', '(receiver (receiver values))')
        source = source.replace('(== trace 123)', '(== trace 1123)').replace('(== trace 12)', '(== trace 112)').replace('(== trace 1)', '(== trace 11)')
        self.execute(source)

    def test_existing_native_explicit_free_order(self):
        # NanoVirt does not expose the native map_free operation.
        source = self.source('string', 'string').replace(' return 0\n}',
            ' set trace 0\n (map_free (receiver values))\n assert (== trace 1)\n return 0\n}')
        self.execute(source, vm=False)

    def test_checked_callback_result(self):
        self.execute('''fn fresh() -> HashMap<string,string> { let values: HashMap<string,string> = (map_new) (map_put values "key" "kept") return values }
shadow fresh { let values: HashMap<string,string> = (fresh) assert (== (map_get values "key") "kept") }
fn apply(factory: fn() -> HashMap<string,string>) -> string { return (map_get (factory) "key") }
shadow apply { assert (== (apply fresh) "kept") }
fn main() -> int { assert (== (apply fresh) "kept") return 0 }
shadow main { assert (== (main) 0) }
''')

    def test_wrong_returned_map_arguments_preserve_output(self):
        for key in ('int', 'string'):
            for value in ('int', 'string'):
                k = '7' if key == 'int' else '"key"'
                v = '42' if value == 'int' else '"kept"'
                bad_k = '"wrong"' if key == 'int' else '9'
                bad_v = '"wrong"' if value == 'int' else '9'
                for expression in (f'(map_get (fresh) {bad_k})', f'(map_set (fresh) {k} {bad_v})', '(map_get (fresh))'):
                    with tempfile.TemporaryDirectory() as directory:
                        work = Path(directory)
                        path, output = work/'bad.nano', work/'prior'
                        path.write_text(f'fn fresh() -> HashMap<{key},{value}> {{ let values: HashMap<{key},{value}> = (map_new) return values }}\nshadow fresh {{ assert true }}\nfn main() -> int {{ {expression} return 0 }}\nshadow main {{ assert true }}\n')
                        for compiler in ('nanoc_c', 'nano_virt'):
                            with self.subTest(key=key, value=value, expression=expression, compiler=compiler):
                                output.write_text('prior artifact')
                                command = [ROOT/'bin'/compiler, path, '-o', output]
                                if compiler == 'nano_virt': command.append('--emit-nvm')
                                result = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, timeout=120)
                                self.assertNotEqual(result.returncode, 0, result.stdout + result.stderr)
                                self.assertIn('E001 TYPE MISMATCH', result.stderr)
                                self.assertEqual(output.read_text(), 'prior artifact')


if __name__ == '__main__': unittest.main()
