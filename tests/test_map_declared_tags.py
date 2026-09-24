"""I preserve declared map writes across ordinary VM and native products."""
try:
    from tests.sanitizer_options import asan_options
except ModuleNotFoundError:
    from sanitizer_options import asan_options
from pathlib import Path
import os
import shlex
import signal
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
SANITIZER_CC = shlex.split(os.environ.get("NANOLANG_GUARD_SAN_CC", os.environ.get("CC", "cc")))

class DeclaredMapTags(unittest.TestCase):
    def command(self, args, success=True):
        result = subprocess.run([str(x) for x in args], capture_output=True, text=True,
            timeout=90, env={**os.environ, 'ASAN_OPTIONS':asan_options("abort_on_error=1")})
        self.assertEqual(result.returncode == 0, success, str(args)+'\n'+result.stdout+result.stderr)
        return result

    def native(self, work, module):
        source, binary = work/'native.c', work/'native'
        self.command([ROOT/'bin/nvm2c',module,'-o',source])
        self.command([*SANITIZER_CC,'-std=c11','-O2','-Wall','-Wextra','-Werror',
            '-fsanitize=address,undefined','-fno-sanitize-recover=all',source,'-o',binary])
        return binary

    def test_ordinary_source_insertion_replacement_and_aliases(self):
        for value_type, first, second in [('int','42','7'),('string','"first"','"second"')]:
            with self.subTest(value_type=value_type), tempfile.TemporaryDirectory() as tmp:
                work=Path(tmp); source=work/'map.nano'; module=work/'map.nvm'
                source.write_text(f'''fn main() -> int {{
 let values: HashMap<string,{value_type}> = (map_new)
 let alias: HashMap<string,{value_type}> = values
 (map_set values "key" {first})
 assert (== (map_get alias "key") {first})
 (map_set alias "key" {second})
 assert (== (map_get values "key") {second})
 assert (== (map_size values) 1)
 return 0
}}
shadow main {{ assert (== (main) 0) }}
''')
                self.command([ROOT/'bin/nano_virt',source,'--emit-nvm','-o',module])
                self.command([ROOT/'bin/nano_vm',module])
                self.command([self.native(work,module)])

    def test_exact_dynamic_value_guard_matches_native(self):
        for declared, value in [(1,'PUSH_STR text'),(5,'PUSH_I64 7')]:
            with self.subTest(declared=declared), tempfile.TemporaryDirectory() as tmp:
                work=Path(tmp); source=work/'write.nasm'; module=work/'write.nvm'
                source.write_text('.entry main\n.string key "key"\n.string text "text"\n'
                    '.function main 0 0 0 int 1\n'+f'HM_NEW 5 {declared}\nSTORE_GLOBAL 0\n'
                    'LOAD_GLOBAL 0\nPUSH_STR key\n'+value+'\nHM_SET\nPOP\nPUSH_I64 0\nRET\n.end\n')
                self.command([ROOT/'bin/nanoisa','asm',source,'-o',module])
                vm=self.command([ROOT/'bin/nano_vm',module],success=False)
                self.assertIn('HM_SET key/value tags',vm.stderr)
                native=self.command([self.native(work,module)],success=False)
                self.assertEqual(native.returncode,-signal.SIGABRT,native.stderr)
                self.assertIn('I stopped at a native invariant',native.stderr)
                for diagnostic in ('Sanitizer','runtime error:'):
                    self.assertNotIn(diagnostic,native.stderr)

if __name__ == '__main__':
    unittest.main()
