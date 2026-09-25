"""I keep declared filesystem artifacts exact across VM and native products."""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from tests.native_toolchain import native_cc, native_link_flags

ROOT = Path(__file__).resolve().parents[1]
COMPILER = Path(os.environ.get('NANOLANG_SELFHOST_COMPILER', ROOT / 'bin/nanoc_stage2')).resolve()


class CanonicalFilesystem(unittest.TestCase):
    def command(self, args, expected=0):
        result = subprocess.run([str(x) for x in args], cwd=ROOT, capture_output=True, timeout=180)
        self.assertEqual(result.returncode, expected, (result.stdout + result.stderr)[-6000:])
        return result

    def execute_both(self, source, directory):
        module = directory / 'program.nvm'
        self.command([COMPILER, source, '--emit-nvm', '-o', module])
        self.command([ROOT / 'bin/nano_vm', '--verify-only', module])
        self.command([ROOT / 'bin/nano_vm', module])
        generated, native = directory / 'program.c', directory / 'native'
        self.command([ROOT / 'bin/nvm2c', module, '-o', generated])
        self.command([*native_cc(), '-std=c11', '-Wall', '-Wextra', '-Werror', generated,
                      ROOT / 'bin/nano_aot_runtime.o', '-lm',
                      *(['-Wl,--export-dynamic', '-ldl'] if sys.platform.startswith('linux') else []),
                      *native_link_flags(), '-o', native])
        self.command([native])

    def test_declared_results_and_owned_arrays(self):
        with tempfile.TemporaryDirectory(prefix='canonical filesystem ') as tmp:
            directory = Path(tmp)
            fixtures = directory / 'fixtures'
            fixtures.mkdir()
            (fixtures / 'A.TXT').write_text('first')
            (fixtures / 'b.txt').write_text('second')
            (fixtures / 'other.bin').write_bytes(b'123')
            (fixtures / 'child').mkdir()
            source = directory / 'main.nano'
            source.write_text('module "modules/filesystem/filesystem.nano"\n'
                'fn main() -> int {\n'
                f' let dir: string = {json.dumps(str(fixtures))}\n'
                ' let all: array<string> = (nl_fs_list_files_ci dir ".txt")\n'
                ' assert (== (array_length all) 2)\n'
                ' assert (== (at all 0) "A.TXT")\n'
                ' assert (== (at all 1) "b.txt")\n'
                ' let exact: array<string> = (nl_fs_list_files dir ".txt")\n'
                ' assert (== (array_length exact) 1)\n'
                ' assert (== (at exact 0) "b.txt")\n'
                ' let dirs: array<string> = (nl_fs_list_dirs dir)\n'
                ' assert (== (array_length dirs) 1)\n'
                ' assert (== (at dirs 0) "child")\n'
                ' let none: array<string> = (nl_fs_list_files_ci dir ".absent")\n'
                ' assert (== (array_length none) 0)\n'
                ' let joined: string = (nl_fs_join_path dir "other.bin")\n'
                ' assert (== (nl_fs_file_size joined) 3)\n'
                ' assert (== (nl_fs_file_exists joined) 1)\n'
                ' assert (== (nl_fs_is_directory dir) 1)\n'
                ' assert (== (nl_fs_is_directory joined) 0)\n'
                ' let parent: string = (nl_fs_parent_dir joined)\n'
                ' assert (== parent dir)\n'
                ' let other_parent: string = (nl_fs_parent_dir "/different/child")\n'
                ' let other_joined: string = (nl_fs_join_path "/different" "file")\n'
                ' assert (== parent dir)\n'
                ' assert (== (nl_fs_file_size joined) 3)\n'
                ' assert (== other_parent "/different")\n'
                ' assert (== other_joined "/different/file")\n'
                ' assert (== (at all 0) "A.TXT")\n'
                ' return 0\n}\nshadow main { assert (== (main) 0) }\n')
            self.execute_both(source, directory)

    def test_wrong_declared_abi_preserves_output(self):
        cases = (
            ('extern fn nl_fs_is_directory(path: string) -> string', '(nl_fs_is_directory "x")'),
            ('extern fn nl_fs_list_files_ci(path: string) -> array<string>', '(nl_fs_list_files_ci "x")'),
            ('extern fn nl_fs_file_size(path: int) -> int', '(nl_fs_file_size 1)'),
        )
        with tempfile.TemporaryDirectory(prefix='canonical-fs-refusal-') as tmp:
            directory = Path(tmp)
            source, output = directory / 'main.nano', directory / 'prior.nvm'
            for declaration, call in cases:
                with self.subTest(declaration=declaration):
                    source.write_text(declaration + '\nfn main() -> int { ' + call + ' return 0 }\nshadow main { assert true }\n')
                    output.write_bytes(b'prior')
                    result = self.command([COMPILER, source, '--emit-nvm', '-o', output], 1)
                    self.assertIn(b'artifact', result.stdout + result.stderr)
                    self.assertEqual(output.read_bytes(), b'prior')


if __name__ == '__main__':
    unittest.main()
