"""I preserve inferred local types through the checked canonical frontend."""
from pathlib import Path
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]


class CanonicalLocalInference(unittest.TestCase):
    def checked(self, *args):
        result = subprocess.run([str(x) for x in args], cwd=ROOT,
                                capture_output=True, text=True, timeout=120)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        return result.stdout

    def test_full_inferred_fixture_publishes_and_executes_from_both_stages(self):
        fixture = ROOT/'tests/nanoisa/fixtures/inferred_locals.nano'
        with tempfile.TemporaryDirectory(prefix='nano-inferred-canonical-') as tmp:
            directory = Path(tmp)
            for stage in ('nanoc_stage1', 'nanoc_stage2'):
                with self.subTest(stage=stage):
                    module, generated, native = (directory/(stage+suffix) for suffix in ('.nvm', '.c', ''))
                    self.checked(ROOT/'bin'/stage, fixture, '--emit-nvm', '-o', module)
                    self.checked(ROOT/'bin/nano_vm', '--verify-only', module)
                    self.assertEqual(self.checked(ROOT/'bin/nano_vm', module), 'once\n7\n')
                    self.checked(ROOT/'bin/nvm2c', module, '-o', generated)
                    self.checked('cc', '-std=c11', '-Wall', '-Wextra', '-Werror', generated, '-lm', '-o', native)
                    self.assertEqual(self.checked(native), 'once\n7\n')

    def test_map_value_inference_and_wrong_key_refusal(self):
        source = '''fn main() -> int {
 let values: HashMap<string,int> = (map_new)
 (map_put values "key" 7)
 let fetched = (map_get values "key")
 assert (== fetched 7)
 return 0
}
shadow main { assert (== (main) 0) }
'''
        with tempfile.TemporaryDirectory(prefix='nano-map-inferred-') as tmp:
            directory = Path(tmp)
            path, output = directory/'input.nano', directory/'output.nvm'
            for stage in ('nanoc_stage1', 'nanoc_stage2'):
                with self.subTest(stage=stage):
                    path.write_text(source)
                    self.checked(ROOT/'bin'/stage, path, '--emit-nvm', '-o', output)
                    self.checked(ROOT/'bin/nano_vm', output)
                    original = output.read_bytes()
                    path.write_text(source.replace('(map_get values "key")', '(map_get values 3)'))
                    refused = subprocess.run([ROOT/'bin'/stage, path, '--emit-nvm', '-o', output],
                                             cwd=ROOT, capture_output=True, text=True, timeout=120)
                    self.assertNotEqual(refused.returncode, 0, refused.stdout + refused.stderr)
                    self.assertIn('declared map key type', refused.stdout + refused.stderr)
                    self.assertEqual(output.read_bytes(), original)


    def test_missing_enum_member_refuses_publication(self):
        with tempfile.TemporaryDirectory(prefix='nano-enum-inferred-') as tmp:
            source, output = Path(tmp)/'input.nano', Path(tmp)/'output.nvm'
            source.write_text('enum Mode { Low = -3 }\nfn main() -> int { let value = Mode.Missing return 0 }\nshadow main { assert true }\n')
            for stage in ('nanoc_stage1', 'nanoc_stage2'):
                with self.subTest(stage=stage):
                    output.write_text('previous artifact')
                    result = subprocess.run([ROOT/'bin'/stage, source, '--emit-nvm', '-o', output],
                                            cwd=ROOT, capture_output=True, text=True, timeout=120)
                    self.assertNotEqual(result.returncode, 0, result.stdout+result.stderr)
                    self.assertIn('member in the declared enum', result.stdout+result.stderr)
                    self.assertEqual(output.read_text(), 'previous artifact')


if __name__ == '__main__':
    unittest.main()
