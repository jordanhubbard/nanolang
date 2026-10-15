"""I keep resource-bearing payloads outside my ordinary union storage profile."""
import subprocess
import tempfile
import unittest
from pathlib import Path
from tests.test_selfhost_layout_validation import ROOT, COMPILER


class SelfhostUnionStorage(unittest.TestCase):
    def test_resource_payloads_preserve_prior_artifacts(self):
        cases = [
            ('Handle', 'Handle {value:7}'),
            ('array<Handle>', '[Handle {value:7}]'),
            ('List<Handle>', '(list_Handle_new)'),
            ('Wrapper', 'Wrapper {opaque_value:Handle {value:7}}'),
            ('Box<Handle>', 'Box<Handle>.Some {value:Handle {value:7}}'),
        ]
        for payload, value in cases:
            for mode in ([], ['--emit-nvm']):
                with self.subTest(payload=payload, mode=mode), tempfile.TemporaryDirectory() as tmp:
                    source, output = Path(tmp)/'main.nano', Path(tmp)/'output'
                    source.write_text('resource struct Handle {value:int}\nstruct Wrapper {opaque_value:Handle}\nunion Box<T> {Some {value:T}}\n'
                        f'fn hold(value:Box<{payload}>)->int {{return 0}}\nshadow hold {{assert true}}\n'
                        f'fn main()->int {{let value:Box<{payload}> = Box<{payload}>.Some {{value:{value}}} return (hold value)}}\n'
                        'shadow main {assert (== (main) 0)}\n')
                    output.write_bytes(b'prior artifact')
                    result = subprocess.run([str(COMPILER), str(source), *mode, '-o', str(output)],
                                            cwd=ROOT, capture_output=True, text=True, timeout=120)
                    self.assertNotEqual(result.returncode, 0, result.stdout+result.stderr)
                    self.assertIn('[E0036]', result.stdout+result.stderr)
                    self.assertEqual(output.read_bytes(), b'prior artifact')
