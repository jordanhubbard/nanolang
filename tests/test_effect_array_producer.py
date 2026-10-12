"""I preserve prior artifacts while collection execution remains unqualified."""
from pathlib import Path
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
FIXTURES = ROOT / 'docs/evidence/effect-collections-20261012'


class EffectArrayProducer(unittest.TestCase):
    def check_refusal(self, name, diagnostic):
        with tempfile.TemporaryDirectory(prefix='nano-array-producer-') as directory:
            output = Path(directory) / 'prior.nvm'
            output.write_bytes(b'prior artifact')
            result = subprocess.run(
                [ROOT / 'bin/nano_virt', FIXTURES / (name + '.nano'),
                 '--emit-nvm', '-o', output], cwd=ROOT,
                capture_output=True, text=True, timeout=120)
            self.assertEqual(result.returncode, 1, result.stdout + result.stderr)
            self.assertIn(diagnostic, result.stdout + result.stderr)
            self.assertEqual(output.read_bytes(), b'prior artifact')

    def test_runtime_boundary(self):
        for name in ('scalar', 'lexical'):
            with self.subTest(case=name):
                self.check_refusal(name, 'matched collection runtime cleanup')

    def test_wrong_element(self):
        self.check_refusal('wrong_element', "declared operation's argument type")
