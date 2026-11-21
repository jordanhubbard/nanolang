"""I translate my retained real Stage1 join counterexample without executing its host imports."""
from pathlib import Path
import gzip
import hashlib
import os
import tempfile
import unittest

from tests.nanolang_translator_routes import checked

ROOT = Path(__file__).resolve().parents[1]


class LateScalarJoins(unittest.TestCase):
    def test_actual_stage1_optional_carriers(self):
        fixture = ROOT / 'tests/nanoisa/fixtures/compiler_stage1_late_scalar_joins.nvm.gz'
        data = gzip.decompress(fixture.read_bytes())
        self.assertEqual(hashlib.sha256(data).hexdigest(),
                         'f3b0f758884a6baeec8052c1f946d3ff6bb9a7947fdbf3c308a719098e624d8f')
        with tempfile.TemporaryDirectory(prefix='native-late-scalar-joins-') as directory:
            module = Path(directory) / 'stage1.nvm'
            source = Path(directory) / 'stage1.c'
            module.write_bytes(data)
            source.write_bytes(b'I preserve this prior source until translation succeeds.')
            checked([ROOT / 'bin/nano_vm', '--verify-only', module], timeout=120)
            checked([os.environ.get('NANOLANG_NVM2C', ROOT / 'bin/nvm2c'),
                     module, '-o', source], build=True, timeout=600)
            self.assertGreater(source.stat().st_size, 1_000_000)
            self.assertIn(b'int main(', source.read_bytes())


if __name__ == '__main__':
    unittest.main()
