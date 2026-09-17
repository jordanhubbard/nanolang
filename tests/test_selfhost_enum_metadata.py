"""I keep enum parser metadata consistent across my bootstrap compilers."""
from pathlib import Path
import subprocess
import tempfile
import unittest
ROOT = Path(__file__).resolve().parents[1]

class EnumMetadata(unittest.TestCase):
    def test_shared_parser_values(self):
        with tempfile.TemporaryDirectory() as tmp:
            for name in ('nanoc_c', 'nanoc_stage1', 'nanoc_stage2'):
                with self.subTest(compiler=name):
                    output = Path(tmp) / name
                    for command in ([ROOT / 'bin' / name,
                                     ROOT / 'tests/selfhost/test_enum_metadata.nano', '-o', output], [output]):
                        result = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, timeout=180)
                        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

if __name__ == '__main__':
    unittest.main()
