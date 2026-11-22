"""I validate a real retained native profile and refuse corrupted producer data."""
import json
from pathlib import Path
import tempfile
import unittest

from scripts.ci_capture_assembler_profile import read_fixed_profile
from tests.retained_capture_evidence import fixed_profile


class CaptureAssemblerProfile(unittest.TestCase):
    def test_actual_native_archive_and_independent_drift_refusals(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fixed_profile(root)
            # I parse x86_64 ELF bytes as data on every host; I execute no image.
            accepted = read_fixed_profile(root, machine='x86_64')
            self.assertEqual(accepted['gas_cases'], 2095)
            self.assertEqual(accepted['assembler_sha256'],
                '7e9edb4b7fc0c7ce97fbcdd4ecb63e3eb1c1663e2452cd32ce0249110b253bf5')
            with self.assertRaisesRegex(RuntimeError, 'another native architecture'):
                read_fixed_profile(root, machine='aarch64')
            image = root / 'bin/as'
            original = image.read_bytes()
            image.write_bytes(original + b'changed')
            with self.assertRaisesRegex(RuntimeError, 'source or assembler bytes'):
                read_fixed_profile(root, machine='x86_64')
            image.write_bytes(original)
            summary = root / 'fixed-gas.sum'
            original = summary.read_bytes()
            summary.write_bytes(original.replace(b'PASS:', b'FAIL:', 1))
            with self.assertRaisesRegex(RuntimeError, 'complete passing native GAS outcomes'):
                read_fixed_profile(root, machine='x86_64')
            summary.write_bytes(original)
            metadata = root / 'build.json'
            original = metadata.read_bytes()
            changed = json.loads(original)
            changed['witnesses'] = changed['witnesses'][:-1]
            metadata.write_text(json.dumps(changed))
            with self.assertRaisesRegex(RuntimeError, 'eight original native'):
                read_fixed_profile(root, machine='x86_64')


if __name__ == '__main__':
    unittest.main()
