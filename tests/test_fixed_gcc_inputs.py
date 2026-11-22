"""I refuse compiler-image drift before reusing a native assembler producer."""
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

from scripts import ci_fixed_gcc_inputs as inputs


class FixedGccInputs(unittest.TestCase):
    def test_native_image_identity_reader_refuses_a_changed_digest(self):
        # I exercise only file identity here, not GCC language/tool acceptance.
        actual = Path(sys.executable).resolve(strict=True)
        with tempfile.TemporaryDirectory() as directory:
            profile = Path(directory)
            (profile / 'build.json').write_text(json.dumps({'compiler': str(actual)}))
            with patch.object(inputs.shutil, 'which', return_value=str(actual)):
                digest = inputs.shared.digest(actual)
                self.assertEqual(inputs.require_compiler(profile, digest),
                                 {'path': str(actual), 'sha256': digest})
                with self.assertRaisesRegex(RuntimeError, 'unchanged actual qualified GCC'):
                    inputs.require_compiler(profile, '0' * 64)
                (profile / 'build.json').write_text(json.dumps({'compiler': str(actual) + '.changed'}))
                with self.assertRaisesRegex(RuntimeError, 'unchanged actual qualified GCC'):
                    inputs.require_compiler(profile, digest)

    def test_local_checkout_cannot_run_the_hosted_consumer_preparer(self):
        with patch.dict(os.environ, GITHUB_ACTIONS='false'):
            with self.assertRaisesRegex(RuntimeError, 'isolated GitHub Actions checkout'):
                inputs.main()


if __name__ == '__main__':
    unittest.main()
