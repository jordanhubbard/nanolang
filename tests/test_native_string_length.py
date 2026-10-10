"""I retain signed string lengths in comparisons and arithmetic."""
import subprocess
import tempfile
import unittest
from pathlib import Path
from tests import test_cseed_generic_functions as shared

SOURCE = '''fn main()->int {
    let index:int = -1
    assert (< index (str_length "abc"))
    assert (> (str_length "abc") index)
    assert (== (- (str_length "") 1) -1)
    assert (< (- (str_length "") 1) 0)
    assert (== (str_length "abc") 3)
    return 0
}
shadow main {assert (== (main) 0)}
'''


class SignedStringLength(unittest.TestCase):
    def test_native_comparisons_and_arithmetic(self):
        with tempfile.TemporaryDirectory(prefix='nano-signed-length-') as tmp:
            source, binary = Path(tmp)/'main.nano', Path(tmp)/'main'
            source.write_text(SOURCE)
            for args in ([shared.ROOT/'bin/nanoc_c', source, '-o', binary], [binary]):
                result = subprocess.run(list(map(str, args)), cwd=shared.ROOT,
                                        capture_output=True, text=True, timeout=120)
                self.assertEqual(result.returncode, 0, result.stdout+result.stderr)

    def test_vm_and_c_aot_comparisons_and_arithmetic(self):
        shared.CseedGenericFunctions().execute(SOURCE)
