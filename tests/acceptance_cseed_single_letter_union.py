"""I retain task_c117b20ef5d64339b79cf22e86b42ef7's original positive control.

The naming repair includes this formerly failing case in its normal gate.
"""
import unittest
from tests import test_cseed_imported_unions as fixtures

MODULE = fixtures.MODULE

class SingleLetterUnionAcceptance(unittest.TestCase):
    check = fixtures.CseedImportedUnions.check

    def test_declared_single_letter_union(self):
        self.check('fn main() -> int { let value: T = (Provider.make) assert (== (Provider.read value) 11) return 0 }', module=MODULE.replace('Choice', 'T'))

if __name__ == '__main__':
    unittest.main()
