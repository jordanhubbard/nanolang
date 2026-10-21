"""Immutable borrowed-state preparation without selected execution."""
import unittest
from tests import test_record_map_runtime as runtime

class RecordMapPublicPrepare(runtime.RecordMapRuntime):
    harness_source = 'tests/nanoisa/test_record_map_public_prepare.c'
    result_banner = b'borrowed-state preparation checks; execution remains refused'

if __name__ == '__main__':
    unittest.main()
