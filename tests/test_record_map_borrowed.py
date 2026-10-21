"""Private borrowed-state invocation, never public selector admission."""
import unittest
from tests import test_record_map_runtime as runtime

class RecordMapBorrowed(runtime.RecordMapRuntime):
    harness_source = 'tests/nanoisa/test_record_map_borrowed.c'
    result_banner = b'borrowed map execution assertions; public boundaries checked'

if __name__ == '__main__':
    unittest.main()
