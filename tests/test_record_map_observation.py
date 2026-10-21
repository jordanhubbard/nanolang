"""Private allocation-free copied observation boundary, not public admission."""
import unittest
from tests import test_record_map_runtime as runtime

class RecordMapObservation(runtime.RecordMapRuntime):
    harness_source = 'tests/nanoisa/test_record_map_observation.c'
    result_banner = b'private copied map observation checks; public boundaries checked'

if __name__ == '__main__':
    unittest.main()
