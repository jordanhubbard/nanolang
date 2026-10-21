"""Read-only ordinary heap imports, without execution admission."""
import unittest
from tests import test_record_map_runtime as runtime

class RecordMapImport(runtime.RecordMapRuntime):
    harness_source = 'tests/nanoisa/test_record_map_import.c'
    result_banner = b'ordinary import query checks; public admission remains refused'

if __name__ == '__main__':
    unittest.main()
