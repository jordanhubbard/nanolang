"""Selected DELETE execution, history, tombstone imports and failure recovery."""
import os
import unittest
from unittest.mock import patch
from tests import test_record_map_runtime as runtime


class RecordMapDelete(runtime.RecordMapRuntime):
    harness_source = 'tests/nanoisa/test_record_map_delete.c'
    result_banner = b'selected DELETE consumer assertions'

    def runtime(self, computed_goto):
        # These inputs already carry independently bound Holder map authority;
        # the older runtime selector accepts unannotated CollectResult modules.
        with patch.dict(os.environ, {'RECORD_MAP_MODULES': os.environ.get('RECORD_MAP_DELETE_MODULES', '')}):
            super().runtime(computed_goto)


if __name__ == '__main__':
    unittest.main()
