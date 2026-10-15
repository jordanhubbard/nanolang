"""I retain the paired runtime assertions for resource-bearing selected unions."""
from tests.test_owned_runtime import OwnedRuntime as _OwnedRuntime

class OwnedUnionRuntime(_OwnedRuntime):
    executable = "test_owned_union_runtime"
    case_count = 18
    max_live_records = 4
    refusal_count = 17
    noninteger_cases = set()

del _OwnedRuntime
