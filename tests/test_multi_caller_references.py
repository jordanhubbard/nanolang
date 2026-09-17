"""I preserve actual caller aliases across bounded parameter batches."""
from tests import test_owned_runtime

class MultiCallerReferences(test_owned_runtime.OwnedRuntime):
    executable = "test_multi_caller_references"
    case_count = 12
    refusal_count = 13
    noninteger_cases = set()
    max_live_records = 3
    case_live_records = {6: 8}
