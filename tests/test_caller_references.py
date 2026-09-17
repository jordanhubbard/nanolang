"""I require actual caller storage through the VM and native helper."""
from tests import test_owned_runtime

class CallerReferences(test_owned_runtime.OwnedRuntime):
    executable = "test_caller_references"
    case_count = 9
    refusal_count = 12
    noninteger_cases = set()
    max_live_records = 3
