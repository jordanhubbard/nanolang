"""I require matched nested owner paths and reborrow lifetimes."""
from tests import test_owned_runtime


class NestedReferences(test_owned_runtime.OwnedRuntime):
    executable = "test_nested_references"
    case_count = 11
    refusal_count = 16
    noninteger_cases = set()
    max_live_records = 4
    case_live_records = {9: 5, 10: 33}
