"""I require matching owner-backed VM/native reference behavior and cleanup."""
from tests import test_owned_runtime


class SameFrameReferences(test_owned_runtime.OwnedRuntime):
    executable = "test_same_frame_references"
    case_count = 9
    refusal_count = 16
    noninteger_cases = {6, 7}
