"""I check selected resource-union execution and native allocation cleanup."""
from tests import test_owned_runtime, test_affine_scalar_union_runtime

class OwnedUnionRuntime(test_owned_runtime.OwnedRuntime):
    compiler = test_affine_scalar_union_runtime.AffineScalarUnionRuntime.compiler
    executable = "test_owned_union_runtime"
    case_count = 11
    max_live_records = 3
    refusal_count = 5
    noninteger_cases = set()
    assertion_cases = {9}
