"""I preserve scalar effects, selected transfers and allocation cleanup."""
from tests import test_owned_runtime, test_affine_scalar_union_runtime


class OwnedScalarGlobalRuntime(test_owned_runtime.OwnedRuntime):
    compiler = test_affine_scalar_union_runtime.AffineScalarUnionRuntime.compiler
    executable = "test_owned_scalar_global_runtime"
    case_count = 9
    max_live_records = 4
    refusal_count = 9
    noninteger_cases = set()
    assertion_cases = {6}
