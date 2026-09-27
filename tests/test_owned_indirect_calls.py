"""I transfer resource arguments and results through checked callback targets."""
from . import test_owned_value_graphs as graphs


class OwnedIndirectCalls(graphs.OwnedValueGraphs):
    binary_environment = 'NANO_OWNED_INDIRECT_TEST'
    binary_default = graphs.ROOT/'obj/test_owned_indirect_calls'
    case_count = 66
