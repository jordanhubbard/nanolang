"""I execute declared enum and string reduce values through each selected route."""
import unittest
from tests import test_cseed_generic_functions as base


class ReduceNominalValues(unittest.TestCase):
    run_command = base.CseedGenericFunctions.run_command
    execute = base.CseedGenericFunctions.execute

    def test_enum_parameter_and_array_storage(self):
        self.execute('''enum Choice { One, Two }
fn fold(a:Choice,b:Choice)->Choice{return a}
shadow fold {assert (== (fold Choice.One Choice.Two) Choice.One)}
fn apply(xs:array<Choice>,initial:Choice)->Choice{return (reduce xs initial fold)}
shadow apply {assert (== (apply [Choice.One,Choice.Two] Choice.Two) Choice.Two)}
fn main()->int {
    assert (== (apply [Choice.One,Choice.Two] Choice.Two) Choice.Two)
    assert (== (apply [] Choice.One) Choice.One)
    return 0
}
shadow main {assert (== (main) 0)}
''')

    def test_string_literal_accumulator(self):
        self.execute('''fn join(a:string,b:string)->string{return (+ a b)}
shadow join {assert (== (join "x" "a") "xa")}
fn main()->int {
    assert (== (reduce ["a","b"] "x" join) "xab")
    let empty:array<string> = []
    assert (== (reduce empty "kept" join) "kept")
    return 0
}
shadow main {assert (== (main) 0)}
''')
