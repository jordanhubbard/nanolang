# My self-hosted named function-container checkpoint

I admit named function fields and array elements through my self-hosted NanoISA
producer. I preserve exact function signatures, infer literal element types,
retain contextual empty-array types and emit function tags for literals and
filled arrays. Function signatures carry a visited-signature path separately
from record-storage ancestors. A callback can refer to its enclosing record;
a record-storage cycle still fails the existing check.

I build the complete compiler with dependency shadows, translate its module to
native C, and use that prepared native compiler for 58 passing methods: 34
callable methods and 24 adjacent CLI/product methods. The retained source that
previously failed `unsupported local type Ops` now executes through both VM and
native products. Additional source cases cover nested records, aliases, global
function arrays, empty/filled arrays, returned elements, recursive callback
signatures and rejected record/array/function signatures with prior-output
preservation. The native callable controls retain ASan/UBSan/LSan checks.

My first new empty-function-array shadow fails because element-type lookup still
returns no type; I retain its terminal and repair that lookup. My first alias
fixture also incorrectly assigns the void result of `array_set`. I retain that
refusal and correct the fixture to use the mutation as a statement. These are
separate observations, not evidence that the typechecker should accept void.

The manifest binds commands, terminals, sources and prepared compiler artifacts.
This is source/product parity for named functions, not captured-closure parity,
a current raw bootstrap fixed point, complete platform qualification or release.
The earlier clean compiler gate remains qualification of its own pinned source.
