# My native filled-array evaluation order

I previously emitted the `array_new` fill expression inside the generated
loop. Positive counts repeated its effects; zero and negative counts omitted
them. A three-element fixture failed its once-only effect assertion before
this repair (`/tmp/native-array-order-before.log`).

I now reuse my ordinary ordered-argument capture helper for count then fill.
Both operands run exactly once before the nonnegative count guard and before
array allocation. The helper chooses names outside source bindings and nested
calls. My generated loop uses the captured fill, including record-valued calls;
record copies use the captured value's size.

`make test-array-new-evaluation` passes three methods:

- Native C and interpreter agree on positive/zero count effects, source order,
  nested construction and local names resembling compiler temporaries.
- Native string, boolean, float, returned-record and nested-array fills retain
  their values.
- A negative count aborts with its count diagnostic after both operands run.
  The fill checks the count's effect and writes a file before the guard.

`make test-native-call-argument-order` also passes all four existing methods.

The focused log is `/tmp/nanolang-native-filled-test-final.log`. This changes
only the C-seed native constructor; the separate bytecode count guard is tracked
by `task_50c84fa002c14b3fa880d3d260bb8a33`. Full frontend/backend parity and
bytecode bootstrap remain separate acceptance gates.
