# Self-hosted byte-array emitter checkpoint

I admit byte-array types in local, signature and record-field contexts, retain
the U8 element tag in contextual literals, and accept integer array writes with
packed modulo-256 storage. I keep scalar byte-literal range checks separate.
I widen byte operands before typed integer arithmetic/comparison instructions
and byte-valued inline returns before an integer return. I keep VM tag guards.

My retained pre-correction component driver refuses all six contextual cases
and the mutation/slice case. Initial byte storage reaches execution but fails
seven subcases on missing integer-operation/return widening. I retain both
failures. The final component driver is built by the C seed from the actual
self-hosted emitter; this is not a fresh installed Stage1/Stage2 qualification.

All three byte emitter methods pass in 0.331 seconds: six contextual cases,
mutation/slicing with independent outer storage, byte arithmetic/comparisons,
and wrong-kind refusals. The shared C/native-VM contextual suite still passes
both methods in 7.792 seconds.

The complete six-method slice suite retains four failures: native translation
rejects U8 literals from both producers and nested literals from the C seed;
self-hosted nested-array lowering still refuses. Every other slice case passes.
I retain these failures rather than removing cases or retiring the legacy gate.
Native byte/nested storage, self-hosted nesting and fresh compiler qualification
remain required under issue #979.
