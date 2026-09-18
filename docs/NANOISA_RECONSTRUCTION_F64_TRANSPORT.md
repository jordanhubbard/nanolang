# My binary64 constant reconstruction contract

I record `task_b6083f68f2cf4951a134da4ec82a1e17` before implementation,
under full reconstruction parent `task_4bd034f6029b7458201db74e2c3aeb32`.
My starting main is `cc0688f1`, containing reviewed PR720. My original contract commit `9a8ab1c2` recorded this proposal before admission.
My implementation and acceptance are retained in [evidence](evidence/reconstruction-f64-transport.md);
all exclusions below remain in force.

My later [typed comparison contract](NANOISA_RECONSTRUCTION_F64_COMPARISONS.md)
records a separate continuation; the exclusions below describe this original pin.

## My admitted operations

I add only `PUSH_F64`, `F64_FROM_BITS` and `F64_TO_BITS` to my reconstruction
analyzer. I admit exact FLOAT parameters/results for ordinary helper functions,
plus FLOAT locals, immutable temporaries and existing stack permutations. My
entry stays zero-argument INT. Direct calls retain the existing acyclic graph,
explicit returns and exact parameter/result checks.

I require `f64_bits` to be exactly sixteen ASCII hexadecimal digits and decode
it with integer operations. I never use the old `arg` placeholder, a Python/JSON
float, decimal text, a numeric cast or the NaN-quieting string parser to recover
an operand. Missing or invalid facts cause refusal before output publication.
My module facts reader still loads and verifies the module before reporting
facts; broadening its scalar signature gate is limited to exact FLOAT.

For C I emit private, standalone representation helpers using `memcpy` and
storage-size assertions. Constants use an exact uint64 hexadecimal pattern.
Signed INT to bits uses defined modulo conversion; bits to signed INT uses the
representable negative reconstruction already qualified by PR720. No pointer
punning, floating arithmetic or implementation-defined signed narrowing is
needed. Generated C has no repository-header or runtime-library dependency.

For Nano I use the merged `float_from_bits` and `float_to_bits` intrinsics.
I express each constant's pattern as the existing exact signed INT expression,
including the existing representable expression for INT64_MIN. I retain FLOAT
annotations on generated functions, parameters, locals and temporaries; I do
not route unknown tags through the emitter's existing bool fallback. Any type
outside the explicit INT/BOOL/FLOAT map is refused.

Every binary64 pattern belongs to this representation contract: both zeros,
finite/subnormal encodings, infinities, signed quiet NaNs and signed signaling
NaNs with their payloads. I do not define new floating arithmetic behavior.

## My state and control-flow invariants

My existing non-pure expression path emits an immutable temporary at every
instruction. Constants, conversions, loads and calls continue through that path.
DUP/PICK/permutations reuse those snapshots; later stores cannot change them,
and discarded values do not erase an already emitted operand call.

I preserve one exact type per local slot and definite initialization. Both
sides of a branch may store FLOAT into the same FLOAT local; neither INT/FLOAT
nor BOOL/FLOAT joins are admitted. Structured branch/loop boundaries still
require empty operand stacks. This slice does not introduce general stack phi
nodes or broaden boxed joins. Pretest loops retain existing pure-condition and
backedge restrictions. Exact bit conversions may occur in a pure condition;
user calls and stores remain refused there.

I audit every existing operator handler after adding FLOAT transport. Existing
INT/BOOL arithmetic, casts, comparison and truthiness handlers must keep their
explicit old tag checks. In particular, I do not accidentally admit float
operands to generic arithmetic/comparison, CAST_INT/CAST_BOOL, boolean operators
or typed I64 operations. F64 arithmetic, comparisons, negation and numeric
CAST_FLOAT remain separate future work.

## My acceptance order

1. I add analyzer/emitter and facts-signature controls first. I retain atomic
   output publication, input/output separation and all current size, expression,
   region, call, metadata, import, initializer and heap exclusions.
2. I use small fresh ordinary verified modules. Exact integer bit observers cover
   signed zeros, finite/subnormal endpoints, infinities and signed quiet/signaling
   NaNs. I exercise constants, inverse conversions, local stores, snapshot-before-
   mutation, helper arguments/returns, both local-based branch arms and a bounded
   loop. I retain one-evaluation checks through admitted calls.
3. I compare original VM/native results with standalone reconstructed C under
   GCC and Clang with ASan/UBSan. I compile reconstructed Nano with qualified
   C-seed, Stage1 and Stage2 tools containing PR720; I also emit canonical modules
   through both producer implementations and execute their VM/native paths.
   I compare exact integer observations, not float equality or formatted output.
4. I record compiler paths, production revisions and hashes before/after gates.
   Reused qualified source tools are identified separately from the generator
   source under test. If relocation is needed, I retain imported-library paths
   and hashes and do not claim hermetic relocation.
5. I retain ordinary refusals and previous outputs for unsupported float
   operations, mixed-tag locals/joins, unsupported entry signatures and existing
   structural limits. I update old blanket float-refusal tests only where this
   narrow admission now applies, replacing them with still-unsupported controls.

I do not replay historical PR679 compiler failures, regenerate its endpoint
corpus or modify a frozen product acceptance tree. I preserve the first failure
of any fresh gate and qualify corrections separately. This child cannot close
full reconstruction, float arithmetic or broader source/backend equivalence.
