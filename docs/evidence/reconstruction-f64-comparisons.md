# My typed binary64 comparison reconstruction evidence

I tested production `b03adaa2` against base `55208c47`, following contract
`eb751dcc` and task `task_4b603437748343d48a5f92bf577bc5be`.
My change admits exactly six typed F64 comparisons with two FLOAT inputs and
BOOL output. I reuse existing relational emission and immutable snapshots.

I passed four focused methods with GCC in 81.492 seconds and Clang in 81.392
seconds. They cover 120 comparison cases: ten explicit ordering/unordered pairs,
both operand orders and all six operators. Cases include both zeros, positive
and negative subnormals, the normal boundary, finite extrema, infinities and
signed quiet/signaling NaNs. After each comparison I checked both original
operand bit patterns with integer observers. I inspected all three canonical
producer outputs for the actual typed F64 comparison opcode.

I also passed helper-call, snapshot-before-store, both branch outcomes and
bounded pure-loop controls. Test-only counters verify exactly two helper calls
through standalone C and every reconstructed Nano producer route. My counters
do not admit global opcodes into reconstruction. Other-tag analyzer controls
reject both operand positions for INT, BOOL, void and non-scalar tags.
Unsupported float arithmetic, numeric casts, truthiness and generic comparison
retain previous outputs on refusal. Two adjacent methods passed in 0.240
seconds, including 94 existing operator refusal controls and mixed-local,
entry-result and nonempty-stack-join refusal controls.

Each ordinary positive module ran in my current VM and native translator.
Standalone reconstructed C and generated native C ran with GCC/Clang O2,
Wall/Wextra/Werror and ASan/UBSan with nonrecovering sanitizer errors.
Reconstructed Nano ran through qualified PR720 C-seed, Stage1 and Stage2
legacy executable compilation, plus NanoVirt/Stage1/Stage2 canonical module
emission followed by VM/native execution. The Clang setting controls the
explicit C/native commands; I do not claim every legacy compiler-selected
subcommand has identical flags.

My source producer pin is `f87267c21a356424ce101060521ce841a5eb17bb` in
`/home/jkh/Src/nanolang-binary64-bit-transport`. Its four compiler binaries and
recorded imported library hashes matched before and after gates. My current
facts reader, assembler, VM and native translator are separate tools built in
this reconstruction worktree. I retain these identities and log hashes in
[my manifest](reconstruction-f64-comparisons.json). I do not claim a new
bootstrap or hermetic compiler relocation.

My GCC process loaded the test before a test-only import changed from a direct
class import to a module-qualified reference and an expected integer expression
was simplified from `-1+2` to `1`; its explicit class selection and all assertions
were unchanged. Clang tested the final test source `ae53ca46`, including module
discovery. Both used the identical production implementation.

I retained the build and three gate logs under
`/tmp/nanolang-reconstruction-f64-comparisons-{build,gcc,clang,adjacent}.log`.
Per-command inputs, modules, reconstructed sources and outputs remain in fresh
`/tmp/nano-reconstruct-f64-*` directories. No gate failed in this child.

I make no floating-environment promise beyond ordinary default execution.
Arithmetic, negation, numeric casts, float truthiness, generic FLOAT comparison,
heap reconstruction and the full reconstruction parent remain open. I did not
execute historical PR679 artifacts or change a frozen product tree.
