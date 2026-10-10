# I execute checked indirect File calls in two adapters

I continue #989 from my [indirect carrier](NANOISA_FILE_INDIRECT_RUNTIME.md).
My source-private VM adapter and C11 emitter accept the distinct indirect
hosted plan. They do not convert it to an old cyclic plan or alter ordinary
NanoVM admission. Both remain behind explicit private provider definitions.

## My common execution contract

I check opcode coverage for every instruction, including unreachable bodies,
then every retained variant, successor, cleanup obligation and candidate set.
Each indirect candidate must be in the prepared function table and agree with
the call shape. The carrier checks the selected target's complete parameter
and result declarations and membership before any argument transfer.

My VM adapter handles `FUNCREF` through the exact-site carrier constructor and
`CALL_INDIRECT` through checked frame entry. It retains the existing File
service, reference, region, scalar and aggregate operations. Each instruction
charges the shared invocation budget before effects. Initializer and entry
share that budget and publish only a clean scalar terminal.

## My generated native calls

I emit a real C function for each original function and a label for each
instruction. A direct call names its generated callee. An indirect call first
enters the carrier's checked child frame, then switches on that frame's original
function index. Only candidates from the prepared call site have cases; each
case calls its corresponding generated C function. Its entry checks the exact
function and instruction identity before executing effects. I generate no
opcode interpreter or FFI fallback.

Before host acquisition, generated code compares its complete embedded module
bytes and copied startup, function, local, instruction, variant, reference,
region and catalog facts with the new owning plan. For every indirect variant
I also compare original function/PC, both candidate sets and the complete common
call obligation. The generated-native ABI checks revision and all exposed C
layout sizes. Emission is bounded and failure preserves the caller's output.

## My paired qualification

My fixture captures exact serialized inputs and VM traces, emits their C,
compiles it at O0/O2 and replays the same corpus. Traces compare status, original
failure site, fuel, result, File open/read/write/seek/close operations and cleanup
details. I also link generated programs with only the query/core closure and
inspect symbols for VM execution dependencies.

The corpus retains previous loops, direct borrows, ownership transfers and
failure controls. Indirect cases select both targets, change targets across
iterations, retain caller operands across calls, nest direct calls inside an
indirect callee, pass/return File and OpenResult owners, and exercise initializer
fuel, denial, assertion failure and cleanup errors. Candidate-fact and generated
target mutations test startup and function-entry refusals separately.

This private execution boundary does not publish indirect File source or an
installed public grant route. Callable parameters/results, indirect borrowed
formals, paired C/Nano source lowering and full mandatory shadows remain required
within 5.1, along with Linux/Darwin release qualification.
