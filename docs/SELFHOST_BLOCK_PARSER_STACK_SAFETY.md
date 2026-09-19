# I parse long statement lists without consuming one C stack frame per statement

I record this defensive parser contract under
`task_a613986ffa6f476293e3befa8d9accfd`, discovered by the corrected affine
route matrix under `task_bd8ebe2ba75457af1686f095275229dd` and installed
product task `task_e8d860a16da0464891dd32e91c42bef1`. I have not replayed or
minimized the failed compiler invocation. This contract follows static source
inspection and the operating system's retained crash report.

## What failed

The unchanged `owner_257` source contains 514 sequential statements in one
function block: 256 owner declarations, 256 one-statement unsafe blocks, one
remaining owner declaration and one return. Its SHA-256 is
`86f1b89e0dbcf662f0d9da8ac41e9abca3e987e0f285fe14c69fbcb4b5968909`.
The copied Stage1 compiler has SHA-256
`b3ac51df6bd3adc0014700c29a9572d80ce2fcab60c41f14d7147809c0ffce03`.
It terminated by signal 11 after 0.0563 seconds, emitted no stdout or stderr,
and preserved the known 14-byte prior output exactly.

macOS retained
`nanoc_stage1-2026-09-18-213056.ips`, 10,898 bytes, SHA-256
`d0f5a4746eaf8d3423af39a36e973fcb1e9b9e1ca96fcfbf61473396e4cd9597`.
The report says `Thread stack size exceeded due to excessive recursion` at the
stack guard. Its main-thread trace contains 303 recursive
`parser__parse_block_recursive` frames, with the next allocation faulting in
`___chkstk_darwin`. The intervening frames parse the next unsafe block and its
call expression. The report's process launch and capture timestamps match the
sealed `owner_257` observation.

The source explains that trace. `parse_block_recursive` calls itself after
every successful ordinary statement, tuple destructure or owned pattern.
`Parser` is passed and returned by value, so every sequential statement retains
another large native frame until the closing brace unwinds the whole chain.
`parse_unsafe_block_recursive` has the same linear-recursion structure. The
current fixture reaches only one statement in each unsafe block, so that twin
is a latent instance of the same defect rather than the observed deepest
chain.

I also inspected the ownership-state and dynamic-array paths. Record-array
pushes snapshot borrowed values before growth, array length and capacity are
64-bit, and this run is below the GC threshold. Those paths do not explain the
retained stack-overflow trace. I do not change them under this task.

## My bounded correction

I will replace linear statement-list recursion in exactly these two parser
helpers with explicit loops over a mutable `Parser` cursor:

- `parse_block_recursive`
- `parse_unsafe_block_recursive`

I retain their existing signatures for this bounded repair. Each loop must
preserve these rules:

1. An existing parser error returns immediately.
2. End of input before `}` returns the existing parser error state.
3. `}` is consumed once, then the accumulated source-ordered statement list is
   stored once with its original opening location.
4. An owned pattern may append its synthetic owner and projection statements;
   the returned parser cursor becomes the next iteration's input.
5. Ordinary blocks retain tuple-destructure detection, mutability and source
   locations exactly. Unsafe blocks retain their current statement grammar.
6. An ordinary parsed statement is appended exactly once using its returned
   node id and node type.
7. A parser error never publishes a partial block.

This removes statement-count-dependent native stack growth. It does not make
arbitrarily nested syntax non-recursive, change expression parsing, add an
ownership capacity, change diagnostics, change AST ordering or weaken the
exact `owner_257` negative.

## Qualification after review

I will not execute a correction before this contract and its production
checkpoint are reviewed. After review I require, in order:

1. parser controls for a long ordinary block and a long unsafe block, plus
   owned-pattern and tuple-destructure ordering;
2. missing-closing-brace and malformed-statement controls that retain checked
   parser failure and do not publish output;
3. fresh bootstrap and installed Stage1/Stage2 parser controls;
4. the complete frozen 36-source affine route matrix, where `owner_257` is
   rejected by ownership checking in both self-hosted explicit-C routes and
   every nonzero route preserves the known prior output;
5. the independently required deeper public-C, canonical-selection and affine
   gates already ordered by the parent contract.

I will stop at the first new terminal and seal it. I will not execute the
historical crashing binary or treat a larger host stack as a repair.
