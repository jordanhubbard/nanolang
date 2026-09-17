# My loop-control targets

I save and restore enclosing loop targets while lowering nested `while` bodies.
`break` jumps to the current loop's end; `continue` jumps to its condition.
Both terminate their statement block, so I omit following unreachable code and
unneeded backedges. I omit the condition/false branch for literal `while true`,
matching the C seed. I clear loop context at each function and refuse controls
outside a loop.

On Linux ARM64, `make -j8 test-nanoisa-src-nano` passes 86 baseline checks and
eight focused cases. The new loop case compares 12 module/function checks for
nested targets, restoration of outer targets, conditional/unconditional loops,
condition re-evaluation, and branches whose arms break or continue. Assertions
after terminating controls cannot execute. Both C-seed and self-hosted modules
verify and run in NanoVM and through strict C11 AOT. Break and continue outside
a loop remain refused even after a previous function used a loop.

I reran emission of `src_nano/nanoc_v06.nano`. `parse_options` now passes its
loop-control boundary. The first rejection becomes `undefined function getenv`;
this builtin needs host-call lowering and an import contract. Full compiler
emission and canonical bootstrap equality remain open.

Task `task_942fb307760a4460a49eb49986049103` records this slice.
