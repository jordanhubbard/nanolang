# My typed local-clear correction

A C-seed insertion stages a record in a temporary, then clears that temporary
with PUSH_VOID/STORE_LOCAL after its final use. My classifier previously demanded
optional record storage even when no later instruction could observe the void.

I analyze reachable successors after a literal-void store. A read before another
store prevents this optimization. I follow both conditional edges and backedges;
I stop at replacement stores and function exits. I reset the literal marker at
branch targets so an incoming record cannot be mistaken for fallthrough void.
For an unobserved clear, I retain the existing storage representation and emit a
zero value to clear its runtime roots. I do not erase the runtime cleanup.
Allocation failure declines the optimization.

My complete four-method insertion suite passes in 3.377 seconds. Two control-flow
methods pass three VM/sanitized-native programs and four observable-clear
refusals preserving prior output. The initial control-flow harness used `true`
instead of the assembler's numeric boolean operand; I retain that harness failure
and the corrected terminal. The original parser corpus remains unchanged.
Full native regression and final fresh compiler qualification are separate gates.

`make -j2 test-nvm2c CC=/opt/homebrew/opt/llvm/bin/clang` exits zero.
My 2,431 execution, 3,092 shape and 379 callable checks pass, together with
opcode coverage, sanitizer-driver checks and the new control-flow gate.
