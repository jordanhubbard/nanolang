# My zero-result NanoISA functions

I emit `void 0` function headers, bare/implicit `RET`, zero-result direct and
tail calls, and statement calls without a spurious `POP`. I reject a bare
return in a value-result function and mismatched value/void return expressions.
List-push lowering retains its existing emitted stack effect and trailing POP.

On Linux ARM64, based on `6ca8f5f7`, `make -j8 test-nanoisa-src-nano` passes
86 baseline checks and four Python cases. The new void case compares 14
module/function checks against the C seed: empty and printing functions,
bare return, conditional early return, tail forwarding and their caller.
Both emitted modules verify and execute in NanoVM and through `nvm2c` plus
strict C11 compilation, producing exactly `usage\ncontinued\n`. Four invalid
return-count forms are refused without publishing assembly. Flat-record
execution and nested-record refusal remain green.

I reran emission of my `nanoisa_emit.nano` driver. `show_usage` now lowers;
my first rejection becomes `undefined function nanoisa_emit_nasm`. The driver
still reads a single raw source file without merging imported definitions.
This is the existing import-closure milestone, not complete self-hosting.
The full `nanoc_v06.nano` input also still has the separately tracked scalar
expression typing gap.

Task `task_862889b9369e4511a465fe838e69224e` records this bounded slice.
