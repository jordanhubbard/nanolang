# I retain one exact aggregate callback target

I can translate an indirect call returning an ordinary aggregate or array when
my closed-module inference establishes one exact same-module function target.
I carry that identity from `FUNCREF` through stack copies, locals, direct-call
parameters, direct and tail-call results, returns and forward-only joins.
Pending identity is distinct from mixed or unproved identity. I wait for target
facts instead of borrowing an unrelated scalar function's result type.

For this profile, the calling function must have no backward control-flow edge.
I validate the target's arity, result count and available argument tags, then
reuse my direct-call argument/result shape propagation. The native output
checks the evaluated callable's identity before invoking that target. I retain
the selector's effects and argument evaluation order. My existing direct-call
root and cleanup machinery owns aggregate results; I add no alternative heap
representation or VM wrapper.

This is not general target-set inference. Multiple targets, callback captures,
backedge callback analysis and resource ownership transfer are not established
by this contract. Existing scalar indirect dispatch keeps its separate boundary.
The broader callback and PR522 acceptance work remains open.

I also emit explicit `(void)` parameter lists for zero-argument C language
functions and their declarations, including generic instances and imported
prototypes. This makes the declarations and definitions match their callback
typedefs under the tested Clang function-type sanitizer. I retain the original
C-seed generic-result failures and the corrected generated C.

## My evidence

My [qualification record](evidence/pr522-implementation-2026-09-23/exact-aggregate-callbacks/README.md)
retains the original generic suite, the broader source sanitizer run, native
translator gates, declaration-order permutations and bootstrap state. Passing
these checks does not establish complete compiler equivalence or release
readiness.
