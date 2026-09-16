# Main through PR #317

I reconcile `4936f8cf` and `710a1cdb` with my tested integration checkpoint.
I preserve my production parser, match checker and native frame implementation.
Their contents remain unchanged from `9d0b3186`; I import the new regression
intent and both commits' ancestry, not their conflicting implementations.

## Parser and return contract

My parser already binds bare unary operators before infix parsing and uses
`parenthesized_call_start` to distinguish calls, grouped operators and tuples.
I import `test_parenthesized_infix` and invoke it from the fixture's main
function as well as its shadow.

My match checker excludes arms that always return from match-value type
unification. I retain that rule and the final-expression block value. I do not
derive a match value from a return statement. Return still exits the enclosing
function, as my creator specified for 5.0.

I extend the match contract suite to my C seed, Stage 1, Stage 2 and NanoVM.
It checks values, side effects, conditional and unconditional function returns,
strings and grouping. Negative cases reject wrong return/arm types before
native C compilation and preserve an existing output artifact.

## Record frames

PR #317 writes record counts as zero-padded three-digit C literals. `[008]`
does not compile; `[010]` means eight, not ten. I retain my existing decimal
high-water count insertion and invocation-scoped heap storage, including
cleanup after return snapshots and reuse during self-tail restarts.

I import its one-record regression, adapting the storage assertion to my
one-element heap allocation. My existing expanded tests cover counts 8, 9,
10, 18 and 100 under generated-code ASan/UBSan, recursion, tail calls,
allocation failure and a bounded native stack.

## Gates and boundary

- `make -j4 test-nvm2c`: 1,718 translator checks and 1,073 shape checks pass.
- `python3 -m unittest tests.test_match_block_semantics tests.test_one_ir_compiler`:
  all 27 methods pass, including the four-compiler match/grouping matrix.
- `scripts/check_shadow_tests.sh tests/selfhost/test_infix_ops.nano`: passes.
- The preceding production-identical checkpoint passed bootstrap and the broad
  quick gate. I do not call this test-only merge a new full release acceptance.

My refreshed inventory contains 21 open PRs and no open GitHub issues. I read
titles, bodies, labels and milestones; no item explicitly labels or names 5.0,
but the requested all-branch reconciliation remains open. I fetch the nine
newer PR heads for review. I have not merged this integration branch into main,
published 5.0, enabled Horde SSO or cleared a fleet hold.
