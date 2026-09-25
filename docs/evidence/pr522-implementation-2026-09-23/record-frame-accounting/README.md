# My record-frame allocation accounting

I retain the debugger backtrace for the original abort in `tracked_free`.
My fixture counted only `calloc` allocations but decremented that count for
root-list storage obtained by `realloc` too. My generated root-list destructor
correctly frees both its item buffer and its hash slots.

I now track allocation identities across `malloc`, `calloc`, `realloc` and
`free`. I count the explicitly identified record-frame allocations separately
from root bookkeeping. My existing self-tail peak of two frames, ordinary and
cross-tail minimum of 26 frames, ten repetitions, result assertions, 2 MiB
stack limit and allocation-failure trap remain checked. Every iteration must
release **all** tracked allocations, including root bookkeeping.

The original method passes all three modes and its five temporary-count
boundary cases with Homebrew Clang. It also passes with generated C compiled
under ASan/UBSan, recovery disabled and leak detection enabled. The retained
logs report 2.536 and 3.128 seconds respectively. I changed no compiler code.

My complete One IR gate remains open: twelve generic-array ABI cases, two
tagged projections through locals and the C-seed compiler translation refusal
still require correction. This fixture repair does not qualify those cases.
