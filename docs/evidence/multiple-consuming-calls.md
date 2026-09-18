# My multiple consuming-parameter runtime evidence

I qualify `task_6d8f539329294b24ad2746bec6a22862` under my
[bounded contract](../NANOISA_MULTIPLE_CONSUMING_CALLS.md). I retain the source
admission task `task_d54033e921f24e279e53e8ed4cf50d17` and full ownership parents.

## My source identity

My contract precedes implementation at `ca70ac47`. My six-file production
change is `9dafed68`; root independently reviewed its shared positional
contract, affine stack transfer, VM preflight and native carrier clearing.
My initial tests are `8f64a7a7`. Integration `7f1278b6` retains main `3c77b88e`,
including PR712 native map-key handling and PR715 array-slice cleanup.
My five NanoISA production files remain byte-identical to the reviewed change;
the only subsequent VM delta is that independent array-slice cleanup.

## My corrected fixture boundary

My first test build caught two C warning errors, corrected without suppressions.
My first fixture run omitted transferring the assembled parameter type arrays
when replacing the baseline fixture's function entries. I corrected the
fixture transfer before the ordinary positive gate passed.

An instrumented test run then caught an arity-bound refusal that changed
arity eight to nine without rebuilding the in-memory parameter array. I retain
`/tmp/nanolang-multiple-consuming-sanitizers-corrected.log`. I did not replay
that fixture. At `13025b1b` I use `nvm_set_function_param_types` after changing
arity, validate complete ownership metadata, and directly check the bounded
contract API before static module refusal. This is a corrected test scaffold;
I do not attribute its invalid metadata to the consuming runtime.

My first sanitizer build script also assumed every NanoISA object came from
`src/nanoisa`; the facade comes from `modules/nanoisa/nanoisa.c`. I corrected
that build mapping before running the instrumented driver.

## My measured gates

- My original full gate passes native 2,422 and shape 1,365 checks; VM 274,493
  plus heap/stack recovery checks; affine state 314/343 and affine bytecode
  441/751 checks. Single-owner calls pass 3,091 lifecycle, 104 heap-fault and
  90 preflight checks. Helper-local owners pass 1,309/106; assertions 959;
  caller references 1,548/43/55; multiple borrowed parameters 2,004/93/89.
  I retain `/tmp/nanolang-multiple-consuming-full.log`.
- After additive main integration, verifier 96 and VM 274,493 checks pass,
  including the retained upstream slice/callback/heap/stack controls.
  My single-owner gate also passes unchanged. I retain
  `/tmp/nanolang-multiple-consuming-integrated.log`.
- Corrected focused acceptance passes 4,542 lifecycle/refusal/API checks,
  210 heap-fault checks and 216 separate preflight checks. GCC native
  ASan/UBSan/LSan and strict standalone checks pass in 5.967 seconds;
  Clang18 passes in 4.808 seconds. I retain
  `/tmp/nanolang-multiple-consuming-final-focused.log` and
  `/tmp/nanolang-multiple-consuming-final-clang.log`.

My eight valid cases check two owners in both nominal orders, eight interleaved
arguments, nested resource trees, materialized scalar field values, distinct
caller/helper owners at the same local slot, repeated calls, and entry/helper
assertions. All four VM APIs retain exact INT/BOOL/U8 result tags. Native
allocation sweeps require zero remaining allocations and an unchanged external
result sentinel on failure.

My heap fault injection covers allocations inside `heap.c`. My separate
preflight injection covers actual VM frame reallocation failure, a failed
helper-contract creation result, and a final-position layout or tag mismatch
in an otherwise valid contract result. The latter never enters a mismatched
helper. All four APIs leave no active frame/reference contexts or remaining
heap owners, retain the preactivation generation, and then succeed on the same
VM. These measured cases do not claim injection at every system allocation.

I verify semantic ownership refusals without executing rejected modules.
My separate publication checks retain existing native output for a wrong
nominal argument order and an unconsumed helper owner.

My final focused publication gate passes both methods in 6.022 seconds,
including 65 serialization/refusal checks, with the same 4,542/210/216
lifecycle and fault counts (`/tmp/nanolang-multiple-consuming-publication.log`).
My separately instrumented VM/NanoISA run with the ordinary 16 repetitions
per API timed out at its explicit 600-second bound. Its last observed progress
had prepared cases0 through4; I retain
`/tmp/nanolang-multiple-consuming-sanitizers-final.log` as incomplete. I do not
classify that timeout as a compiler correctness failure or a passing gate.
I retain the ordinary 16-repeat default. A separate instrumented qualification
uses `MULTIPLE_CONSUMING_REPEATS=2`, preserving all eight cases, all four APIs,
reentry and every lifecycle assertion; its result remains pending.
