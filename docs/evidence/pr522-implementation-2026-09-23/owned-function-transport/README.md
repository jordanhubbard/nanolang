# Owned function-reference transport

I preserve noncapturing, same-module `FUNCREF` values through initialized locals,
stack copies, direct-call parameters and direct-call results in my bounded owned
value graph. I retain their function tag; numeric operations do not accept them.
My entry remains a zero-argument scalar result, so callers cannot inject an
unchecked function reference. Direct call graphs remain bounded and acyclic.

This is a prerequisite for resource callbacks. I still refuse `CALL_INDIRECT`
in this graph. Source callback parameters, finite indirect targets and consuming
argument/result transfer remain unfinished; I do not claim the four original
native resource-callback failures are repaired.

I preserve the refusal of closures, function-valued resource fields, borrowed
function slots, nominal layouts on function slots, uninitialized loads, wrong
argument/result types, invalid targets and arithmetic on function references.
Initialization meets require a definition on every incoming path.

## Evidence

I ran on Darwin arm64. My new positive fixture transports a function through a
direct helper while an owner is live, then consumes the owner and returns 42.
Its trap variant asserts before consumption. Four VM entry APIs, each repeated
twice, check empty frames/stacks/reference state and the original heap count.
The generated native harness repeats success/trap execution 20 times and injects
allocation failures until the first uninterrupted run, requiring zero retained
allocations after every invocation. ASan/UBSan and leak detection stay enabled
for generated native code. VM/compiler objects in this graph gate are ordinary.

- `make test-affine-state test-affine-bytecode`: 452 ordinary and 484
  allocation-failure state checks; 837 ordinary and 1,199 allocation/visit-failure
  bytecode checks. [Terminal](core-final.log.gz).
- `make CC=/opt/homebrew/opt/llvm/bin/clang test-owned-value-graphs`: all 12
  cases pass, with 2,169 graph assertions, 338 preflight checks, 529 invocation
  proof checks and 69 verification-reuse checks. All cases retain VM/native
  parity, byte-identical retranslation, generated-program sanitizer checks and
  allocation-failure cleanup. [Terminal](graphs-final.log.gz).

My first graph run found a missing verifier local-type admission check, retained
in [the original terminal](graphs-before.log.gz). After correcting it, Apple's
sanitizer runtime refused `detect_leaks=1` in every generated case
([terminal](apple-lsan-refusal.log.gz)). An environment-only `CC` override was
superseded by Make. Passing LLVM as a Make command-line variable selected the
runtime that supports leak detection; I did not weaken the sanitizer options.

My complete translator gates pass 2,446 assertions each:
`make test-nvm2c` ([terminal](translator.log.gz)) and
`python3 scripts/run_nvm2c_sanitizers.py --cc /opt/homebrew/opt/llvm/bin/clang`
([terminal](translator-san.log.gz)). The latter builds fresh private objects and
verifies ASan/UBSan symbols in translator and shape objects. Its existing broad
gate disables leak detection; the graph-specific generated-native gate above
keeps leak detection enabled. These are distinct scopes.
