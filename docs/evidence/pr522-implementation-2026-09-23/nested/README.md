# My nested ordinary union checkpoint

I admit bounded nested ordinary union payloads and validate unused arguments
separately from stored ownership. My existing checker expansion bound is 128;
unknown arguments, repeated-instance cycles and expanding recursion remain
refusals. The new shadows and raw producer tests exercise those boundaries.

My first corrected producer executes the unchanged nested fixture in NanoVM,
then native translation refuses `variant-scalar to string`. I retain that
terminal. I now widen only inferred optional payload storage when the incoming
constructor establishes a finite payload set. Exact payload constraints and
unboxed destinations keep their refusal boundary.

Validation at this source checkpoint:

- Fresh two-stage bootstrap, mandatory shadows and installed checks pass.
- All 23 selected methods pass: the existing nested selected pattern, three
  ordinary instantiated-ownership cases, the complete native nested-generic
  class and the complete affine generic-identity class. The applicable cases
  run with C-seed, Stage 1 and Stage 2.
- All ten emitter-driver methods pass, including VM/native nested values,
  phantom resource arguments, cycles and preserved outputs.
- The full translator suite passes 2,428 assertions. Shape tests pass 1,500
  assertions under strict ASan/UBSan with leak detection.
- Eleven finite variant-carrier methods pass with Homebrew LLVM and mandatory
  leak checks. My initial default-Apple-runtime attempt aborted before their
  semantics; I retain its explicit unsupported-leak diagnostics.
- Both padding methods pass. The old integer-array negative already succeeds
  with the unmodified `881022af0` shape implementation; I retain that baseline,
  use an unsupported string-array payload for the refusal and retain the
  separate integer-array positive corpus.
- The generated nested native program passes strict ASan/UBSan with leaks
  enabled. `native-instrumentation.json` records the source/module hashes and
  exact instrumentation choices.

## Remaining alias-return blocker

The existing `test_result_arguments_survive_alias_and_return` still fails.
`alias-return.nano` retains its source. Its main program constructs only the
string `Err` variant; native classification rejects the integer return in the
unreachable `Ok` arm. I need branch-sensitive evidence while preserving
reachable wrong-tag refusals. The first five-method Stage 1 attempt retains
that failure; the later 23-method gate does not include this unfinished case.

Resource-bearing unions and the complete platform/release gates remain open.
`logs.json` seals every retained uncompressed terminal; none of these focused
results establishes complete PR522 acceptance.
