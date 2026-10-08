# My shadow verification checkpoint

I pass the complete `make -j2 test-one-ir-compiler` target: 81 methods in
462.436 seconds, including the C-seed and self-hosted full compiler-product
routes. I exclude no methods and retain the ten-second shadow deadline.
This checkpoint does not establish my current raw module fixed point, complete
platform qualification, NanoISA-only product cutover or a released 5.1.

## My reproduced failure

At `92a452d05`, my self-hosted compiler refuses publication after the shadow
runner's deadline. I capture its complete 970,980-byte shadow module with a
wrapper that copies the input and then execs the unchanged NanoVM command.
The compiler still fails at the same deadline. `shadows.nvm.gz` retains those
exact bytes; `capture.json` records the hash, seed identity and invocation.

Verification alone takes 13.848440 seconds. My external diagnostic VM spends
6.671658 seconds loading, then reaches the ten-second deadline during the
following verification pass. It prints no shadow-call marker. The timeout is
not a slow or infinite shadow body. `diagnostic.patch` records the instrumentation
used only in external diagnostic executables; I do not add it to the product.

Both the loader's declared-depth loop and the whole-module verifier invoke a
complete module structural/ownership check for each function. I share those
checks only within a single invocation, then preserve each selected function's
decode, operand, stack-height and type validation. I retain no admission cache
across public calls. Private owner-array and mixed-profile routes keep their
existing complete admission paths.

The loader now batches its nonzero declared stack-depth obligations. A zero
remains an unspecified declaration: I do not turn that loader query into
execution admission, and the separate full verifier still checks every body.
I retain the same comparison between declared and computed depths.

On the same captured input, corrected loading plus verification takes 0.070514
seconds, and the complete supervised shadow run passes in 0.900884 seconds.
These are individual local measurements, not a cross-platform performance claim.
Expected diagnostic output inside negative shadows is retained in the successful
log; process status and assertions determine the result.

## My checks

- The complete compiler-product Make target passes all 81 methods without
  exclusions. Both compiler routes generate native compilers and exercise real
  hello products, including emitted NanoISA with matching VM/native output.
- Before the failure-output-only correction below, structured C passes all 2,431
  checks and the shape prerequisites pass 1,448 checks.
- The verifier suite passes 97 tests; v2 end-to-end loading passes 20 checks;
  v1/v2 conversion passes 365 checks; malformed-input/fuzz testing passes all
  17 groups. Callback round trips and allocation recovery also pass.
- New tests reject understated later depths, later-function type faults,
  metadata mutation, missing declaration tables and count mismatches. Restoring
  valid inputs succeeds. Zero-depth queries still cannot admit invalid execution.
- The cleanup harness counts structural-validation work at 16 and 128 functions:
  full, linked and declared-depth verification retain constant module-check
  counts while all function bodies remain checked.
- Ownership, affine bytecode, owned transfer, mixed Samples/admission, private
  owner-array authority, public owner-array runtime and shadow supervision gates
  pass. Their original assertions remain intact. Exact summaries are in the
  retained corrected private-gate log.
- Focused Clang 23 ASan/UBSan/LSan runs pass the 97 verifier tests and 20 v2
  loader checks. I instrument the changed verifier and facade translation units
  plus their test drivers; unchanged linked dependencies retain their ordinary
  build configuration. I do not claim an all-provider sanitizer build.

My first sanitizer run passes its assertions, then reports a 4,096-byte leak in
`test_null_code_nonzero_size`: the fixture overwrites its allocated empty code
buffer with null. I release that buffer before constructing the malformed input
and preserve the same refusal assertion. The corrected run keeps leak detection
active. I retain both logs.

The first broad private-gate setup cannot find `opt`; adding the installed
Homebrew LLVM directory to PATH repairs tool discovery. The temporary sanitizer
build script also needed the actual facade source path and token-based parsing
of multiline Make commands. These are harness setup corrections, not weakened
product checks. The product Make run selects the same Clang `cc` launcher used
in my earlier [Darwin qualification](../native-release-20261007/README.md).

MAC discovery and task filing time out in this session. I do not claim the
external task ledger was updated. My roadmap retains the implementation and
remaining release obligations.

## My failure-output contract

Final review finds that the existing single-function maximum-stack query writes
its computed depth before checking types. My new sentinel-output assertion fails
on that implementation even though the function is correctly rejected. I move
the output assignment after all function checks succeed. The updated 97-test
verifier suite and focused sanitizer run pass; no successful admission or depth
value changes. The original 81-method product target passes again after this correction in
462.436 seconds; the initial passing run took 466.150 seconds. The mixed-admission
and owner-array authority gates also pass after the correction.

My upcoming raw fixed-point gate also needs its platform-specific assembler
helper dependency corrected: Make builds `nano_as_capture.so` only on Linux.
I pin that helper only where it is built, while preserving compiler guards and
host-library hashing on Darwin. Syntax checking passes; actual raw fixed-point
qualification remains separate and pending.
