# My standalone owned-transfer execution evidence

I implement task_bd2f689969f044a583548cefee6d4f11 after PR555. My shared
`nvm_verify_owned_module` contract admits one zero-argument, zero-capture
function with an int/bool/u8 result and an explicit owned-transfer instruction.
I require complete finite nested records with those scalar leaves, value-mode
locals, no imports/linked ownership contracts and actual affine dataflow
verification. Calls, references, floating fields/operations, aggregate results,
collections and source borrow production remain outside admission.

My normal verifier, VM and native translator use that eligibility contract.
Moves clear their source; pack transfers fields; whole-record unpack transfers
all fields before releasing its emptied shell. My native translator emits
constant-index C temporaries and direct labeled control flow. It neither
embeds bytecode nor calls a VM. I reclaim native shells immediately; VM unpack
also reaches the existing collector safe point for a shell previously buffered
by a field observation.

At my implementation checkpoint based on `01b6e1bf`, I pass:

- 1,051 runtime/eligibility checks across eight ordinary programs, including
  nested unpacking, a branch, 1,000 loop iterations, field observations,
  Boolean/byte fields and wrapped signed division. Twenty VM invocations per
  program return to the same heap-object baseline.
- Every native record allocation failure in those programs, under ASan/UBSan
  and leak detection. Successful runs return the same scalar result, failures
  leave zero live allocations, and peak live shells stay at most three.
- 32 VM heap-allocation failure checks. My focused sanitizer build instruments
  VM execution, heap/cycle collection, affine state/dataflow, verifier and
  native translation; remaining linked objects are ordinary builds.
- Seven ordinary exclusion fixtures, including shared parameters and floating
  record payloads, preserving previous native output. Linked explicit
  transfers remain excluded even when their record flags are non-resource.
- 184 transfer/roundtrip checks and 272 allocation-injection checks. Earlier
  three error-message assertions now correctly become successful admission
  assertions for the supported subset; unsupported cases remain checked.
- 300/419 affine bytecode checks, 157/182 state checks, declaration publication
  controls, 272,624 VM checks, 2,735 ISA checks and 33 schema checks.
- 2,412 existing native checks and 1,092 shape checks before the final shared
  opcode-scan guard extraction; the lowering source is unchanged by that
  extraction.

My initial observation/unpack probe found a zero-reference shell retained in
the existing cycle-suspect buffer. I preserve
`/tmp/nanolang-owned-runtime-initial-lifetime.log`; the added unpack safe point
passes the original exact heap-baseline assertion. An intermediate linked
check was too broad for ordinary checked-module fallback; I narrowed it to
ownership contracts and explicit transfers, and the complete VM suite passes.
Neither intermediate failure is reported as a passing gate.

Logs include `/tmp/nanolang-owned-runtime-linked-final.log`,
`/tmp/nanolang-owned-runtime-adjacent-final.log`,
`/tmp/nanolang-owned-runtime-aot.log` and
`/tmp/nanolang-owned-runtime-asan.log`. Integrated seed/dispatch evidence follows
once those gates complete. My full ownership, caller-reference and release
parents remain open. Float-record transport remains the independent task93574.
