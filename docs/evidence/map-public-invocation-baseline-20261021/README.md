# Ordinary borrowed-VmState baseline

No public map authority/admission changed. The existing unannotated modules use
ordinary public VM execution. Exact source, compiler/tool/object hashes, commands,
statuses and outputs are retained here; no new frontend/compiler was built.

- Four unchanged map_record_fields originals (seed, retained C-hosted Nano emitter,
  retained stages1/2) run twice in the same VmState with explicit stack-result
  consumption: 84 checks including the existing alias fixture, all PASS.
- Unannotated existing runtime_alias_failure byte fixture: first result is a record
  aliasing global0; after result consumption the second call fails ASSERT (8),
  preserving global identity and changed map value2. Ordinary legacy failure leaves
  stack3/frame1 until destruction. vm_destroy releases all objects. A future selected
  route's immediate unwind is a deliberate selected-array-style contract, not a
  claim about this ordinary legacy behavior.
- New minimal global.nano is an acceptance probe, not an unchanged existing fixture.
  All four producers accept it, original verification passes, and each resulting
  module executes twice in one VmState (53 checks each). Its initializer replaces
  global record identity on the second invocation. An explicitly retained alias to
  the prior record/map remains valid until released. Both executions return0;
  globals remain1. This establishes the concrete source-global lifetime obligation.

The driver is O0 with inherited ASan/UBSan provider flags, existing ordinary VM
objects and no private runtime policy. No assertion or source rewrite was needed.
`first-run.log` predates the added explicit old-alias identity check; final run and
source logs include it. Publication remains absent; this is lifecycle evidence,
not singleton-boundary proof or selected execution qualification.
