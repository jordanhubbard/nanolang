# My paired File/TCP source plans and namespaces

I qualify this batch above parent `8cde5a352` on Darwin arm64. I extend the
shared descriptive planner and both namespace/type collectors; I do not yet
admit TCP execution. My legacy File planner remains File-only. The mixed API
retains catalog identity on every original and aliased row, including public
Conn and scalar Endpoint. Request/module identities distinguish overlapping
File/TCP type names.

## Checks

- My complete File corpus and17-case mixed corpus pass through the C seed,
  installed Stage1 and installed Stage2 drivers, with exact selected shadows
  and identical C/Nano output. The initial combined run passes three methods
  in125.107 seconds. I then add TCP-specific counted-span and exact1MiB budget
  shadows; the mixed corpus passes again in38.833 seconds with all expected
  shadows. I retain that run's command, terminal and selection records.
- My C planner ownership checks pass with LLVM ASan/UBSan and GCC16. The mixed
  plan copies names/module strings before input mutation, preserves alias
  catalog/category facts, and preserves the caller output pointer when its
  single owning allocation fails. The original exact-budget checks remain.
- The existing real-loader namespace cases and independent Nano File identity
  fixture pass. My final mixed namespace fixture passes in140.743 seconds:
  actual C recursive imports and allocation-prefix failures; alias collisions
  and private Socket refusal; Conn, Endpoint, SocketError and all method types;
  parameter/return annotations; C-producer and installed Stage2 compilation;
  and VM/native execution. Both body checkers retain the explicit unsupported
  TCP boundary, and actual C-driver output remains unchanged on refusal.
- All nine existing body and ownership methods pass in the retained combined
  log. That same run still contains the earlier new-fixture failure; I do not
  describe its overall terminal as passing.
- LLVM sanitizers pass the original C namespace graph and the new mixed graph.
  Only the embedded namespace implementation/fixture is instrumented in this
  target; linked ordinary compiler objects are not a fully sanitized compiler.
  GCC16 also compiles `src/service_namespace.c` with C99 and strict warnings.

## Retained failures and scope

I preserve the first mixed fixture's incorrect expectation of an internal
body diagnostic at the CLI boundary. The loader exposes a generic unsupported
consumer refusal; the corrected fixture inspects the retained body fact and
checks publication separately. I also preserve its one-versus-two annotation
count failure and Stage2 undefined-tokenizer refusal. The corrected source
checks both parameter/return identities and imports its dependencies explicitly.

My first namespace sanitizer invocation supplied CC only in the environment;
Make selected Apple Clang, whose runtime rejected leak detection. The corrected
invocation supplies LLVM CC as a Make argument and passes. I retain both logs.

I compile changed Nano helper sources through previously installed compiler
generations. I have not performed another complete bootstrap for this batch;
that qualification remains required after the connected Socket compiler work.
TCP body/ownership lowering, wire metadata, VM/native dispatch, selected
network shadows, public connect/WebSocket and exact-candidate Linux/Darwin
acceptance remain open under #990 and the full5.1 release contract.
