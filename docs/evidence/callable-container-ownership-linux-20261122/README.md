# My interpreted callable constructor prerequisite

I publish source `d732326d70392d4f254d524b35824420a16adebb` for #1140 on
PR1168. I repair borrowed callable metadata in `create_struct` and
`create_union`. I copy each name and complete signature before its source
binding can retire. I change no evaluator cleanup, Value/container layout,
foreign ABI, translator, capture, regex, assertion, corpus selection or deadline.
My [required ownership contract](../../EVAL_CALLABLE_OWNERSHIP.md) remains open.

At original source `53abc0a57aebf11f2becead97478e924910caddb`, my freshly
built private providers reproduce all576 retained names in the original nine
64-call routes. A separate valid source factory returns a record containing a
local callable, then a later call invokes that field. It returns VOID instead
of7. The retained targeted ASan/UBSan diagnostic instruments actual eval/env
and the probe, with other providers fresh but uninstrumented. It reports the
name allocated by `own_function_identifier`, freed by activation cleanup and
read by `eval_call`. I do not describe that diagnostic as full instrumentation.
My raw expression observer separately records borrowed variable/field results
and fresh declaration/array/call results; I free only proven owned probe values.

After the constructor repair, the permanent record/union escape fixture
returns7 and runs all fixture shadows. My direct constructor controls retain
six independently owned values through explicit source retirement, including
record copies, nested records, unions and the existing tuple boundary. Names,
signatures, nested return signatures and parameter arrays are independent.
All six observed names are released by explicit test cleanup. The same control
fails its independent-name assertion against the retained original provider.
This does not claim production carrier reclamation under #1116.

A fresh complete ASan/UBSan provider build uses the same fixture. All source
assertions and shadows finish, then LeakSanitizer fails: **3920 bytes in169
allocations**. I preserve that unsuccessful Make invocation. Its
`--constructors-only` control passes with leak detection enabled. I suppress
no finding. The unchanged original strict audit still retains576 names and
exits2. Ordinary interpreter and typechecker suites pass, linked in private
paths against the fresh candidate providers. The ordinary interpreter retains
its original `obj/test_interpreter_ffi_native.so` fixture path; I archive that
actual image separately. I change no original fixture assertion or timeout.

I retain the first small escaped-record fixture as well as the extended
permanent record/union fixture. A discarded fixture spelling called
`payload.action` as a module function; its source remains `fixture-first.nano`.
The corrected fixture binds the field to a typed local before invoking it.
My unapplied owned-expression draft is archived as a proposal, with no
constructor-receipt qualification attributed to that draft.

My archive contains1468 records,1380 exact Git source inputs and313398223
logical bytes. It retains raw scripts/streams/results, baseline source bytes,
actual linked executable images, source/tool guards and observed individual
provider identities. Linked images retain their provider code; I do not claim
to archive each separate object file. Compressed objects reuse independently
hashed entries from my preceding nested-array receipt. My reader validates
every compressed/raw identity, Git blob, required pass and retained failure.

```sh
python3 docs/evidence/callable-container-ownership-linux-20261122/read_archive.py
```

Puck-owned `nvm2c.c` stays at SHA256
`12d43399e8b3fcda44964f90830d8aa3b24c637df31f0ce2f151527756372d67`;
its consumer stays at
`c94adddd92eefc081ee5cb28df971a36fe2f7cbe8e9730ec68ca6b1c3d24a7ea`.
My retained09cab Stage2 source producer stays at
`346dc0d73af5960dea863fa724a0b1845f5613dff10b8661180ed74bf4eb3d83`.
Linux tools remain GCC13.3/as2.42. I hold no Darwin reservation and perform no
NAS/migration/TCC action. Frozen roots, primary dirty audit, private candidates
and stashes stay preserved. Complete callable ownership, full-source/platform
qualification, #1202/#1204 caller repairs and the69-requirement release remain
open.
