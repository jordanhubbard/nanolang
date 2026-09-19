# Native TCO binding evidence

I qualified the remaining native self-tail-call binding work from canonical
`2fc5496613104707dd153c14c4777d62bc3e318f`. The first production and test
checkpoint was `7f6dc47c` on `fix/native-tco-bindings`; the contract checkpoint
was `1f4f7fe6`. Independent review of the later pushed head found two admission
defects, so that head was not merged. I preserve the first evidence below and
record the corrected boundary separately rather than relabeling it.

## What I admit

I retain complete hidden binding metadata for metadata-complete scalar-element
arrays, flat scalar-field records, scalar tuples and function-valued parameters. I evaluate next
arguments once from left to right before publishing any parameter, resolve
parameter references lexically, and propagate a tail iteration out through
original nested loops. Calls through a function-valued parameter remain
indirect callable expressions.

I refuse missing aggregate metadata, composite record fields, nested arrays or
tuples, unresolved/resource/extern records, unions, opaque or borrowed
parameters, binary strings and `par`/`flow` blocks. Unsupported AST forms leave
the function tree unchanged. This evidence does not establish a new aggregate
ownership ABI.

## Independent review correction

Review of `88a16642194009d38fab01af73709709767d41e4` found that my `AST_PAR_BLOCK`
traversal evaluated initializer expressions but never published the resulting
bindings. A later reference could therefore be renamed to an outer TCO
parameter, and a flow dependency could be misread. The same review found that
the aggregate preflight admitted tuples without inspecting their metadata,
arrays with unknown or composite immediate elements, and records without a
transitive ownership proof.

I recorded those defects before correction at `d4ec2ada`. The corrected pass
refuses every `AST_PAR_BLOCK` before mutation and narrows aggregate admission to
the exact boundary above. Five new pass units preserve the original function
body, statement vector and tail-call spelling while refusing a `par` block, a
nested resource-bearing record, a nested array, missing array metadata and a
tuple containing a record.

The corrected strict C pass unit binary passes every optimization test. Log
SHA-256:
`37635d1dca701ec4b49bc7641ed1aae70beb134868582379571aa60132e80671`.
The unchanged 14 executable optimized/unoptimized native cases pass in 32.731
seconds. Log SHA-256:
`dd9ae9ac445c20eb7c0febf409f119d7466f0d20f44c64cac921abefa559ba80`.
This correction still requires independent review of its exact pushed head
before merge.

## Retained terminals

I kept each first terminal and corrected only the demonstrated boundary:

- The first focused run passed the 200,000-step array case, then the optimized
  callable case returned binary stderr because a generated callable binding was
  dangling. Log SHA-256:
  `07fdbc9e8543a60a1fed9c3288bb8f46692d95e1fb8aeccb18fc3a54c209b2bf`.
- Converting the synthetic callee to an indirect expression was necessary but
  not sufficient. The second terminal has SHA-256
  `b585fb981e85bb14ce36c181a0a60a59fd3fbc13ae9e2a6729563e9228b98a83`.
  Raw capture then showed a freed function name during shadow execution. I copy
  borrowed function identifiers when `let` or `set` creates a new owner.
- The next two fixture terminals preserve the unsupported bare block and native
  same-name C initializer outcomes. Their SHA-256 values are
  `a250a367b8a367b03a75f0f77820a4bdb293629b3d8a3ad9d23dc8e84ba0a892`
  and `dd1975a4238611a55ef1ce80f60b74ade81e067776b762836df1eaa035a7e5c4`.
  The separate same-name native emitter task remains
  `task_19e8c98dcc774e21975f763648e91adb`; the TCO unit gate checks the exact
  initializer binding directly without claiming that backend repair.
- The first loop fixture tried to assign an immutable counter. The retained
  terminal SHA-256 is
  `68db6603f4da0e0fc46770742c29d0c6dcb5cda57d9199a7c67a8b3058981b85`.
  Only those counters became mutable.
- The first resource-refusal unit used reserved `handle` as a parameter name.
  That parser terminal is retained in the first unit log, SHA-256
  `c358fe991d6e7c9501b0bf6bad23100fa72e975a7b34c48ad482df3232fe9e70`.

## Passing gates

- `make -j8 test-opt-passes` performed a fresh Stage 1 and Stage 2 bootstrap,
  both hello smokes, the installed-compiler/no-C-seed smoke, all then-current 14
  executable TCO methods and all optimization units: PASS in 329.79 seconds.
  Log SHA-256:
  `7280a74a364ffdd0b73969e096b2a1d86d0e9779ac0f0983c1e620089dd15d17`.
  The native Stage 1 and Stage 2 binaries differed; I preserve that output and
  make no fixed-point claim here.
- After adding the conservative resource preflight, the exact strict C-seed
  rebuild passed. Log SHA-256:
  `b036d866b6cacbbfabbaa40334d629e473c7847d658e9ce2ccca71d36cdcf84e`.
- `python3 -m unittest -f -v tests.test_native_tco`: 14/14 PASS in
  32.209 seconds, including the callable swap, arrays, records, tuples, lexical
  shadows, both loop forms, nested-loop propagation and one million tail calls.
  Log SHA-256:
  `90662474d0955182360629f6cf3efba5e75038e14b1fe2d3544d2db59819842d`.
- The exact optimization unit binary linked against the final strict objects
  and passed every unit, including same-name initializer identity, unsupported
  AST immutability and resource-parameter refusal. Log SHA-256:
  `153d7c03bfb70f32089232b1ee5a8cf3f4681ffc63a14987066203a86865101d`.

## Frozen identities

At the corrected production checkpoint before its final commit:

- `src/tco_pass.c`:
  `7c18305bb1e53172654cfd5cf30cfe2726efe3e78440a7991d33fbcb50b82b2b`
- `src/tco_pass.h`:
  `fe6c2b50f9f676aa2d53b43b3967577e4ce4d678452275ba5b605a70f889a208`
- `src/eval.c`:
  `6ad79837473f7ca736610476a9f9e1b5bc17e735d14a68365a45ac9e914f7187`
- `tests/test_native_tco.py`:
  `2430919fc46da56842521785f6cff569a819e92455ff426469bc211ceb607691`
- `tests/test_opt_passes.c`:
  `f2ab401e96be808662c872e8d98765b23a2ee3953c7024d065b98279db810190`
- Apple Clang 21.0.0 actual executable:
  `1590ac950a3d627817d09ade5cb60b2115f17a72182a3141e010b4bcc482a0c9`
- corrected `bin/nanoc_c`:
  `5124eb0c7d1a6b34a090f4f3f9d7832cfaf9b0fe49bdff45b5681a057cc12483`

Host: Darwin 25.6.0 arm64; Python 3.14.6. This is bounded native TCO
evidence, not product PR522 acceptance or release permission.
