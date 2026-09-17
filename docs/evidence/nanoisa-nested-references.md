# My bounded nested-reference execution evidence

I implement MAC `task_556d6702b3c74377b0cb830b6d988c68` under my
[declared contract](../NANOISA_NESTED_REFERENCES.md). My source checkpoint is
`fa49f5c4`; my integrated compiler checkpoint is `67545490`, including main
`c3166cd8`. I preserve one standalone zero-argument function and int/bool/u8
scalar results. I do not admit caller references, borrowed parameters,
imports, aggregate results or frontend borrow production.

I retain OWNERSHIP format 1 byte-for-byte and add format 2 numeric paths with
256 entries and 32 fields per path. I resolve paths against retained nominal
layouts. Four new instructions create nested shared/exclusive references and
reborrow existing references. My verifier enforces overlap, permission,
parent suspension, region lifetime and exact joins. Runtime descriptors alone
do not establish permission to run unverified instructions.

My VM retains root/path indices across suspension and stack relocation, then
resolves the actual owner on access. My native output walks those same owner
fields. Both mutate the original record without copy/writeback. Ending a
child region restores its parent's permitted access; terminal errors and
completion clear the activation, while a core yield preserves it.

## My measured gates

At `fa49f5c4` I pass:

- 1,662 nested-reference checks, including eleven paired VM/native programs,
  sixteen refusal cases, exact reconstruction and a 32-field path;
- 63 allocation-failure/recovery checks, with no remaining owners or slots;
- 144 ownership metadata checks, including old-format preservation, public
  verifier/native refusal and maximum table/path bounds;
- 1,336 root-reference checks plus 29 allocation checks, and 1,051 owned
  transfer checks plus 32 allocation checks;
- 300 affine bytecode checks plus 419 counted conditions, and 157 affine
  state checks plus 182 counted conditions;
- 272,624 VM checks under both dispatch modes, 2,845 ISA checks and 33 schema
  checks;
- computed-goto nested execution and ASan/UBSan nested execution, allocation
  recovery and emitted native programs, with leak detection;
- the genuine canonical native compiler seed build, help, hello bytecode
  publication, verification and execution through its host module;
- 2,414 native translator checks, 1,092 shape checks, 22 LLVM methods and
  16 WebAssembly methods.

My integrated checkpoint reruns nested/root/owned execution, ownership
metadata and the newly merged U8 string checks successfully. I also repeat
computed-goto VM/nested execution and the nested ASan/UBSan allocation and
execution gates at that integrated checkpoint. I retain the
broader translator counts as evidence for the earlier named checkpoint;
they are not an assertion that the full release suite passed at this head.

My retained local logs are `/tmp/nanolang-nested-boundary-final.log`,
`/tmp/nanolang-nested-integration.log`, `/tmp/nanolang-nested-restack.log`,
`/tmp/nanolang-nested-computed.log`, `/tmp/nanolang-nested-asan.log`,
`/tmp/nanolang-nested-seed-build.log`, `/tmp/nanolang-nested-seed.log`,
`/tmp/nanolang-nested-help.log`, `/tmp/nanolang-nested-hello.log` and
`/tmp/nanolang-nested-translators.log`,
`/tmp/nanolang-nested-computed-integrated.log` and
`/tmp/nanolang-nested-asan-integrated.log`. My checked-in tests are the durable,
repeatable evidence; these local logs are not published release assets.

Caller-place substitution, multi-parameter alias checks, source production
and full affine acceptance remain required work. My parent tasks ed702 and
718 and the full v5.1 publication hold stay open.
