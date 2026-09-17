# My one-parameter caller-origin evidence

I implement MAC `task_48dcff3ff3314e9aad325209387968d1` under my
[call contract](../NANOISA_CALLER_REFERENCE.md). My source checkpoint is
`ec5dddd6`, based on merged nested-reference main `925b2127`.

I connect CALL_REF (primary 0x0f) to actual caller-place substitution,
affine helper analysis, two bounded VM descriptor contexts and a private native
helper. I preserve ownership section formats 1 and 2. I admit entry 0 plus
one nonrecursive helper with one exact shared/exclusive scalar-leaf record
parameter and an int/bool/u8 result. My helper's value local 0 stays void and
non-authoritative; only its reference slot carries caller authority.

My callee retains the caller frame, generation, owner local and path. A call
checks permission as a reborrow against every existing caller hold; the
synchronous helper cannot access caller storage except through that authority.
Returning expires helper references and leaves the caller's earlier holds
intact. The verifier establishes overlap and parent-suspension permissions;
runtime descriptors alone are not permission to execute unverified code.

## My measured checks

At `ec5dddd6` I pass:

- 1,548 caller checks across nine paired VM/native programs and twelve refusal
  cases, including original nested-owner mutation, repeated calls, shared
  downgrade, helper reborrow, caller restoration and scalar result tags;
- a genuine yield inside the helper, relocation of the value stack and resumed
  mutation through the original caller root/path;
- direct, callable, invocation and module-entry rejection while suspended,
  preserving the live descriptors, plus helper entry rejection before execution;
- 43 heap-allocation checks and 55 helper-fact allocation checks. I fail each
  malloc/calloc in runtime parameter-state construction, reclaim owners without
  publishing a partial helper activation, then execute a later call successfully;
- 83 concrete caller bytecode analysis checks, 214 ordinary affine-state checks
  and 243 checks with allocation injection, plus the existing 300/419 affine
  bytecode checks;
- 2,856 ISA checks, 33 schema checks, 144 ownership metadata checks and existing
  root/nested/owned execution gates;
- 274,416 VM checks with both dispatch modes, plus computed-goto caller execution;
- ASan/UBSan caller execution and heap allocation recovery, emitted native
  programs with leak detection, and exact serialized/reconstructed native output;
- the genuine canonical native compiler seed build, help, hello bytecode
  publication, verification and execution through its host module.

My initial integrated gate exposed an overbroad host-entry guard: it also
refused ordinary metadata-bearing helper functions. I narrowed the guard to
modules containing actual owned/reference instructions and reran the existing
ordinary ownership-metadata controls successfully. I preserve the original
failure in `/tmp/nanolang-caller-integration.log` and the corrected result in
`/tmp/nanolang-caller-metadata-corrected.log`.

My other retained local logs are `/tmp/nanolang-caller-final-focused.log`,
`/tmp/nanolang-caller-computed.log`, `/tmp/nanolang-caller-asan.log`,
`/tmp/nanolang-caller-seed-build.log`, `/tmp/nanolang-caller-seed.log`,
`/tmp/nanolang-caller-help.log` and `/tmp/nanolang-caller-hello.log`.
Checked-in tests are my durable repeatable evidence; these logs are not
published release assets.

I keep multi-parameter alias substitution open under
`task_7a2c8017c0c04b82a48ba069561e9d36`. Nested/deeper calls, imports, callbacks,
owned helper locals, aggregate results and source frontend production remain
outside this admission. My affine/borrow parents and full v5.1 publication hold
stay open.
