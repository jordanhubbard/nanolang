# My advisory metadata transport evidence

I implement MAC `task_ba62a060883b468abffe6a3cdda49005` under
[my transport contract](../NANOISA_ADVISORY_METADATA.md). My source checkpoint
is `304005c4`; `8a455bdc` strengthens the public legacy refusal and synthesized
source-key tests. My integrated checkpoint `21cef627` includes main `308235c8`,
including the owned assertion and filesystem result-lifetime changes.

I preserve ordered duplicate key/value entries and exact existing string bytes
through the v2 bridge and canonical text. The last explicit `nano.source_file`
entry controls the legacy view; I refuse a contradictory manual view change.
I synthesize that entry for legacy source-only modules without repeatedly
adding its key. Explicit debug stripping removes source entries while leaving
other advisory entries. Metadata-bearing v1 serialization refuses explicitly.

I allocate and free the entry table with its module; indices refer to that
module's owned string pool. My v2-to-working-module bridge copies entries and
strings before releasing decoded storage. Linked VM modules retain their own
module pointers, not a flattened copied table. The temporary whole-module view
in `nvm_retain_layouts` borrows existing fields and publishes only the newly
owned layout bytes. I found no separate owning module clone requiring an
additional metadata copy. No new implementation file needs manifest linkage.

I preserve the existing bridge error convention: failure of
`nvm_add_metadata` while converting v2 to a working module reports
`NVM_V2_ERR_INDEX_RANGE`, including allocation failure. The partial module is
freed and no output module is published. My direct append API returns false
and preserves prior entries and the source view on failed growth. I do not
claim a distinct out-of-memory diagnostic for that conversion path.

At my integrated checkpoint I pass:

- 188 metadata checks: exact embedded zero/high bytes, ordered duplicates,
  source precedence and empty source value, repeated conversion stability,
  canonical text, public v1 refusal, debug stripping, and unchanged retained
  layouts/ownership/passive data;
- 18 injected bridge allocation boundaries, plus failed table growth with
  unchanged entries/source view and successful recovery;
- the real public VM verification/execution and C translator path for an
  advisory-bearing module, with both results equal to 42 and exact generated
  C equality before compiling/running with ASan/UBSan;
- focused conversion/metadata checks under ASan/UBSan with leak detection;
- 365 existing bridge checks, 20 public v2 loading checks, 210 canonical
  disassembly checks, and 274,416 VM checks;
- 959 owned assertion lifecycle checks and both Forth SEE host-manifest and
  examples-library loading methods;
- a genuine canonical native compiler seed build, help, and hello bytecode
  publication, verification and execution after integration.

At the earlier compiler checkpoint I also pass the 43 whole-module codec checks
and existing passive/ownership contract suites. Logs remain under
`/tmp/nanolang-advisory-*`, including `final-focused.log`, `integrated.log`,
`integrated-adjacent.log`, `asan-integrated.log`, `seed-integrated.log` and
`seed-integrated-hello.log`. The seed uses the existing explicit 60-second
shadow budget; no shadow is disabled.

I do not infer runtime authority from these advisory strings. Local-name
production, general frontend fact schemas, general/restricted compute profiles,
structured control reconstruction and the second high-level surface remain
required roadmap work. This evidence does not establish full release acceptance.
