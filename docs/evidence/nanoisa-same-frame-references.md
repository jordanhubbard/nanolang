# My same-frame reference execution evidence

I implement the [bounded root-reference contract](../NANOISA_SAME_FRAME_REFERENCES.md)
under MAC `task_e31a51fc661f4102b68ad81432369a1d`. My implementation checkpoint is
`a9734e63`, based on merged owned-transfer runtime `0b8ebee8`. I allocate vacant
primary bytes 0x16–0x1b without renumbering existing instructions.

I use reference slots separate from values. Region and exact owner identities
are checked by affine CFG analysis before either backend admits execution.
NanoVM retains descriptors across core yields, resolves local indices after
stack relocation, and clears them on completion or terminal core error. A
refused nested host invocation preserves its suspended caller. Native C reads
and writes the same owner fields directly. Neither backend copies a borrowed
record and writes it back.

I measured these Linux ARM64 gates on 2026-09-17:

- 1,336 reference checks: nine paired VM/native programs, sixteen unsupported
  lifetime/overlap/identity decisions, instruction encoding and canonical
  reconstruction, core yield/resume with live references and stack relocation.
- 29 VM heap-allocation failure/recovery checks, including failure while an
  exclusive reference is live. Every unwind leaves no descriptor or owner.
- All nine emitted native programs pass ASan/UBSan and allocation-failure
  cleanup checks. Serialized artifacts verify and execute through the normal
  CLI; regenerated native source matches the in-memory translator output.
  My CLI uses int results as exit codes. The runtime harness separately checks
  exact bool/u8 tags and values.
- 1,051 prior owned-runtime checks and 32 allocation checks; 184/272 owned
  transfer, 300/419 affine bytecode and 157/182 affine state checks.
- 272,624 VM checks under ordinary and computed-goto dispatch. The reference
  suite also passes all 1,336 checks with computed goto.
- 2,801 ISA and 33 schema checks. The schema route inventory names all six
  specialized verifier operations.
- Instrumented VM, heap/cycle collector, affine state/dataflow, verifier and
  native generator pass the 29 allocation and 1,336 reference checks with
  ASan/UBSan and leak detection.
- The actual canonical compiler native seed builds against the nanoisa host
  module, runs help, emits hello bytecode, and verifies/executes that artifact.
- 2,418 native translation and 1,092 shape checks, seventeen LLVM and eleven
  Wasm translator methods pass before the main integration below.

I preserve logs under `/tmp/nanolang-reference-`: `paired-final.log`,
`integration-final.log`, `computed.log`, `asan.log`, `seed.log`,
`seed-hello.log` and `translators.log`. The initial combined integration log
includes an over-specific missing-metadata error-code assertion in my new
fixture; `paired-final.log` records the corrected non-success contract. My
first CLI assertion also assumed bool/u8 results set process status; the
corrected harness distinguishes that CLI policy from exact runtime values.

I keep caller-place alias substitution, nested reference paths, reborrows,
reference parameters/results, floating reference fields and source producer
admission outside this checkpoint. They remain required work under my affine
and borrow parents. I do not close full v5.1 acceptance or the publication hold.

## My integrated checkpoint

I integrate main `79968058` through PR569 at `74cb9d42`. I preserve both additive
Makefile and roadmap entries. The merged common `vm_return_values` path clears
my activation before return-value extraction; the owned eligibility contract
still requires explicit returns.

On that integrated source I repeat 1,336 reference and 29 allocation checks,
1,051 owned-runtime and 32 allocation checks, 2,801 ISA and 33 schema checks,
272,624 VM checks and all nine ordinary implicit-return methods. Computed-goto
repeats 272,624 VM and 1,336 reference checks. ASan/UBSan repeats 29 allocation
and 1,336 reference checks with leak detection. The canonical native seed is
rebuilt from integrated source; help, ordinary hello emission, verification
and execution pass again. I retain `restack.log`, `restack-computed.log`,
`restack-asan.log`, `restack-seed.log` and `restack-hello.log` under the same
`/tmp/nanolang-reference-` prefix. This final documentation update does not
change compiler or test source.

I subsequently integrate main `d8a88cca` through PR571 at `ea8f371b`, preserving
PR567 projected-global support and the completed LLVM implicit-return roadmap
row. Only that additive roadmap row conflicts. My VM, affine, owned-native
and verifier source is unchanged from `74cb9d42`; I repeat the affected
reference target (1,336 reference plus 29 allocation checks) and all 33 schema
checks. `/tmp/nanolang-reference-final-restack.log` records that final gate.
