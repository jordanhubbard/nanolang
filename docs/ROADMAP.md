# My Roadmap

- [x] I index native root membership and mark each owner once per collection (MAC `task_869e7e8e12e946d2a3ffc9cac6e16882`), preserving traversal order, collection points and lifetime guarantees. My [scaling and sanitizer evidence](evidence/native-root-tracking-cost.md) separates this lookup repair from full bootstrap convergence.
- [x] I guard native loop collection with collectible-owner allocation debt (MAC `task_5928905033844a4a8fb498d63cb68c39`), preserving fresh mutable-root tracing, forced collection and bounded retention. My [measured scan and lifetime evidence](evidence/native-collection-debt.md) records 9998 scans reduced to two and the distinct record-capacity boundary below.
- [x] I grow native AOT record arrays beyond 256 elements with checked allocation, alias preservation and complete teardown (MAC `task_617006c46f9746da9694c6a4e0a0ceaf`). The full native compiler now reaches this existing limit while tokenizing; my retained 257-record fixture passes VM and emits byte-identical failing native C with unchanged and debt-scheduled translators. The 50-method functional/sanitizer suite passes; full native self-compilation remains blocked by the separate parser stack task.
- [x] I reduce generated native compiler record-local stack use with invocation-owned heap frames (MAC `task_4c5aeade9fa944dd80eebfbd1ae91072`). I preserve historical parser crash evidence and the ordinary stack limit; my [static and functional evidence](evidence/native-record-local-frames.md) records the bounded repair without claiming full native self-compilation.
  - [x] I measure retained native compiler frames without executing failing inputs: GCC 13.3 on AArch64 reports 397216 bytes for `parse_primary` and 123664 for `parse_expression_recursive`; 4360-byte records dominate their local storage.
  - [x] I allocate record locals in invocation-owned heap storage, preserving value copies, root addresses, return snapshots and simultaneous self-tail argument staging.
  - [x] I verify native semantics, record/map/string lifetime and tail calls, measure reduced static frames of 17904 and 19024 bytes, and pass both ordinary compiler product compilation gates. I preserve the initial independent shadow-deadline failure under `task_628759a2daf743b9bf13c9a7fea2ced0`.
  - [x] I publish the measured storage and acceptance boundary; full native self-compilation remains a separate acceptance result.

- [x] I support verified `JMP_TRUE` in native translation while preserving VM truthiness semantics (MAC `task_211f22859e164287a07a63cba74ace5b`). My bounded companion covers control-flow discovery, local initialization, taken-edge transfers and loop-root collection, with VM/native true/false branch effects and actual float-format artifacts before acceptance. My original record-local fixture exposed the classifier refusal; my [true-branch evidence](evidence/native-jump-true.md) now records four focused methods,34polarity cases,2390native checks and1092shape checks, plus both actual float-format artifacts in VM and sanitized native products.

- [x] I preserve tagged native map globals, checked operations and lifetime roots (MAC `task_af839ea3c3d14ebfa3191a0322f08298`), with VM/native and sanitizer regressions. Whole-record globals remain on `task_95796f5f49564ed4a911fd05a1aac5b4`.
- [ ] I reconcile declared raw hashmap key/value tags between VM acceptance and native rejection before changing either policy (MAC `task_b19f8bf0527d4a33911be26706629616`).
- [x] I preserve forward projected string branches at native stack joins in the fresh compiler, with the strengthened compiler product gate (MAC `task_55002ea4e4c64f80a6ba70b7f147ebef`). I reproduced the same `check_let_statement` failure with unchanged main `b3f79449` and repaired it with checked join storage. Backward joins and the separate selfhost-emitted artifact remain below.
- [ ] I converge tagged string storage across backward native stack edges before widening an already classified loop header (MAC `task_ea3c8acd272a49669bd6ae6aa75cdf49`). I preserve the VM-positive loop fixture and native refusal separately from the forward compiler join repair.
- [x] I remove the false `purity_node` to `purity_call` parameter conflict by clearing reused signature-pool bytes for undeclared function/import tags (MAC `task_04376d3e430c478d968af69e26543a0f`). My rebuilt selfhost artifact preserves unknown declarations, round-trips identically, runs VM help, and passes the original native guard. Its next native reconstruction blocker remains below; I do not claim its native compiler product complete.
- [x] I reconstruct diagnostic record arrays in the real selfhost compiler using its truthful nominal type counts (MAC `task_250092bed54749ad988f06af5b88c228`). My canonical frontend emits compiler bytecode that translates to native, runs help, and emits a hello NanoISA product executed in both runtimes. I preserve the separate seeded bridge and [exact proof boundary](evidence/selfhost-native-compiler-product.md); byte-identical bootstrap convergence remains open.

I keep this document to outline my development journey.

I execute active work from top to bottom. Before implementation begins, I add
the work here as checkable items, including its tests and documentation. I mark
an item complete only after I have verified it. MAC tasks track ownership and
execution; this document records product direction and order.

When I discover a defect, an overclaim, or required work while executing this
document, I add a concrete checkbox here in dependency order before I continue.
Chat is not the ledger. A defect I already fixed in the same session still gets
an `[x]` item so it stays in product history.

**The full 5.0 roadmap is my v5.1.0 completion contract.** On 2026-09-16
my user explicitly selected every 5.0 obligation, including NanoISA-only
compilation and matching Stage 1/Stage 2 bytecode. My published
[`v5.0.0`](https://github.com/jordanhubbard/nanolang/releases/tag/v5.0.0)
contains tested language/runtime repairs, but it does not complete that
architecture. `docs/RELEASE_5.0.md` records that historical release;
`docs/NANOISA_ONLY.md` and the unchecked 5.0 milestones below govern the
remaining work. I preserve the published tag and do not redefine an unfinished
gate as complete. My user assigned all remaining work to v5.1.0 on
2026-09-16, superseding the earlier `v5.0.1` candidate name. I hold release
publication until these gates pass; the native map
lifetime repair alone does not satisfy this scope. Phase 22 / 6.0 remains separate.

## Active Execution Queue

- [ ] I define explicit native float-to-int conversion for finite values and exceptional/range boundaries before matching VM behavior (MAC `task_b927827f37734658bce360d7ecf913aa`). Static float `CAST_INT` is already refused; tagged float transport must also refuse instead of silently returning zero. I retain that boundary in the typed-float regression.

- [x] I lower typed F64 arithmetic, negation and comparisons in native AOT with strict operand tags, boolean result tags, signed zero and the VM's zero-divisor result (MAC `task_fd4c63cf9f3e46f09ece380ce00c7a58`). The actual paired scalar fixture also requires float global transport and float `CAST_STRING`; I preserve current VM formatting rather than changing the separately tracked source-builtin formatting policy.
  - [x] I implement checked float storage/transport and typed operations without substituting generic opcodes or admitting implicit integer conversion.
  - [x] I verify all typed operations, signed-zero/global/function transport, malformed operand tags and the paired real scalar fixture in VM/native, with focused sanitizer and adjacent native gates: four methods including33bad-tag cases,2390native checks and1092shape checks pass. My [scalar evidence](evidence/native-typed-floats.md) distinguishes generic NaN ordering, typed IEEE comparisons and the remaining conversion boundary.

- [ ] I retain nominal record identity through nested empty-array append results. My C-seed checker currently refuses `array_push (array_push [] Item {...}) Item {...}` before VM lowering; I retain the fixture separately from supported scalar append inference (MAC `task_439297c5a6934857a90cbec93bb7958d`).

- [x] I infer the supported element type of an unbound empty-array append from its value, preserve nested append types and source evaluation order, and reject mismatched established receivers. I require bytecode parity and VM/native execution before advancing my complete compiler-shadow closure (MAC `task_d5ed194093434b5cbfc2e3ec6bc2d37a`).
- [x] I release my owned Environment import-tracker array and container during teardown (MAC `task_73162d300ee44cb4a16d723bd49beeec`). The symbol-index LSan fixture exposes a pre-existing 208-byte leak; I retain the failure and keep entry-payload ownership separate.

- [x] I restore margin within my ordinary compiler-shadow deadline without changing its value or selected tests (MAC `task_628759a2daf743b9bf13c9a7fea2ced0`). One diagnostic sample completes 823 shadows in 9.953 seconds; bounded stack samples identify repeated variable-name scans, including misses before ordinary function calls.
  - [x] I retain the timeout/pass evidence, a 60-second diagnostic timing sample, and 24 stack samples without changing host profiling policy. These samples select a cost to investigate; they do not prove every timeout's cause.
  - [x] I audit insertion, same-file lookup, lexical rollback, reused slots, imported constants and environment teardown before indexing symbol names. Normal declarations synchronize before insertion; the separate imported C-constant append will explicitly invalidate the index.
  - [x] I index names without changing reverse-most selection, owner/source filtering or value lifetime; allocation failure retains correct linear lookup.
  - [x] I verify scope reuse, metadata preservation, imported names and function-variable shadowing, then compare unchanged-source full shadow timing at the default deadline and publish the [measured boundary](evidence/interpreter-symbol-index.md). All 826 shadows complete in 2.392 seconds on the same source whose baseline reaches the default deadline; fresh bootstrap and imported callback checks pass.


- [ ] **Paired call-scoped resource borrows.** I implement the existing
      `&T` / `&mut T` contract with retained annotation identity and explicit
      borrowed call arguments. I first preserve syntax and metadata, then
      enforce shared/exclusive access and call-argument overlap, then prove
      observable shared reads and exclusive mutation on C seed, Stage 1 and
      Stage 2. I reject moved, escaping, stored and unsupported callback
      borrows before publication; I do not pass exclusive mutation by value.
      My positive/negative matrix and fresh bootstrap precede closure. This
      is a prerequisite to verified NanoISA ownership facts, not that IR gate.
      MAC `task_71821d84befc46e198795122c1112a27`.
      - [x] I retain shared/exclusive named-parameter annotations in both
        parsers and explicitly reject them before unsupported lowering. My
        parser-copy, paired refusal/retention and ordinary ownership controls
        pass; `evidence/call-scoped-borrow-annotations.md` distinguishes the
        diagnostic bootstrap budget from the open default-deadline task.
      - [ ] I first admit explicit shared borrows of fixed resource records
        with scalar fields, passing the caller's address to native code and
        preserving interpreter identity. I prove repeated reads, forwarding,
        post-call consumption and refusal of consumption, mutation, escape or
        a later argument that moves a borrowed owner. Unsupported generic,
        aggregate, foreign and callback shapes remain explicit boundaries.
      - [ ] I then add exclusive pointer mutation with observable caller
        updates and paired overlapping shared/exclusive argument rejection.
        Broader projections and shape support follow their own paired gates.
- [x] I lower my declared NanoISA file assembly and disassembly artifact contracts with exact parameter and result validation, and execute their real module shadows in VM and native products (MAC `task_f5f873fccfff4b5b88f14f4d825ba3b4`).

- [ ] **Concrete native specialization closure.** I discover union instances
      reachable only through substituted payload fields, including checked
      cyclic and expanding-type boundaries. MAC `task_98dab2489ab749e7b9053f944b3fe489`.
- [ ] **Direct nested native match identity.** I retain the concrete selected
      payload type when matching `payload.value` directly; my current emitter
      produces `nl_UNKNOWN`. MAC `task_7bc727794375435ca72e6fef466ff161`.

- [x] **Self-hosted native union contexts.** I substitute nested generic payload
      types and initialize declared union globals in source order alongside
      guarded globals, preserving native shadows and executable payload values.
      MAC `task_85a8db6e186440eaad80442bfc133dd8`; evidence:
      `docs/evidence/selfhost-native-union-contexts.md`.

- [x] **Primitive List record fields.** I preserve `List<int>` and
      `List<string>` keyword elements and their runtime typedefs during
      native token lowering. MAC `task_f0a59def9afe4a08ae63c55d4ba22109`;
      evidence: `docs/evidence/native-primitive-list-fields.md`.

- [x] **Record-valued call pushes.** I capture the receiver and record-returning
      value once in source order before taking its address for native
      `array_push`. MAC `task_6ce6ece22ae34698a178bf29ee04412b`; evidence:
      `docs/evidence/native-record-array-append.md`.

- [x] **Mixed native nominal definition order.** I order record and union
      definitions by their by-value dependencies, preserve pointer-backed
      collection boundaries, and diagnose unsatisfied layout cycles. I test
      both dependency directions and alternating chains before enabling
      selected resource payload execution in both self-hosted stages. MAC
      `task_68a6b53f768245bfacfc79b6db78b621`. Fresh bootstrap and ten
      source cases in both self-hosted stages pass; I preserve anonymous
      primitive-list runtime typedefs. Evidence:
      `docs/evidence/native-nominal-definition-order.md`.

- [x] **Native module artifact facade adapter.** I accept only the exact
      artifact-bound `nlc_module_artifact(string) -> string` signature,
      snapshot its borrowed result before another call, and retain rejection
      of unsupported namespaces, import kinds and types. Companion to MAC
      `task_81e682a57a3d431e845b0f41f140352e`. Exact contract tests and
      a real immutable-generation native call pass; see
      `docs/evidence/native-module-artifact-adapter.md`.

- [x] **v5.1.0 concrete union payload metadata.** I preserve nested C payload
      TypeInfo before implementing selected-variant ownership transfer. MAC
      `task_1f64c9b88a5248dcbda2258dcbee99f7`.
      - [x] I record nested ordinary execution and resource-rejection baselines.
      - [x] I preserve metadata through parsing, declaration registration,
        nominal/formal binding, module copies and teardown with explicit ownership.
      - [x] I use concrete payload metadata in supported native matching/emission
        and retain existing conservative generic-resource rejection. Fixed nested
        resource classification remains explicitly open below.
      - [x] I pass paired frontend execution/rejection and relevant metadata,
        bootstrap and instrumented checks before integration.
      Selected-variant ownership transfer remains a subsequent open obligation.

- [ ] **Classify fixed nested resource union payloads.** I reject unsupported
      ownership in `Owners.Some { values: array<Handle> }` as consistently as
      `Box<Handle>` after preserving complete payload metadata. My C seed
      previously accepted an `abandon(Owners)` declaration. My paired
      rejection and ordinary controls now pass; I still reject collection ownership. MAC
      `task_e1ce4d21563d4fb3bbb998e30fc9652f`.
      - [x] After merged payload metadata PR411, I classify nested fixed array
        payloads using declaration identity and a least fixed point, preserving
        ordinary cycles and generic formal shadowing.
      - [x] I reject unsupported resource collection payloads consistently at
        ownership boundaries, including pass-through signatures, and execute
        ordinary union-array controls in C and both self-hosted stages.
      - [x] I retain the owned-match guard and record selected-variant transfer
        as the separate open task below; this classifier does not implement it.

      Evidence: `docs/evidence/fixed-union-resource-collections.md`.

- [x] **Preserve self-hosted union array-literal context.** I retain concrete
      variant field annotations during recursive literal emission and share
      token-aware generic substitution with my checker. The C seed and both
      self-hosted stages execute fixed/generic/nested record-array payloads and
      reject wrong nominal payloads without replacing prior output, after
      prerequisite PR423 and PR430. MAC `task_de0fb8219008442db8bc83e2a79eba26`.
      Evidence: `docs/evidence/union-literal-context.md`.

- [x] **Preserve direct-call match identity.** I infer the declared union return
      type for statement and expression scrutinees without evaluating the call
      again. My ordinary and owned exactly-once controls pass the C seed and
      both self-hosted stages after fresh bootstrap. MAC
      `task_7eb80289f352425aa2abd26af55d0ffc`.

- [x] **Parse complete nested generic arguments.** My C annotation parser
      preserves the inner arguments in `Box<Result<int,string>>`. I retain
      recursive argument `TypeInfo`; 64 parser checks, focused parser sanitizer
      checks, fresh bootstrap and 16 paired generic-affine methods pass. Native constructor
      context is repaired in the C seed; self-hosted nested/global emission
      remains task `task_85a8db6e186440eaad80442bfc133dd8`. MAC
      `task_8178b6b71fe147bd851629713e7be14d`.

- [x] **Retain instantiated ownership metadata.** I preserve complete concrete
      C `TypeInfo` and matching self-hosted generic identities through ownership
      bindings, aliases and calls. I substitute ordinary `Result<T,E>`/`Box<T>`
      fields before resource classification, with paired positive and negative
      controls and lifetime checks. Generic owned matching remains rejected
      until this prerequisite is verified; resource collections remain rejected.
      I do not infer complete tuple/row/function substitution from copying their
      metadata. Fresh bootstrap, 12 paired methods, 80 adjacent methods and
      240 sanitizer lifetime checks pass; evidence is in
      `docs/evidence/instantiated-ownership-metadata.md`. MAC
      `task_d329989e8acf43149c38d6a998bd730f`.

- [x] **Keep the resource-flow allocation boundary test aligned with ownership
      metadata.** I pass an explicit absent type tree for the deliberately
      metadata-free overflow fixture after `own_add` gained concrete `TypeInfo`
      context. The focused allocation target and all 12 instantiated-ownership
      methods pass. MAC: `task_a822c3af6ff10226a2dde139a8ac1d7c`.

- [ ] **Restore my Darwin bootstrap within the ordinary shadow deadline.** At
      source `1277bce2`, the first self-hosted compiler finishes under an
      explicit bounded 300-second budget but reproducibly exceeds my default
      10-second shadow deadline. I will diagnose the regression or justify a
      platform budget without weakening mandatory dependency shadows. MAC:
      `task_0ea74f24799d9c9604bdf8abc7250d3d`.

- [x] **Substitute generic selected-variant patterns.** I retain concrete
      arguments through complete-field validation, hidden payload captures and
      inferred projections. Ordinary integer, string, empty and nested selected
      patterns execute across my C seed and both self-hosted stages; malformed
      fields and generic owned transfer remain rejected. Source `81cc5ebd`
      passes fresh bootstrap, 59 new/adjacent methods, C typechecker tests
      and 160 sanitizer lifetime checks. Evidence:
      `docs/evidence/generic-selected-patterns.md`. MAC
      `task_bbda7f126bda403aa74034a762930f24`, after instantiated ownership metadata.

- [x] **Emit concrete aggregate match results.** I retain the checked nominal
      result of contextual union constructors when declaring a match result,
      and use valid scalar/aggregate initialization in both native emitters.
      I require ordinary, generic and scalar executable controls across my
      C seed and both self-hosted stages before generic ownership acceptance.
      MAC `task_ce96caed26b843ecb1365def0c733b58`.

- [x] **Transfer concrete generic selected payloads.** After ordinary generic
      pattern validation, I substitute only the selected arm's fields before
      assigning ownership obligations. I retain declared initializer, parameter
      and return contexts while transferring constructor payloads; I do not
      reconstruct missing arguments from sibling variants. I require paired
      executable resource, ordinary and empty-arm controls plus drop, duplicate-consumption,
      use-after-move and branch-join rejection. Conditional statement returns and match-expression
      constructor results pass the bounded native gates recorded in
      `evidence/generic-selected-ownership.md`.
      I preserve guards for unresolved
      payload shapes, collections and incomplete function signatures. Global
      ownership and direct nested field-scrutinee inference remain separate. MAC
      `task_08428ceb1d674de49383aab1ba9a78c8`, after
      `task_bbda7f126bda403aa74034a762930f24`.

- [x] I retain owned parameter and return annotation trees inside my C
      function signatures, including deep copies and parser/interpreter teardown
      (`task_42ad1f3109374581a3f88f5dc58c049c`). This representation prerequisite
      does not close the paired callback execution and ownership boundary below.
      See `docs/evidence/function-signature-storage.md`.

- [x] I preserve concrete union arguments when I render selfhost function-value
      signatures and array element types. I test nested ordinary types and
      distinguish this identity repair from callback lowering and ownership.
      MAC `task_8ef982b488d8421fb7e5b5ea280e7d2c`; evidence in
      `docs/evidence/selfhost-callback-identity.md`.

- [x] I compare retained C callback annotation trees, reconstruct owned
      signatures from declarations and emit concrete native parameter/result
      layouts (`task_cf555f2a672e43d9921ca44b817ec631`). I share the deep-copy
      and structural comparison helpers. Matching ordinary callback execution
      and wrong named/local/forwarded signature rejection are bounded gates;
      complete indirect-call and qualified-call contexts, serialization and resource boundaries
      remain in `task_e05a42e2e09b47cc9c53fa6923eeeaef`.

- [x] I retain callback call-site context across indirect and qualified calls.
      I compare complete arguments, preserve checked signature/result metadata
      after lexical scopes end, and require matching execution plus rejection
      before publication (`task_3ca0e46fbbc64aa8bc39bfdaf65b8833`). Resource
      transfer remains a separate boundary in the parent task.

- [ ] **Retain generic function-value signatures.** My C `FunctionSignature`
      now owns complete annotation trees. I still require comparison, lowering,
      serialization and paired ordinary
      generic callback execution plus conservative resource rejection before
      claiming complete function-value ownership. This representation gap does
      not establish an executable ownership escape. My `fn()->Box<int>` probe
      fails native emission in the C seed and loses generic identity in both
      selfhost stages; NanoVirt publishes while later printing missing-signature
      errors. I require each boundary to retain metadata or fail before publication.
      MAC `task_e05a42e2e09b47cc9c53fa6923eeeaef`.

- [x] **Classify concrete generic payloads retained inside records.** I use
      full retained field annotations and substituted union arguments in the
      same fixed point as nominal and public TypeInfo queries. My paired
      resource/ordinary/phantom controls, nested chains, cycles and collection
      guards pass. Fresh bootstrap, 68 adjacent methods and 520 ASan/UBSan
      lifetimes are recorded in `docs/evidence/global-resource-boundary.md`.
      This establishes classification, not native record layout or global
      ownership support. MAC `task_ef807591eb104cc8b664cd55581ec505`.

- [x] **Guard unsupported generic resource callback signatures.** I inspect
      complete callback parameter/result annotations and reject generic owners
      and resource collections before publication. My ordinary and phantom
      callback values remain copyable; I check their transfer contracts
      separately. All 13 boundary methods pass through C seed/Stage 1/Stage 2,
      including executable nongeneric resource parameter/result controls and
      preserved-output rejection. Fresh bootstrap and 78 adjacent paired
      methods pass; `docs/evidence/resource-callback-boundary.md` records exact
      checkpoints and limits. This guard does not implement generic callback
      ownership. MAC `task_f05070eff34a4b169318fe949da45630`.

- [x] **Retain concrete union identity when reading record fields.** I retain
      complete annotations and declaration-aware union kinds for typed reads
      and direct matches. I reject wrong concrete arguments and wrong union
      declarations before native compilation. My C seed and both self-hosted
      stages pass nested and multiargument controls. Evidence is recorded in
      `docs/evidence/native-generic-record-fields.md`. Selected union payload-field
      origin remains separate (`task_7bc727794375435ca72e6fef466ff161`).
      MAC `task_9ad126f2c5a61aadfa672f29134aa9ec`.

- [x] **Emit ordinary concrete generic union record fields.** I preserve
      complete field spelling, discover field-only instantiations and order
      concrete by-value dependencies. Empty/nonempty constructors, nested
      records, imports, forward declarations and typed payload reads execute
      through my C seed, Stage 1 and Stage 2. Fresh bootstrap, 57 paired methods,
      25 integrated field/callback methods and 280 ASan/UBSan metadata lifetimes
      pass; details and limits are in `docs/evidence/native-generic-record-fields.md`.
      Global lifetime and full ownership remain separate. MAC
      `task_6e5fc4b3cd4f9e25eea18e792de0f2b0`.

- [x] **Define my unsupported global resource boundary.** I reject owning
      global declarations explicitly in both frontends before artifact
      publication. Mutable/immutable, inferred, nested/generic and imported
      rejections preserve prior output; ordinary scalar/union globals and
      local owners remain executable controls. At `3168885d`, a fresh bootstrap,
      16 affected methods and the complete C typechecker suite pass after
      callback identity integration. I do not implement global lifetime or
      transfer semantics here; whole-record global emission remains separate
      under `task_95796f5f49564ed4a911fd05a1aac5b4`. Evidence:
      `docs/evidence/global-resource-boundary.md`. MAC
      `task_8afaef937f934a6e9919e41b91b7a41c`.

- [x] **Transfer selected owned union payloads.** After fixed collection
      rejection, I implement exhaustive unguarded nongeneric matching with one
      scrutinee move, per-arm payload obligations and complete qualified variant
      destructuring. I require both frontends to accept resolved payloads and
      reject ignored fields, repeated use, unresolved exits and incompatible
      joins. I keep unsupported guarded/generic/tuple/borrow cases rejected and
      do not infer NanoISA ownership verification from native execution. MAC
      `task_c17b55115379414980609a5d867ccad1`, dependent on
      `task_e1ce4d21563d4fb3bbb998e30fc9652f`.
      - [x] I first preserve a qualified variant pattern name in both parsers
        and validate its complete field set against the selected arm identity.
        Five methods pass across the C seed and both self-hosted stages,
        alongside a fresh bootstrap and 44 adjacent methods. That prerequisite
        checkpoint retained owned-match rejection; my later transfer checkpoint
        is recorded in `docs/evidence/selected-variant-ownership.md`: fresh
        bootstrap, 21 paired methods and 58 adjacent methods pass.
      - [x] I move an exhaustive unguarded nongeneric scrutinee once, classify
        each selected payload independently and transfer its complete fields.
      - [x] I test ordinary and owned arms, empty variants, nested resources,
        unresolved fields, repeated use, outer joins and terminating exits on
        the C seed and both self-hosted stages before relaxing the owned guard.

- [x] **Diagnose unsupported tuple ownership before emission.** For
      `Bundle<T>.Some { value: (T,int) }` with `Bundle<Handle>`, I reject
      resource ownership before emission in the C seed and both self-hosted
      stages, including nested `Box<Bundle<Handle>>`. My C classifier retains
      conservative resource-argument rejection for unresolved payload shapes;
      my self-hosted classifier recognizes resource leaves in tuple spelling.
      Both negative methods preserve prior output. This does not implement
      general tuple substitution or ownership transfer. MAC
      `task_bcd773ad3c084ce099a3da5aef682fef`.

- [ ] **Release legacy union metadata allocations.** My payload lifetime check
      exposed 245 bytes retained by existing registered field-name/formal arrays
      and the environment import tracker. I establish borrower ownership before
      freeing these and require LeakSanitizer evidence; scoped ASan checks with
      leak detection disabled do not satisfy this item. MAC `task_00c47a5d65d04c48914864ec0de553d6`.

- [x] **Adopt v5.1.0 as my full-roadmap release target.** I preserve the
      published `v5.0.0` tag and move every unreleased `v5.0.1` contract,
      version identifier, candidate note and presentation obligation to
      `v5.1.0`. My package metadata, public compiler version, changelog,
      release-facing prose and focused release tests must agree before I call
      the rename complete. The publication hold follows the renamed release.
      MAC `task_7bad6bb81bdc3eef2e9a8bf0ba52f2ff`.
      My six version methods and release-workflow checks pass, as do fifteen
      markdown/document-pair methods. The focused review below now satisfies
      my documentation-freshness gate for post-v5.0.0 language changes; I keep
      it distinct from this completed identity change.

- [x] **Refresh my v5.1.0 release documentation after frontend and verifier
      changes.** I review the contributor contract and selected user-guide
      lessons against every post-v5.0.0 change to my C parser, typechecker and
      NanoISA verifier. I document only implemented behavior, compile every
      checked guide example, build every guide edition and require the release
      documentation freshness gate to pass without an override. My contributor
      rules now name passive-metadata and closed-purity gates; my guide records
      checked purity, typed global collection context and qualified nominal
      identity. The freshness gate, 24 release-document tests, 40 checked
      snippets and all six rendered guide editions pass. MAC
      `task_7bad6bb81bdc3eef2e9a8bf0ba52f2ff`.

- [x] **Establish explicit v5.1.0 GitHub release scope.** I inspect every open
      issue and pull request, create `release/v5.1.0`, and apply it to the only
      release-scoped open item, PR #414. My baseline finds zero open issues and
      six open pull requests; five old drafts are visible but unscoped, so I do
      not close them to make the gate green. I record the exact queries and
      require a fresh zero-blocker audit before tagging. Evidence:
      `docs/evidence/v5.1-release-readiness.md`. MAC
      `task_7bad6bb81bdc3eef2e9a8bf0ba52f2ff`.

- [ ] **Reconcile completed v5.1.0 roadmap and MAC records.** I compare active
      checkboxes with authoritative task states and merged commits, close only
      work targeted to this session, and leave genuinely unfinished follow-ons
      explicit. I verify the Darwin parser portability gate before closing its
      stale task and correct release-tree restoration evidence after its owner
      closes the ledger row. I have reconciled the native opaque-null duplicate
      and the completed compiler AOT bridge below; the remaining active rows
      still require the same evidence check. MAC
      `task_7bad6bb81bdc3eef2e9a8bf0ba52f2ff`.

- [x] **Preserve opaque null arguments in native call snapshots.** I retain my
      accepted integer-zero spelling for an opaque null when foreign call
      arguments are evaluated once into ordered temporaries. I reject nonzero
      integers, compare my C seed and self-hosted native emitters, require
      strict C compilation, and restore the current quick GLUT gate. MAC
      `task_64b0d006cff713ffa197dcec1d22a894`. I cancelled
      `task_77dab245060bc54e6f445bdde3431ec1` as its duplicate after attaching
      the merged implementation and fresh test evidence to the authoritative
      row.
      Six native call-order/null methods, both bootstrap stages, 22 self-hosted
      compiler cases, the paired native shadow emitters and the GLUT boundary
      gate pass. My broader quick gate advances through that repaired boundary
      and stops later at the separately owned VM example-purity failures under
      `task_7ee12d8737363c126a040fde905a7114`.

- [x] **Native compiler NanoISA product.** I require my compiler built from
      bytecode to compile `nl_hello.nano` with explicit `--emit-nvm`, execute
      that module in NanoVM, and translate/run the same product through
      `nvm2c` and `cc`. The default C-backend hello does not satisfy this
      acceptance. MAC `task_16425cd8a2404735a5cb246db12c5f59`; evidence:
      `docs/evidence/native-compiler-bytecode-product.md`. Compiler seeding
      still uses my C frontend; self-emission and fixed-point gates stay open.

- [x] **Restore compiler AOT artifact binding.** I preserve exact library
      bindings and typed adapters for the NanoISA facade imports introduced by
      my canonical bytecode route. I continue to reject unsupported imports and
      require generated native compiler execution rather than treating adapter
      unit coverage as the completed bridge.
      MAC `task_600074c773904b119b39bdafd85c07a5`. Evidence:
      `docs/evidence/compiler-aot-artifact-binding-gap.md`.
      My five exact facade adapters preserve borrowed string snapshots; real
      VM/native execution and malformed import contracts pass. After the boxed
      array, character-host, projected-global and tagged-local prerequisites,
      PR #426 compiles my compiler bytecode to strict native C, runs the
      generated compiler's help, compiles `nl_hello.nano`, and verifies its
      executable output on a fresh unchanged gate. I pass 2,215 native checks,
      1,092 shape checks and 22 tagged-local regressions. This closes the
      bounded AOT bridge, not my NanoISA-only bootstrap or Stage 1/Stage 2
      canonical `.nvm` fixed point.
      That PR #426 result is a historical checkpoint, not evidence that the
      gate remains green on every later commit. At `a16b2d8a`, the same gate
      rejected the new `tc_is_opaque_name` call because a projected optional
      string no longer satisfied its exact string parameter. PR #445 repaired
      that regression under MAC `task_031b36c92dbe44e49cea3888878d1963`.
      My fresh Darwin rerun then exposed and repaired the separate portable-C
      facade failure below before the unchanged gate passed again.

- [x] **Keep my compiler artifact facade portable under strict C11.** I replace
      the GNU-only `asprintf` path construction introduced with PR #439 by
      checked standard-C allocation and formatting. I require direct strict-C
      compilation on Darwin, the facade behavior suite, and the unchanged
      compiler-bytecode-to-native-to-program gate before closing MAC
      `task_3fd3b526b480593968aa41337fd3e00b`.
      Direct strict compilation and all four facade methods pass. After a fresh
      build of `nvm2c` and its AOT runtime, the unchanged full compiler gate
      passes in 25.320 seconds: bytecode translates to strict native C, the
      generated compiler runs `--help`, compiles `nl_hello.nano`, and its
      executable prints the expected output. This is still not my NanoISA-only
      fixed-point proof.

- [ ] **v5.1.0 concrete generic resource classification.** I substitute generic
      union payload types before classifying concrete ownership, distinguish
      ordinary instantiations from resource-bearing ones, and preserve declaring
      module identity. MAC `task_27d3d1bee3f84b5a9c1fc79e1f0c0748`.
      - [x] I record paired baseline cases and the parser/emitter prerequisites.
        Both self-hosted stages reject explicit `Box<int>.Some` constructors,
        leave match payload `T` unsubstituted, and emit `DynArray*` for a
        `Box<array<int>>` parameter/local. My baseline C parser dropped nested
        payload TypeInfo; the completed concrete-payload row above now retains
        it before general substitution.
      - [x] I reject unsupported generic resource obligations consistently in
        both frontends while executing ordinary-generic positive fixtures.
      - [x] I reject arrays of concrete generic resource payloads in both
        frontends, with ownership diagnostics and prior-artifact preservation.
      - [x] I preserve union formal-parameter scope when a same-named resource
        record exists, including nominal binding and paired ordinary/resource
        instantiations. My baseline `resource struct T` plus `Box<T>` fixture
        wrongly rejected `Box<int>` on all three stages. I also repair declared
        one-letter C record identities, empty union-arm metadata cleanup and
        imported concrete union parameter prototypes. I add ordinary self-hosted
        match traversal with ownership joins and visit ordinary union
        constructor fields while retaining
        resource-payload rejection and unsupported guards.
        I retain inline union constructor identity in C ownership lookup. All
        sixteen generic methods pass across three stages, including ordinary
        native execution and retained resource-payload rejection. My fresh
        bootstrap, 45 combined methods, 15 adjacent methods and instrumented
        C-only corpus pass; evidence records the earlier failed checkpoints.
      - [ ] I complete nested generic List/HashMap type preservation and paired
        ownership diagnostics while retaining conservative rejection.
      - [ ] I implement resource union payload transfer before admitting generic
        resource consumption; classification alone does not complete this gate.
      - [x] I bootstrap and pass the scoped generic/ownership/import checks,
        recording the initial missing-tool setup correction explicitly.
      - [x] I restore the required full compiler AOT gate through the separately
        recorded exact artifact-binding repair. Owned payload transfer above
        remains open; I do not claim the complete generic ownership contract.
        MAC task `task_27d3d1bee3f84b5a9c1fc79e1f0c0748` closed its bounded
        classification slice; the explicit transfer, nested collection and AOT
        follow-ons keep this umbrella unchecked.

- [x] **v5.1.0 passive metadata text preservation.** I preserve validated
      eligibility records through canonical disassembly and reassembly,
      reject malformed textual payloads, and compare canonical v2 bytes.
      MAC: `task_64dda3aeaab042df96a914ccd974209a`.

- [x] **v5.1.0 self-hosted enum metadata.** I retain ordered enum variant
      names and signed explicit/implicit values in my shared AST instead of
      discarding them during parsing. I verify all bootstrap stages and
      schema consumers before my NanoISA emitter uses these facts.
      MAC: `task_e66b50097fe343e3b78e6b750a5c7315`.

- [x] **v5.1.0 C-seed global initializer context.** I apply declared map and
      array types before checking top-level initializers, matching local
      declarations. I test typed constant maps, empty string arrays, and
      rejected mismatches through my native and VM frontends.
      MAC: `task_026e73d59e9e45b0b732b43883feea9a`.
- [x] **Checked boxed indices in native array updates.** I reconcile declared
      integer indices that inference boxes as dynamic values, preserving tag
      checks and rejecting wrong tags. Nine native cases preserve aliases,
      reject invalid tags/bounds, and advance compiler inference. MAC
      `task_959f620cc9294ef693072702f35ba44f`.

- [x] **Native compiler character-classification imports.** I bind eight
      emitted character imports through exact typed host contracts. I retain
      malformed-import rejection and test 232 VM/native boundary assertions. MAC
      `task_9e4e686f52bc426bbe4d2e694fa565bc`.

- [x] **Checked tagged native scalar-local assignments.** I retain tagged
      storage across local writes and reassigned parameters, preserving exact
      payload constraints and checked consumption. Twenty-two cases and the
      fresh compiler-bytecode/native/program execution gate pass. MAC
      `task_9d72bb9f29074e51ad826bc26e87b13a`.

- [x] **Native compiler projected global-store facts.** I resolve nested
      field shapes before deciding whether a global store is supported.
      Fifteen cases retain scalar/array stores and reject unsupported
      representations; twelve pass VM/native and sanitizer parity. MAC
      `task_3636ea1587cd41a88e4660abe94acb53`.

- [x] **Tagged native array-update bounds.** I reject negative and full-width
      out-of-range indices before narrowing dynamic array updates, preserving
      aliases and matching the checked VM contract. MAC
      `task_40b4cf1f065f49609cec599123f06557`. Twenty-one VM/native
      cases and the full compiler native bridge pass; see
      `docs/evidence/native-dynamic-array-update-bounds.md`.

- [x] **Boxed primitive-array argument inference.** I retain primitive-array
      identity and exact payload constraints across tagged call parameters.
      Eighteen cases cover caller order, tail calls, absence, aliases, and
      incompatible payloads; twelve pass VM/native and sanitizer parity.
      Full compiler execution remains open. MAC
      `task_e3a639dac7b54940ab11546c1ffa5eb9`.

- [x] **Raw NanoISA array-update index checks.** I reject noninteger and
      invalid-range indices in NanoVM while preserving valid alias mutation
      and ownership cleanup. I pass 272,485 VM checks under normal and fresh
      ASan/UBSan builds, plus nine VM/native parity cases. Native guards remain
      enforced. MAC
      `task_f64074441cf64f47b5f40ccefc78233c`.
- [x] **v5.1.0 self-hosted array field mutation.** I select native array setters
      from the complete receiver type, including record fields, nested fields,
      and returned arrays. I execute bool, string, float and nested-array
      mutations through all bootstrap compilers.
      MAC: `task_d32adbdff13241dc8ad9b0a889071352`.

- [x] **Current native compiler shape convergence.** I restore aggregate
      conversion solving for current compiler bytecode without weakening exact
      string constraints. Current and pre-read-fix translators both reject the
      same retained module before C emission. MAC `task_031b36c92dbe44e49cea3888878d1963`.
      I now separate inferred string parameter storage from exact producer
      shapes; focused parity and the fresh native compiler bridge pass.
      Evidence: `docs/evidence/native-projected-string-call-storage.md`.

- [ ] **NanoCore exporter sanitizer build.** I diagnose the GCC O1
      null-format warning in `sbuf_appendf`; default O3 sanitizer checks
      pass without warning suppression. MAC `task_c08bf552c04740daa060cbc5759eb19d`.

- [ ] **Source versus raw array-read policy.** I establish how source bounds
      panics and static native reads relate to raw/dynamic missing-read void
      results before changing either contract. MAC `task_c9561b3912a84a67a491cf9a78c1cc4b`.

- [x] **Tagged native array-read bounds.** I preserve missing reads as void
      while checking full signed indices before narrowing. I retain strict
      index tags and verify VM/native parity. MAC `task_7f62cc6dc9bc4d748a4fd8e78ca721f9`.

- [x] **Raw NanoISA array-read index checks.** I check integer tags and
      full-width ranges before reads, preserve native parity, and release
      consumed references on rejection. MAC `task_e4e7048e334e4b9d8eed2086f792bbcf`.
      I preserve missing reads as void; 38 VM/native cases and normal/sanitized
      VM suites pass. Evidence: `docs/evidence/checked-array-read-indices.md`.

- [x] **v5.1.0 module-owned affine record identity.** I compare same-named plain
      and resource records across modules on my C seed and self-hosted stages,
      preserve declaring-module identity through classification and flow lookup,
      and test aliases, nested fields, moves and unresolved owners. I keep
      unsupported generic and borrow boundaries explicit. MAC
      `task_d7c2aa83a60b43e792fb02ae7881e397`.
      Reproduction also exposes cross-module nominal registration and emitted
      type-name collisions. I preserve declaration identity and representation;
      my baseline and bounded native acceptance are recorded in
      `docs/evidence/affine-module-identity.md`.
      I preserve qualified record annotations in both parsers and exported
      source names while giving generated record layouts distinct identities.
      I verify same-module duplicate rejection separately from legal imports.
      Long module names also expose silent truncation in my native formatted
      output builder; I retain complete generated identities and test execution.
      All 49 compiler/order cases pass across my C seed and both self-hosted
      stages. Foreign collisions remain explicitly unsupported; union/enum
      identity, generic/borrow/capture rules and IR ownership metadata remain
      separate unfinished obligations.
      - [x] I include the new nominal-binding object in the manual NanoVirt
        wrapper link manifest and verify actual foreign-module wrapper execution.
        PR review exposed an unresolved `bind_nominal_records` reference.
      - [x] I zero-initialize all generic annotation metadata before nominal
        traversal. A `Box<Handle>` annotation exposed uninitialized row fields
        and a C-seed crash; I retain generic resource rejection and test ordinary
        scalar/array generic annotations under allocator perturbation.
      - [x] I preserve those qualified annotations without GNU `asprintf`;
        strict Darwin C99 must build the parser with overflow-checked portable
        allocation (`task_af9ed9b1529b4c65853e5669d4171126`).

- [ ] **Passive immutable input proofs.** I prove external input values before
      admitting their reads in eligibility metadata; declared signature tags
      alone do not close the current verifier's unknown call/local types.
      MAC `task_bf571298c10d4cc5a387b9f233ff3c40`.

- [x] **Passive eligibility metadata.** I implement a bounded scalar record
      codec and verifier for dependency graphs, stable serial order, instruction
      ranges and declared reads/writes, preserving records through v2 bridges.
      Calls and resource/effect operations remain refused until closed summaries
      support them. Frontend syntax and full passive conformance remain separate.
      MAC `task_abd0941bc1f34580845eb31ecd68a12c`. The scalar subset passes
      204 checks, VM/native execution and an instrumented verifier check.
      External reads remain refused pending their separate proof obligation.

- [x] **Map constructor diagnostics.** I count constructor type errors through
      my structured diagnostic path, rejecting direct untyped returns, nested
      calls, discarded calls and invalid arity before artifact publication.
      Both C and NanoVirt preserve prior output; typed maps still execute.
      MAC `task_306e982151974f42bd7c191ad017201a`. Two focused methods and
      the full C typechecker unit suite pass.

- [x] **Emitter build diagnostics.** I retain each native function parameter's
      declared nominal metadata instead of inheriting a same-name parameter
      from another function. Three regressions pass, real wrong-field access
      remains rejected, and the emitter builds without the spurious E004 errors.
      MAC `task_233ab81a36a0437db947eb11a8a4421b`.

- [x] **Emitter source preservation.** I reject input/output identity aliases
      before assembly or bytecode publication, including hardlinks and symlinks.
      I fail closed on identity errors and test unchanged source bytes.
      MAC `task_f430ed807f1743fdafc2f95cf67cb530`. Five driver tests pass,
      including both output modes, four alias forms and identity errors.

- [x] **Inline union match metadata.** I preserve the union identity of
      variant literals in expression and statement matches, and test payloads,
      side effects and lexical returns across interpreter, native and VM paths.
      All four cross-backend regressions and the typechecker suite pass.
      MAC `task_c28c39b66f704cf8af71827955c9edc0`.

- [x] **Callee snapshots in VM and interpreter.** I evaluate function values
      before argument side effects and retain the interpreter callee name until
      invocation completes. Both paths pass mutable-binding and computed-callee
      regressions; the previous VM fails both. The evaluator suite and all
      89 NanoVirt tests pass.
      MAC `task_8555af61281944eb9ac4ca9043849a94`.

- [x] **Standalone VM guest arguments for v5.1.0.** I separate VM options
      from guest arguments with `--`, expose the module path as guest argument
      zero, preserve empty arguments, and reject ambiguous module paths and
      unsupported daemon argument transport. I test compiler-style flags.
      Five regression methods pass, including a bytecode emitter invocation
      that produces and executes the Cut A addition module with result 42.
      This is driver coverage, not compiler bytecode fixed-point acceptance.
      MAC `task_d36776ab4ed143118c577f74f88ad3be`.

- [x] **Restore the accepted compiler baseline after PR #359.** I preserve
      the published `v5.0.0` tag and restore the released compiler, tests and
      documentation removed by the reviewed whole-tree replacement. I preserve
      the intended call-order fix separately and require a focused reapplication.
      PR #360 restored the accepted tree and merged as `4077b505`; its exact
      tree comparison, release-document check and nine release-gate methods
      passed. PR #365 then reapplied call ordering independently. Its targeted
      owner closed the previously stale MAC row on 2026-09-17. MAC
      `task_3665bd4322b35fb959ef77a46f761a0a`.

- [ ] **Complete the full 5.0 acceptance contract for v5.1.0.** I reconcile dated and
      duplicate ledger entries against current code, then complete the remaining
      5.0 work in dependency order. I distinguish a closed repair from a complete
      architectural milestone. MAC `task_7bad6bb81bdc3eef2e9a8bf0ba52f2ff`.
      - [ ] I close the ownership, purity and frontend semantic prerequisites
        specified by Phase 20, with a rule-by-rule shared conformance corpus.
      - [ ] I lower the complete compiler subset from `src_nano`, including
        imports, globals, required AST forms and declared foreign imports.
      - [ ] I emit deterministic verified `.nvm` v2 and provide a working
        compiler driver with guest arguments, shadows and diagnostics.
      - [ ] I compile and run the compiler through standalone `nvm2c` C11
        output with its declared host ABI and no embedded VM execution.
      - [ ] I build Stage 1 and Stage 2 compiler bytecode and require exact
        canonical `.nvm` equality before retiring the product C pretty-printer.
      - [ ] I finish Phase 20 metadata, reconstruction, profiles, ownership
        preservation, translators and VM/AOT/translator equivalence gates.
      - [ ] I close the other explicit 5.0 audits against measured evidence,
        update the documentation and run full platform acceptance before a
        subsequent release claims the complete roadmap.

- [x] **Inline union literal match dispatch.** I resolve the union type when
      matching a constructed literal directly, preserving branch values and side
      effects instead of emitting an unknown-union zero placeholder.
      This duplicated my completed inline-union metadata row above; its four
      cross-backend regressions and typechecker suite already pass. MAC
      `task_c28c39b66f704cf8af71827955c9edc0`.

- [x] **Mutable callee capture across backends.** I retain a named function
      value before its argument expressions mutate that binding. I repair my
      interpreter's invalid-name lifetime and my VM's replacement-callee call,
      preserving the original call and subsequent replacement behavior.
      This duplicated my completed callee-snapshot row above; the evaluator
      suite and all 89 NanoVirt tests already pass. MAC
      `task_8555af61281944eb9ac4ca9043849a94`.

- [x] **Native call argument ordering restoration.** I restore ordinary and
      module-qualified argument evaluation once in source order on the corrected
      release tree. I preserve effects, foreign ABI checks and existing lowering,
      test nested calls, mutable callee capture and temporary-name collisions,
      and submit only the focused patch.
      MAC `task_c897ac40d20b43669817b766dbe1c5a3`.

- [ ] **Typed null pointers in ordered native calls.** I retain the declared
      pointer type when I snapshot a null literal argument for a native call,
      without weakening left-to-right argument evaluation or callee capture.
      My strict macOS framework check currently rejects the integer temporary
      passed to `glfwCreateWindow`'s pointer parameters.
      MAC `task_64b0d006cff713ffa197dcec1d22a894`.

- [x] **Isolated NanoISA facade shadows.** I replace shared temporary fixture
      names with exclusive directories, retain assembly/load/error assertions,
      and verify concurrent execution and cleanup.
      MAC `task_a91eb62cd7ca44b1b9b6a1865a5f2300`.
      `make test-nanoisa-shadow-isolation` passes four concurrent compiler
      runs, fixture cleanup, prior sentinel preservation and unavailable-temp
      rejection without an output artifact.

- [x] **Closed purity summary foundation.** I preserve `pure fn` annotations in
      both frontends, derive transitive effects for known bodies, and reject
      false purity, mutable state, unknown calls, unsafe operations and resources.
      I compare shared positive/negative fixtures, including recursive closure.
      I reject mutable aggregate globals/parameters, including scalar-record
      wrappers around arrays: immutable handles are not deep-immutability proof.
      I also reject explicit extern declarations that collide with intrinsic
      names; a spelling alone does not establish a builtin effect contract.
      I remove the unsupported `pure` claim from `complex_exp`, whose `exp`
      dependency is an extern declaration rather than a checked body.
      This is an eligibility prerequisite, not completed `par`/`flow` semantics.
      MAC `task_41966fd9c9da4f1babfab0a8a25a66c4`.
      My three-stage bootstrap, shared Cseed/Stage 2 conformance, typechecker,
      runtime-list and schema checks pass; see
      [`closed-purity-foundation.md`](evidence/closed-purity-foundation.md).
      I bind global reads to their function source owner, so same-named
      globals in unrelated modules do not contaminate the summary.
      I also preserve complete imported parameter metadata: my old module
      registration left `type_info` uninitialized, which resource-signature
      inspection exposed during bootstrap.

- [ ] **Selfhost imported scalar global identity.** I retain ordinary scalar
      representations when separate modules define the same global spelling.
      My current native shadows emit `NlGuarded_int` as an integer return even
      without purity annotations. I test both import orders and actual values.
      MAC `task_2905cd2c7dc44e759bf64818ed0a3d77`.

- [ ] **Map-valued global factory initialization.** I preserve function
      declarations and initialized map handles when a global calls a typed
      map factory. Cseed currently reports a late prototype and Stage 2
      crashes when that global is read. MAC `task_83a9577deffb4f998924e9cd4a5cddc1`.

- [ ] **Verified foreign intrinsic purity identities.** I establish exact
      intrinsic ABI/binding contracts before restoring closed eligibility
      to wrappers such as `complex_exp`. I reject user extern collisions
      and wrong bindings in both frontends and IR facts. This remains full
      passive scope. MAC `task_20f6cb36fbf24bba987b4ea503529438`.

- [ ] **Selfhost bool-array record-field mutation.** I must select the bool
      setter for `array_set record.flags`, preserving mandatory native shadows.
      My purity bootstrap exposed an incorrect int setter; typed local aliases
      isolate the foundation while this repair remains open.
      MAC `task_d32adbdff13241dc8ad9b0a889071352`.

- [x] **Self-hosted string prefix runtime.** I implement my `str_starts_with`
      native runtime contract so importing NanoISA lowering does not leave an
      undefined `nl_str_starts_with` while compiling Stage 2 shadows.
      MAC `task_894769717bba4e0f8ea11432098406fc`. I rebuilt Stage 2,
      passed native/no-C-seed smoke checks and seven prefix-boundary assertions.

- [x] **Canonical executable NanoISA reachability.** I route explicit
      `nanoc_v06 --emit-nvm` through the program-closure API after full binding,
      type checks and shadow execution. Unreachable unsupported helpers may be
      excluded from executable bytecode; their shadows must still run and can
      reject publication without replacing prior output.
      MAC `task_771b3fdc89aa42f7bf100ce7bfd0d40d`.
      My fresh three-stage bootstrap and all five canonical driver methods pass.
      - [x] I report the lowerer's exact refusal at this checked boundary and
            preserve prior output when a reachable declaration is unsupported
            (`task_6c940d99e2674a9fabe6b27ce16517eb`).

- [x] **C-seed native nested-array literals.** I lower non-empty array-valued
      literals to an `ELEM_ARRAY` dynamic array rather than a C compound-array
      pointer. I preserve left-to-right child evaluation, aliases and the
      expected nested element context for empty replacement values, with strict
      native compilation and VM/native parity for direct, local and record-field
      forms. This item changes only my C seed; self-hosted field/type propagation
      remains separate work. MAC `task_c2ebfd28c24345daaa8c31dac75b45ac`.
      My two focused methods pass, including strict native compilation and
      byte-for-byte VM/native output parity. My transpiler, typechecker,
      empty-record-array-field and native-call-ordering gates also pass. My
      quick suite passes bootstrap, all 17 core programs and its following
      regression groups before reaching the separately tracked macOS typed-null
      failure above.

- [x] **Canonical checked frontend NanoISA output route.** I accept explicit
      `--emit-nvm` after import merging, binding, typechecking and dependency/root
      shadow checks, publishing verified module bytes without reparsing raw text.
      I retain native C as my default and C-hosted shadows as a prerequisite;
      this route does not establish my NanoISA-only fixedpoint.
      MAC `task_d176e9deba444b1e9445e1b4f41aef01`.
      Evidence: [canonical module output](evidence/canonical-nvm-output.md).

- [x] **Self-hosted NanoISA driver module output.** I add explicit `--emit-nvm`
      through my assembler facade, verify deterministic v2 bytes and publish with
      atomic output replacement. I preserve prior files on failure and test VM
      and native execution while retaining the C-seed-hosted subset boundary.
      MAC `task_1b6ed994e19447f6b58305aa52768548`.
      Evidence: [driver module output](evidence/nanoisa-driver-module-output.md).

- [x] **Hosted VM example shadow budget.** I allow a finite explicit
      budget for instrumented VM-example compiler children while preserving
      normal and deadline-regression behavior elsewhere in the full suite.
      I verify the pi example and keep every shadow assertion.
      MAC `task_840c38c505ae425da92124a08cccc1ce`. A fresh gcov NanoVirt
      reproduces the ten-second pi timeout; the scoped 60-second budget passes
      in 25.27 seconds. Hosted boids compilation also exceeds ten seconds,
      so full CI uses the same scoped override. Four environment-isolation
      cases retain other defaults.

- [x] **Full CI VM example dependencies.** I install the foreign libraries
      required by dependency shadows in all 244 eligible VM examples for full
      Linux/Darwin and coverage jobs. I retain every eligible source and all
      four verified exclusions.
      MAC `task_35a68f2cee2f4f1994cec322d72dd2b9`.

- [x] **Distinct CI platform checks.** I include the OS in full-suite check
      names so Linux and Darwin ARM jobs cannot hide each other in PR summaries.
      I inspect raw job and required test-step outcomes for release acceptance.
      MAC `task_6edaa699fd1145b8b89ab0b4ed223c47`.

- [x] **Interpreter epoch-time acceptance.** I remove the mixed-clock
      second-boundary assumption from the millisecond regression, retain
      integer and epoch conversion checks, and verify interpreter acceptance.
      MAC `task_d6d0fcccd1634cd9a1fe652f1dc3743c`. Five deterministic clock
      samples preserve the conversion contract; all 118 interpreter tests pass.

- [ ] **Publication stop fence after this release.** My MAC publisher can
      finish an admitted merge after its task is stopped. PR #357 demonstrated
      that race. I track a final pre-forge state check and deterministic race
      regression in MAC `task_de6e652dae12be62f654c1578f68c018`, dependent on
      release finalization. Temporary GitHub draft status blocks other admitted
      publications while I validate this release; it does not repair the hub.


- [x] **Native float integration after main PR #357.** I preserve float
      comparison capability with distinct value and shape tags, preserve VM tag
      ordering and typed return boundaries, and verify local/call
      transport alongside existing arrays and booleans.
      MAC `task_da116ced2b104050b956a3526ae473e8`. My focused gate passes
      1,761 translator and 1,076 shape checks, including nonfinite constants.
      Float arithmetic and result/aggregate parity remain outside this repair.

- [ ] **General native call-order audit after this release.** I compare
      other C-seed multi-argument calls against interpreter and VM ordering
      before claiming general parity. The documented self-hosted computed-call
      contract and repaired array-search contract retain their own tests.
      MAC `task_8bee9e1410ad44c2ab714b8dab98fdfa` depends on release finalization.

- [x] **Coverage-built array search and reverse.** I reproduce the hosted
      runtime failure with instrumented objects, retain the runtime diagnostic,
      and preserve ordering, typed-array and aliasing assertions.
      MAC `task_1460c2ef56414a7a98e9ce2bcd6b2890`.

- [x] **Repaired OPL VM coverage.** I include the now-compilable OPL codegen
      and compiler fixtures in example coverage, preserving the check that
      every remaining exclusion really requires exclusion. I also provide
      the compiler alias invoked by shadows from the standalone test target.
      MAC `task_0afb69d3691b416cbd1659b2600d9cdc`.

- [x] **Sanitized FFI generation identity.** I diagnose duplicate ABI-metadata
      registration across independent fixture libraries, preserve per-image
      binding and memory instrumentation, and test both loaded generations.
      MAC `task_3f7613b3c2d7453aa2d365bb5eb8d8b3`.

- [x] **Instrumented shared FFI fixture compilation.** I require PIC in
      shared-library recipes even when a caller replaces CFLAGS, keeping
      sanitizer instrumentation and real shared-object linkage.
      MAC `task_a2498192c9bf4490aa5f5f2a995214b0`.

- [x] **Public compiler release version.** I synchronize my existing compiler
      version header with package metadata, update both during future releases,
      and test metadata parity and actual CLI output.
      MAC `task_8b774619e731439aa9aef4fef2eb2656`.

- [x] **Strict CI OpenGL dependencies.** I install GLFW and GLEW alongside
      freeglut when my strict job selects the complete OpenGL example set.
      I preserve strict selection and the ban on implicit package installation.
      MAC `task_d7e811304f2a4669bfcce7a4a89198eb`.
      - [x] I also declare GLFW/GLEW for the full Darwin job, whose native
        GLUT-framework regression compiles the same OpenGL fixture.

- [x] **Darwin recursive test discovery.** I enumerate nested and root test
      fixtures on the default Bash without globstar, and reject empty negative
      coverage rather than reporting a vacuous pass. I preserve argument
      boundaries when discovered fixtures contain whitespace.
      MAC `task_0c4db2d50e2d4428828a487053535661`.

- [x] **Darwin launcher StringBuilder linkage.** I retain the runtime symbols
      referenced by imported StringBuilder objects in strict native builds,
      preserving transitive link dependencies and cache identity. I replace
      the silently truncating 2,048-byte closure buffer with complete checked
      accumulation and test a link closure that exceeds the old limit.
      Bounded compile-flag buffers must reject overflow explicitly too.
      MAC `task_7c505ffcb1f3464aa8cda8fb461dc4ee`.

- [x] **Implement a bounded hosted full-suite budget.** I preserve every release test
      while allowing the full hosted suite a bounded 3,600 seconds and its CI
      job 75 minutes for the suite, JUnit corpus and user guide. At `86aa2006`,
      Ubuntu ARM job `104950451685` exhausted the old 1,800-second make budget
      during progressing Forth Jackson tests with zero reported errors. I
      pass focused workflow checks. Final release acceptance still requires
      a fresh exact-head hosted pass, recorded in my publication evidence.
      MAC `task_0675dace285d3c8d0a593800f50ffa16`.

- [x] **Instrumented bootstrap shadow budget.** I diagnose the ten-second
      sanitizer-build deadline without dropping shadows or instrumentation,
      and test an explicit bounded budget when the workload requires it.
      My bounded parser and one/sixty-second instrumented probes pass; the
      hosted sanitizer build and bootstrap also pass at `094cfa80`;
      the full sanitizer test suite remains a release acceptance gate.
      MAC `task_6a2f75e259e84f09bbec61554b9ff9a7`.
      - [x] I carry the same bounded instrumented budget into test-unit
        prerequisite compilation, including `nanoisa_emit`. Timeout unit tests
        isolate ambient configuration and still verify the normal default.

- [x] **MAC shadow execution isolation.** I keep command execution shadows bounded without relying on a live hub or CLI startup, preserving single execution and failure status checks.
      MAC `task_5e530abdfdd3459c978149ae4b25f80f`.

- [x] **Imported fixture discovery.** I keep imported empty-array-return coverage while distinguishing module fixtures from standalone programs.
      MAC `task_853ace7b4ef841a6adf345f069bd54cd`.

- [x] **Main PRs #346 and #350 reconciliation.** I preserve my broader
      tagged global semantics, ownership and checked call formatting while
      integrating new main ancestry and reviewing its globals regressions.
      MAC `task_a8786d9cf2c1437fa46c7cfb8b99c30e`.

- [x] **Stable full-suite compiler selection.** I pin the compiler requested
      by my test entry point even when bootstrap changes the installed symlink.
      I also propagate that selection into the negative-contract runner, which
      otherwise follows the mutated symlink, including direct CI invocations.
      My C-reference suite must execute
      that compiler; the explicit self-hosted
      suite must retain its own compiler. I test the link-mutation case.
      MAC `task_9bdaa5f642b04645948f37acc282ea41`.

- [x] **List failure fixture jump state.** I preserve the fault-injection
      loop state across longjmp and keep strict GCC warnings and all ownership
      and aliasing assertions enabled.
      MAC `task_9c9cc0a069f0431f93fb12778fd78d01`.

- [x] **Escaped module compiler staging.** I preserve unusual source names
      while giving generated C inputs compiler-safe private paths. I avoid the
      GCC __FILE__ crash caused by staging directories with control bytes.
      I retain invocation, snapshot and publication checks.
      MAC `task_3488384a0b1140e389e26ef10ca1694d`.

- [x] **NanoAmp audio-list shadow.** I prepare the audio fixture correctly
      from root and examples working directories, preserving its file-list
      assertions and the strict example selection.
      MAC `task_f8d4b1586a43497cb94516601909c727`.

- [x] **OPL native dependency shadow.** I diagnose and repair the failing
      opl_compile_ast assertion in strict opl_cli compilation, preserving
      dependency shadow execution and the complete example selection.
      MAC `task_62d2355ac0f84625a51574786e3e9aa5`.

- [x] **Packaged native effect state.** I link the shared effect runtime into
      packaged wrappers so foreign modules can resolve its TLS symbols.
      I verify the real module loader path and preserve instrumentation flags.
      MAC `task_e7e676830c454787acfd6a600e3306a2`.

- [x] **Launcher shadow fixture paths.** I prepare and verify my prebuilt
      artifact fixture from both root and examples working directories.
      I retain the dependency shadows in strict example builds.
      MAC `task_d43b4d7f2ede47788cbeb2b94db32db3`.

- [x] **Effect integration bootstrap regression.** I preserve normal string
      type inference and concatenation while lowering native handlers. My clean
      bootstrap had emitted invalid pointer addition in error_messages; I now
      confine capture identity to emitted symbols and pass the clean bootstrap.
      MAC `task_7ecadba95e9445098841f83707a1d7c6`.

- [x] **UI bounds fixture math linkage.** I link the standalone C regression
      against its math dependency on Linux and retain its bounds assertions.
      I query the actual SDL2 dependencies for Darwin runtime lookup, instead
      of unrelated SDL3. The focused regression passes on Linux and Darwin.
      MAC `task_ed795be2aaa843c291abece7cffc2111`.

- [x] **Interpreter union string lifetime.** I give returned union string
      fields their own lifetime instead of borrowing a callee local. Darwin
      exposes the dangling Result payload in my user-guide checker. I retain
      all shadows and test returned payloads directly.
      MAC `task_3f227591ba94400a831005e1eec376f5`.

- [x] **Imported effect declarations.** I register imported effect signatures
      before checking module functions, and execute a handler across the module
      boundary with the shared native runtime. I preserve imported operation
      signatures when checking the module and the importing program.
      MAC `task_36491565f7db6038fb0b1591f6164c36`.

- [x] **Main PRs #338 and #340 reconciliation.** I retain dynamic native
      storage and bounded call formatting, import string-array write
      regressions, and rerun AOT acceptance. My existing wide-call fixture
      already covers the incoming ordinary and tail calls.
      MAC `task_a8786d9cf2c1437fa46c7cfb8b99c30e`.

- [x] **Packaged runtime instrumentation.** I retain build instrumentation
      flags when linking standalone and daemon wrappers, and test against
      actual coverage objects. MAC `task_2da82432bba24a13b05e693fc2c55e83`.

- [x] **Sentence dictionary build paths.** I find my bundled dictionary from
      both the repository and examples directories, preserving the shadow
      deadline and explicit dictionary override. MAC `task_a212d46b307b443db949733ed336ba60`.

- [x] **Coprocessor diagnostic portability.** I copy error text with bounded,
      terminated formatting under strict GCC sanitizer builds.
      MAC `task_6dfbada4bccb4bc4a330ecbcf38dc9e6`.

- [x] **Verifier corpus completion.** I lower supported async and effect
      forms consistently, and make my valid-resource fixture explicitly
      consume its destructor parameter. I retain all 175 selected programs.
      MAC `task_2f10710beeee47138abb2a2489dd5ef8`.

- [x] **Retained C++ module inputs.** I preserve the selected driver language
      when replaying preprocessed module sources. I verify cache reuse and
      invalidation and compile my Bullet examples. MAC `task_15fa0efcbcc14c11b6a102e3fb3b8e4f`.

- [x] **Module-object callback declarations.** I use the full callback
      signature when compiling a module object, including integer returns
      and parameters. Darwin Clang rejects the old void callback placeholder.
      MAC `task_2883edb70fb348778b7193a52bd83c8a`.

- [x] **Dispatch host availability.** I honor my existing availability query
      in dependency shadows and the API lesson. I run the boids force passes
      sequentially on hosts without dispatch. I retain every example in
      compilation coverage, verify unavailable stubs on Linux, and execute
      concurrency acceptance on Darwin. MAC `task_a960cfd36ffa408288941efab82df6fa`.

- [x] **Module-local callback names.** I preserve a function declaration
      namespace when emitting its value in generated C. I test imported
      callback shadows with named functions. MAC `task_6b378f16155b4db78c399299e56bfe44`.

- [x] **Packaged VM ownership linkage.** I include my independent ownership
      pass in packaged interpreter links and verify wrapper generation.
      MAC `task_0014f18c18694852a7b4bccce95264f6`.

- [x] **Imported native callback shadows.** I execute foreign callback tests
      through a supported ABI with owner-thread safety. My Darwin strict
      examples previously failed because the tree interpreter could not marshal
      dispatch function arguments. My shared VM bridge now passes the focused
      Darwin callback checks while retaining dependency shadows.
      MAC `task_ae64b91721834616ae8979c76b6b09df`.

- [x] **Lifted callback lexical captures.** I retain source-bounded symbols
      for later checking of lifted lambdas. I remove the conflicting eager
      visibility flags while preserving exited-block and shadowing checks.
      MAC `task_b0cd5c144a2d44a6b158e3e57288300b`.

- [x] **SDL2 mixer fixture portability.** I name the public post-mix callback
      signature locally instead of requiring a newer header alias. I retain
      the callback lifetime checks. MAC `task_461188d55ed34eae903d9c0143db7235`.

- [x] **Module metadata gate prerequisites.** I build the cache probe and
      bytecode tools before the foreign-path regression, including from a
      clean tree. MAC `task_82ee0874f7894e23add656e651843c4d`.

- [x] **GCC VM allocation fixtures.** I match the stack-count assertion to
      its unsigned runtime type and pass allocation-failure acceptance. The
      full release gate is tracked separately under finalization. MAC `task_ecf93f572e624145baff78bb3a8a2345`.

- [x] **Parenthesized parser release acceptance.** I reproduce and repair
      the explicit parser node-kind mismatch in the quick gate, preserving
      grouped operators and tuple/call distinctions. My probe shows the parser
      returns the correct kind (38), but an enum literal passed through C
      varargs is read as an eight-byte integer after four-byte promotion on
      ARM64. I cast integer-array elements to the helper's `int64_t` ABI. MAC `task_de6fc42342d54f9085cc85c342d0edfa`.

- [x] **Imported callback C typedefs.** I collect implicitly public module
      signatures before emitting declarations and preserve opaque pointer
      parameters. I verify callback imports and the SDL mixer example.
      MAC `task_2883edb70fb348778b7193a52bd83c8a`.

- [x] **Main PR #335 reconciliation.** I retain my existing 1,024-local
      runtime, import the high-slot and malformed-arity tests, and preserve
      the bounded call formatter. I rerun the native translator gate.
      MAC `task_a8786d9cf2c1437fa46c7cfb8b99c30e`.
      I pass 1,730 translator and 1,073 shape checks on Linux ARM64.

- [x] **Native test signal status.** I preserve the tested process wait status
      through shell supervision, distinguishing a signal from an ordinary
      nonzero exit. MAC `task_e77f2d8201d344d3a105ac32181e9cb7`.

- [x] **GCC native helper emission.** I brace emitted allocation and tagged
      comparison conditions so strict GCC accepts my generated runtime. I
      retain all translator assertions. MAC `task_9e5e6347b6f342aba0ff470fafe6904f`.

- [x] **Linux native regression buffers.** I size composed assembly fixtures
      for their input bounds and retain fortified GCC warnings. I rerun the
      translator suite. MAC `task_b6d77b789ec4404f8c3d64d81647c286`.

- [x] **Linux process-capture build.** I consume the child diagnostic write
      result explicitly so fortified GCC builds retain `-Werror`. I verify
      capture behavior after rebuilding. MAC `task_c4e1e543cb674af69bdc343eb9d11dcc`.

- [ ] **5.0 finalization on sparky.** I finish the user-authorized release
      from integration checkpoint `5cf21fc1`, preserving its audited contracts.
      MAC `task_a8786d9cf2c1437fa46c7cfb8b99c30e`.
      - [x] I reconcile main through `7b982db3`, including affine rejection,
        lexical bindings, native allocation lifetimes and path normalization.
      - [x] I deduplicate merged filter predicate declarations while retaining
        both branches' literal, variable and empty-result assertions.
      - [x] I reproduce the failures found through `bd6ee293`, repair them in
        dependency order, and reconcile stale checkboxes against executable
        evidence. The complete final gate below remains separate.
      - [ ] I run a clean build and full tests, review all open release work,
        and update documentation, presentation and release evidence.
      - [ ] I merge the validated candidate, publish the 5.0 tag and release,
        and verify the published artifacts against the tested commit. These
        are pre-tag checklist states; my GitHub release validation record and
        MAC finalization task record the completed publication afterward.

- [x] **Self-hosted literal filter dispatch.** I infer direct literal array
      element types before choosing a native filter helper, including the
      boolean callback ABI. My fresh Stage 2 selects the integer helper for
      float and boolean literals in `tests/nl_functions_filter.nano`, stopping
      `make test-quick` at 16 passing cases and one compilation failure. I test
      all scalar literal/variable paths and empty-result representation, then
      rerun the gate. MAC `task_ac94d5cc420e481a896d5f1a2d37f595`.
      My added empty-literal checks also expose C-seed integer fallback; I
      derive that literal's scalar representation from the predicate signature
      and test both compilers before completing this repair. The VM also keeps
      integer storage for an empty literal and truncates an appended float; I
      apply the same predicate-derived scalar type at bytecode construction.
      My expanded regression passes C-seed, Stage 1, Stage 2 and NanoVM
      shadows/execution. Native compiler acceptance passes 21 methods and
      C-backend tests pass seven cases. The quick gate passes all 17 language
      cases and all 242 eligible VM examples, then reaches the known affine
      rejection gap below. Evidence: `docs/evidence/scalar-filter-dispatch.md`.
- [x] **Self-hosted ownership observation.** I keep scalar field reads from
      consuming their enclosing resource and still diagnose observations of
      moved owners. I replace the old fake-close positive fixture with an
      unresolved-callee rejection and add contract-valid source-only probes.
      Fresh Stage 1 and Stage 2 pass all ten cases each; compiler acceptance
      passes 21 methods and the language suite passes 17 cases. The C seed
      still fails seven rejections. Evidence:
      `docs/evidence/affine-observation-boundary.md`. Remaining contract work:
      MAC `task_c60a8d2e14b7494f8875e75b16e9b087`.
- [ ] **Release gate — C-seed affine rejection parity.** I resolve the
      existing ownership task's two reproduced failures before release:
      unresolved resources must be rejected and use-after-move must carry a
      static ownership diagnostic, not merely fail a shadow. I review the
      outstanding recovery branch, preserve prior artifacts on rejection and
      rerun all three frontend cases and `make test-quick`. MAC
      `task_91ae827be4154eaa8f22698aeecc8cf1`.
      My recovery review also finds that the legacy positive `close_file`
      only reads a scalar field, while the self-hosted checker treats that read
      as moving the entire owner. I add declaration/return-only conformance
      probes that need no invented terminal operation, including parameter
      obligations, field observations, and owner counts beyond 256. The
      normative contract remains unchanged; the old nine-case smoke result
      is not conformance evidence.
      The previous task is now terminal `failed`; I continue under replacement
      `task_c60a8d2e14b7494f8875e75b16e9b087`. My new source-only probes use an
      explicitly foreign consuming boundary and do not claim runtime cleanup.
      I also audit `test_resource_tracking.nano` and `test_affine_integration.nano`:
      their print-only close helpers and direct nested-field consumption do not
      establish the normative terminal-operation and whole-owner rules.
      I replace shared-symbol mutation with a separate function ownership pass:
      growable lexical bindings, parameter obligations, moves and observations,
      branch joins, return cleanup and loop-edge checks. I add positive/negative
      source-only flow cases and retain unsupported borrow/destructuring and
      ownership-metadata work as incomplete rather than claim full conformance.
      The expanded flow cases expose self-hosted branch/loop/assignment gaps
      and a C-seed ordinary shadow leaking out of its block into argument
      type checking. I keep these failures in the shared gate while repairing
      lexical typing and self-hosted flow; seven straight-line cases are not
      enough to claim conformance.
      Lexical symbol repair: `task_a47320503e11474e8a4b51dab4b347a4`.
      I retain exact closing-brace positions for parsed blocks and bound retained
      local metadata to its declaring scope. I test same-line and multiline
      shadowing, exited-block rejection and native value/type preservation;
      keeping symbols for emission must not extend their source visibility.
      The native/shadow regression also exposes interpreter block locals
      leaking into the outer scope. I restore block bindings on normal and
      control-flow exits, preserving returned strings before local cleanup.
      My array-shadow execution probe also catches native emission choosing the
      inner integer array helper for the outer float array. I use source-aware
      lookup for emitter reads and assignments rather than the latest symbol.
      The located-block repair passes a fresh bootstrap, 36 environment checks,
      eight scope methods, evaluator/typechecker/parser/transpiler suites and
      21 native compiler methods. The ownership gate now has 26 failing
      subcases, all self-hosted. Evidence:
      `docs/evidence/lexical-scope-boundaries.md`.
      I replace the self-hosted top-level ownership scan with recursive flow
      checking: independent branch states, lexical ordinary-name masking,
      parameter/local obligations, reachable exits, assignment and loop edges.
      I retain the shared positive/negative matrix and add helper shadows;
      declaration classification and unsupported lowering remain explicit work.
      My rebuilt C seed, Stage 1 and Stage 2 pass the expanded 18-method
      frontend/flow matrix. I retain the full ownership item as open; the
      bounded cases do not establish recursive self-hosted classification,
      borrows, captures or ownership facts in NanoISA. Evidence:
      `docs/evidence/affine-selfhost-flow.md`.
      I propagate self-hosted resource classification through named record
      fields and union payloads to a least fixed point, including cycles. I
      test inherited parameter obligations and moves against the C frontend;
      generic substitution and module identity remain separate requirements.
      A fresh bootstrap and 31 combined ownership/scope methods pass after
      this change, including inherited record/union parameter obligations.
      The broader quick rerun reaches all 242 eligible VM examples and rejects
      `nl_affine_resource_demo.nano`: its close helper merely prints and leaves
      its parameter live. I implement the specified whole-record destructuring
      in both frontends and lowering paths, with complete-field and move checks,
      then give the simulated demo an honest terminal operation. I do not
      waive the obligation or exclude the example. MAC
      `task_826838b808f340968c526f849276b913`.
      The nested runtime fixture also exposes unsafe-block locals being emitted
      as globals by my self-hosted emitter. I keep their lexical ownership and
      emit them only in their block, testing native shadows and execution.
      MAC `task_44c3da4d710e4a939c8879b9b5ecd15e`. I also reject ordinary partial
      resource-field moves; only parser-marked complete projections transfer
      those field obligations after whole-owner destruction.
      My VM shadow test also exposes local names surviving block exit. I retain
      bytecode slots but retire their lexical names and bound their emitted
      type metadata for ordinary and unsafe blocks. MAC
      `task_414223eec172441b88619f5e9744c5c1`. Pattern checking also requires the
      initializer's nominal record identity, not just matching field shapes.
      My expanded runtime cases expose missing array element metadata on
      inferred field bindings and orphan `else` branches in empty-record
      reflection helpers. I preserve field-derived array types and emit valid
      zero-field fallbacks, then exercise both cases on all four compilers.
      MAC `task_4f76b90bc54d47b5b4de531930869431` and
      `task_94d61370acb44862a22b716ad0aa5ef8`.
      The 12-method record-pattern matrix passes native and VM shadows and
      execution on all four compiler paths, including the three repaired
      simulated-resource fixtures. A fresh bootstrap, 74 VM code-generation
      tests, C frontend/evaluator suites and 21 compiler-to-native methods
      pass. Evidence: `docs/evidence/owned-record-patterns.md`. The broader
      ownership contract and release gate remain open.
      The subsequent `make -j1 test-quick` exits successfully: 17 language
      cases, 242 eligible VM examples, 33 affine methods and the remaining
      runtime/Forth checks pass. The IDE builds, but graphical initialization
      is skipped without `xvfb-run` and `timeout`; it remains unverified.
      My broader quick gate stops at GLUT native compilation: source-aware
      emitter lookup selects retained declaration placeholders instead of
      imported constant values, leaving `GL_LIGHT0` and related names undefined
      in generated C. I distinguish global bindings explicitly, preserve local
      shadow precedence, and test imported constants before rerunning the gate.
      MAC `task_c4064660752d4717a8284da8388a43b2`. I also zero-initialize symbols
      created directly from C headers so new scope metadata cannot contain
      uninitialized bounds.
      I preserve each emitted local's nominal type rather than inheriting a
      different function's same-named binding. Ten lexical/global regression
      methods, a fresh bootstrap and 21 compiler-to-native methods pass.
      My C-seed flow checkpoint passes the allocation-failure sanitizer fixture,
      bootstrap smoke, resource classification and 21 native compiler methods.
      The expanded ownership gate still has 27 failing subcases, including
      foreign collection signatures on both self-hosted stages. I keep this
      item open. Evidence: `docs/evidence/affine-c-seed-flow.md`.
- [ ] **Current-main release reconciliation.** I preserve my integration
      compiler/runtime work while reviewing main and the remaining branches,
      verifying affected gates and recording ancestry before release.
      MAC `task_cffdafd16e641ac417ccfddb962534b9`.
      Merge `26397722` includes main through `b37136cc` (PR #297), including
      recursive-array fixes and SDL header metadata. My next checkpoint covers
      nine further commits through `5c358000` (PR #315). Those commits add
      nested-array evaluation, relative-path anchoring, write/close failure
      reporting and native translation fixes. I reconcile nine conflicted files.
      I retain whole-module structural/type checking and representable native
      helpers. I omit only unresolved uncalled packed-layout functions and their
      uncalled callers. I test both execution roots and keep unresolved reachable
      layouts rejected. Dropping every uncalled body broke 101 existing native
      boundary subcases, so I preserve their independent tag/representation probes.
      The pre-merge checkpoint passes 22 compiler-to-native methods, 1,673
      translator checks and 1,073 shape checks, including fresh ASan/UBSan.
      Evidence:
      `docs/evidence/main-reconciliation-pr313.md`.
      I now reconcile nine main commits through `5c358000` (PR #315), retaining
      my owned/growing native string arrays, tagged values, dynamic record shapes and
      `Value`-backed interpreter nested arrays. I import the seven incoming
      native regression functions and preserve the filesystem/shadow cases.
      For write failure tests I compile dedicated production objects with
      test-only stdio substitution instead of GNU linker wrapping. I run the
      affected suites, bootstrap and quick gate before completing this merge.
      This checkpoint passes those gates, 1,712 native translator checks,
      1,073 shape checks, fresh ASan/UBSan and the final 24-method native
      compiler suite. I integrate main through PR #315; the newer PR #316,
      PR #317 and remaining branches keep this parent item open. Evidence:
      `docs/evidence/main-reconciliation-pr315.md`.
      - [x] I replace my native scalar-local initialization refusal with tagged
        storage for locals read before assignment. I preserve void through
        branches, loops and calls, reject invalid typed consumption at runtime,
        and reset locals on self-tail calls. This reconciles PR #305 rather
        than discarding its accepted behavior.
        MAC `task_07fe41e0143b4ed7be809a95032f96e0`.
      - [x] I replace the remaining integer-array arena with checked owned
        capacity growth. My string arrays already grow; the incoming 70,000
        element integer regression exposed the still-fixed integer arena.
        I preserve alias identity and check foreign backing-store growth.
        MAC `task_8ec838f199f242ea96b07e205291c414`.
      - [ ] I extend void-before-store preservation to aggregate locals with
        owned/tagged representation, shape conversion and VM/native tests for
        arrays, records and maps. Scalar tags do not complete this work.
        I first reconcile the overlapping main translator changes below.
        MAC `task_9bc9d52fdd4a4f818064f53a4a071bae`.
      - [x] I reconcile main's newer `4936f8cf` (PR #316) after this checkpoint.
        I review its unary/grouping parser changes and import its grouping
        regression without restoring the old match-arm return-as-value rule.
        My creator's 5.0 rule remains: return exits the enclosing function;
        a final expression supplies the arm value. I also review new PR #317's
        native record-frame changes against my existing frame work.
        PR #317's zero-padded counts are C octal literals: `[008]` fails to
        compile, while `[010]` allocates eight elements rather than ten. I
        reproduce the compiler error and retain my decimal/dynamic frame
        emission when reconciling that branch.
        My integration tests now pass counts 8, 9, 10, 18 and 100 under
        generated-code ASan/UBSan; the external branch still needs repair.
        PR #317 lands as `710a1cdb` during my push. I must retain my tested
        frame implementation when reconciling that newer main commit.
        MAC `task_7a307efeea0e4ea5b2476904b2cbdfea`.
        I preserve my existing unary/grouped-call parser and match checker,
        import the two incoming regression functions, and extend match tests
        across C seed, both self-hosted stages and NanoVM. I rerun affected
        parser, native-frame and match gates before committing the merge.
        I retain the production-identical parser, match checker and native
        frame implementation, import both regression functions and verify
        1,718 translator checks, 1,073 shape checks and 27 native/match methods.
        Evidence: `docs/evidence/main-reconciliation-pr317.md`.
      - [x] I reconcile PR #309's checked direct-call formatting with my
        existing bounded formatter and dynamic operand stack. I import its
        full-arity ordinary/tail-call regression and run the translator gates.
        Both regular and fresh ASan/UBSan runs pass 1,723 translator checks
        and 1,073 shape checks. I retain existing 1,024-argument coverage.
        Evidence: `docs/evidence/native-call-branch-reconciliation.md`.
        MAC `task_ce16c7f531c94738bfca747633289540`.
      - [ ] I reconcile PR #303's early map reclamation only after tracing
        roots across caller frames, globals, aggregate fields and escaped
        strings. Its current-function-only root scan cannot establish safe
        process-wide reclamation. I reproduce these lifetimes and verify
        bounded loop allocation without freeing reachable values.
        MAC `task_d3310bef8bd541ba9e1e267ee213eb9e`.
        - [x] I reproduce a caller-map use-after-free under ASan at PR #303's
          head and replace current-frame-only collection with registered live
          frames, globals and iterative aggregate tracing. I test caller
          locals/operands, strings, arrays, nested records, mutable edges,
          backward conditional branches and 20,000 self-tail restarts. My
          focused tests require at most 16 live map/string owners and zero
          after entry cleanup. Remaining PR ancestry reconciliation stays open.
          Evidence: `docs/evidence/native-map-root-lifetimes.md`.
        - [x] **5.1.0 / caller-safe native map reclamation.** I ship the
          registered-frame repair after verifying globals, nested and mutable
          aggregates, scalar-float safepoints, backward branches, self-tail
          restarts and non-self tail teardown under ASan/UBSan. MAC
          `task_a94efdfee3486a0814f93336cf5c052c`.
          - [x] I isolate the release-wide documentation acknowledgement from
            historical negative-control tests, while applying its stated
            reason to the current-tree documentation and presentation gates.
- [x] **Portable write-failure injection.** Main's `a9f105e1` adds unconditional
      GNU linker `--wrap` flags to three test targets. My Darwin linker rejects
      those flags before tests can run. I preserve injected write/close failure
      coverage with portable test instrumentation, then run the affected
      interpreter, VM builtin and filesystem suites. MAC
      `task_d53cd80b7ac548d6accd9ddbd94817f1`.
      My integration branch already has a shared writer helper and a portable
      production-body probe for seven wrappers; `tests/test_file_write.py`
      passes on Darwin. I must preserve that coverage and reconcile the new
      test targets, not replace the helper with duplicated write logic.
      I now compile dedicated production objects with test-only stdio
      substitution. The three affected suites and generated-writer checks pass
      on Darwin, as does the seven-wrapper production-body probe. I use private
      temporary fixtures and include the filesystem target in `test-units`.
      Evidence: `docs/evidence/main-reconciliation-pr315.md`.
      My merge preserves both histories and passes the focused gates. The
      filter blocker is repaired; the broader quick gate now reaches the
      existing affine rejection gap above. Initial merge evidence:
      `docs/evidence/main-reconciliation-pr297.md`.
- [x] **Sanitizer build isolation.** I make the AOT sanitizer target rebuild
      instrumented objects instead of reusing normal objects when only `CC`
      changes. I test warm-cache behavior before claiming translator coverage.
      MAC `task_f3df199b025042e0b1d83484cd104ed3`. My warm-cache run passes
      1,048 AOT and 952 shape checks with fresh instrumented objects, leaving
      normal translator artifacts unchanged. Evidence:
      `docs/evidence/aot-sanitizer-build-isolation.md`.
- [x] **Chronicle branch reconciliation.** I review main `1a5fed53` and
      worker `633acda1`, retain their identical README chronology update, and
      integrate both histories without replacing compiler work. I verify the
      merged diff and record ancestry under release task
      `task_cffdafd16e641ac417ccfddb962534b9`. Both heads are ancestors;
      source and tests are unchanged. Evidence:
      `docs/evidence/chronicle-branch-reconciliation.md`.
- [x] **AOT temporary directories.** I adapt `vm_mktemp_dir` with checked
      template allocation and exclusive creation. I test unique, independent
      paths and failure inside private roots before compiler acceptance.
      MAC `task_419c47bdc8fc42e4b52eb6af1a0e9a71`.
      My AOT suite passes 913 checks. Compiler acceptance clears imports and
      stops at function 18's `ARR_NEW` element kind. Evidence:
      `docs/evidence/aot-temporary-directories.md`.
- [ ] **Compiler AOT array shapes.** I identify the full compiler's array
      element shapes and support their construction and data flow without
      guessing a scalar representation. I verify complete compiler execution.
      MAC `task_419c47bdc8fc42e4b52eb6af1a0e9a71`.
      My current compiler-to-bytecode-to-native-to-hello acceptance passes all
      21 test methods. I retain the broader unchecked contracts below; this
      execution test does not establish canonical bootstrap identity or general
      equivalence. Evidence: `docs/evidence/aot-owned-scalar-strings.md`.
      - [x] I construct explicitly tagged struct arrays and reject incompatible
        record-field representations on append instead of overwriting facts.
        My AOT suite passes 922 checks; evidence:
        `docs/evidence/aot-tagged-record-arrays.md`.
      - [ ] I complete compiler stack and array/aggregate shape support through
        full compiler acceptance. Function 20 (`parser_init_ast_lists`) now
        reaches unsupported array-valued fields in its 75-field aggregate.
        - [x] I allocate classifier stacks and branch snapshots from checked
          function bounds. My AOT suite passes 924 checks, including deep
          branch snapshots and incompatible heights. Compiler acceptance now
          reaches the 75-field aggregate limit; emitter storage and aggregate
          limits remain separate. Evidence:
          `docs/evidence/aot-dynamic-classifier-stack.md`.
        - [x] I allocate emitter operand storage and independent branch
          snapshots dynamically, including array-literal scratch storage and
          output. Both 75-value branch paths and 200-element integer/string
          literals execute correctly; 936 AOT checks pass. Evidence:
          `docs/evidence/aot-dynamic-emitter-stack.md`.
        - [ ] I remove fixed temporary-count limits and support the compiler's
          aggregate widths and array-valued fields without guessing their
          representation. I require full compiler translation and execution.
          - [x] I derive record field width from decoded module construction,
            use checked dynamic classifier/emitter facts, and emit matching
            value-copy record storage. I test wide mixed-field records through
            calls and branches before enabling array-valued fields. My normal
            and translator-instrumented AddressSanitizer suites each pass
            1,029 checks. Evidence: `docs/evidence/aot-module-record-width.md`.
          - [x] I preserve integer-array and string-array field representations
            through record packing, calls, locals and extraction, with tests
            for empty/nonempty arrays and copied records. Record-array fields
            still require nested shape facts before full compiler acceptance.
            My AOT suite passes 1,043 checks. Evidence:
            `docs/evidence/aot-scalar-array-fields.md`.
          - [ ] I carry recursive aggregate shape constraints through AOT
            calls and joins. MAC `task_9c850e94e5a74b6f8941622e2872af23`.
            - [x] I verify a dynamic, cycle-safe shape constraint graph with
              deep nesting, recursive records, shared children and conflicts.
              Its 952 checks pass normally and under ASan/UBSan. Evidence:
              `docs/evidence/aot-shape-constraints.md`.
            - [ ] I connect those facts to classification and C emission,
              preserve record-array element shapes, and pass full compiler
              acceptance. A standalone graph test does not complete this gate.
              - [x] I preserve nested record values using immutable snapshots
                whose lifetime outlasts returning functions, release owned
                snapshots after entry returns, and test calls, copies and
                repeated construction under sanitizers.
                My sanitizer suite passes 1,057 AOT and 965 graph checks.
                Compiler acceptance now reaches conflicting inferred field
                kinds rather than unsupported nested storage. Evidence:
                `docs/evidence/aot-nested-record-values.md`.
              - [ ] I resolve the compiler's conflicting nested-field
                inference with function/instruction diagnostics and focused
                regressions before relying on those shapes for reclamation.
                - [x] I implement `CAST_INT` stack effects and emission;
                  skipping it currently misclassifies a parsed tuple index as
                  a string. I test conversion boundaries and mixed call sites.
                  The sanitizer suite passes 1,060 AOT and 965 graph checks.
                  Compiler acceptance advances to function 170's array-kind
                  conflict. Evidence: `docs/evidence/aot-cast-int.md`.
                - [x] I preserve explicitly constructed record-array kinds
                  while nested element facts remain unknown. Function 170 at
                  offset 925 previously passed inferred `narr_t` to parameter 1
                  of function 141, which requires `nrarr_t`.
                  I keep unknown element facts unknown and stop `ARR_LEN`
                  from guessing integer elements. Normal and sanitizer suites
                  pass 1,065 AOT and 965 graph checks. Evidence:
                  `docs/evidence/aot-delayed-array-facts.md`.
                - [x] I propagate record-array result field facts through
                  calls and returns instead of supplying integer defaults.
                  Normal and tail-return regressions preserve mixed fields.
                  The sanitizer suite passes 1,070 AOT and 965 graph checks.
                  Evidence: `docs/evidence/aot-record-array-return-fields.md`.
                - [ ] I infer scalar versus record array results in native
                  signatures instead of classifying every array return as a
                  record array. Compiler acceptance reaches function 323,
                  offset 5359: `split_tuple_type_names` returns strings, but
                  its indexed element is inferred as a record when passed to
                  `strip_spaces`. I test direct/tail calls, recursion, empty
                  arrays and preserved record-array behavior. MAC
                  `task_f6b029d7b7e749caa7a064c9566bd666`.
                  - [x] I propagate integer-, string- and record-array result
                    representations through function facts, typed native
                    signatures, direct/tail calls and final shape resolution.
                    I test forward definitions, recursive results, parameter
                    passthrough, empty arrays and conflicting return paths.
                    Full compiler acceptance remains a separate gate. Normal
                    and sanitizer suites pass 1,284 AOT and 994 shape checks.
                    Evidence: `docs/evidence/aot-array-result-kinds.md`.
                  - [x] I reconcile the direct empty array return with the
                    nonempty string array return in `extract_type_args` by
                    retaining its declared element type during lowering.
                    Fresh compiler bytecode passes the former conflict at
                    function 369, offset 553 without weakening native checks.
                    - [x] I retain the declared array element type when
                      lowering a direct empty return literal, including nested
                      function contexts. I test emitted tags, VM execution and
                      native translation without reinterpreting explicit
                      integer-array bytecode as string arrays. All 74 codegen
                      tests pass; source-to-bytecode-to-native empty/nonempty
                      int, string and record return fixtures pass. Evidence:
                      `docs/evidence/empty-array-return-tags.md`.
                - [x] I remove my native translator's 256-local bottleneck
                  without weakening malformed-arity checks. Fresh compiler
                  bytecode reaches function 463, which exceeds that limit;
                  NanoVirt allows 1,024 locals. I test high local indices,
                  parameter facts and checked allocation sizes, and rerun full
                  compiler acceptance. Normal and fresh ASan/UBSan suites pass
                  1,308 AOT and 994 shape checks. Full compiler translation now
                  reaches generic `LT`. Evidence:
                  `docs/evidence/aot-wide-locals.md`. MAC
                  `task_183123ca3822426499719b042132b49f`.
                  - [x] I size local facts from the actual module width and
                    accept up to 1,024 locals with checked arity and allocation
                    bounds. I test high indices, cross-function facts and
                    uninitialized-local rejection.
                  - [x] I size direct-call text for supported parameter counts
                    and check each formatted append before advancing its
                    offset, including separators and the closing parenthesis.
                    MAC `task_ce16c7f531c94738bfca747633289540`.
                - [ ] I lower generic comparison opcodes with the VM's operand
                  rules. Fresh compiler translation reaches unsupported `LT`
                  (`0x2A`) in function 532 at offset 47. I inspect its operand
                  types and test checked native comparison instead of assuming
                  every comparison has integer operands. MAC
                  `task_dc171e3db9cb497c981d7f06621baa5d`.
                  - [x] I lower `LT`, `LE`, `GT` and `GE` for native integers,
                    booleans, strings, tagged values, arrays and maps using
                    value/tag ordering. I retain boolean result tags. Normal
                    and fresh ASan/UBSan suites pass 1,444 AOT and 994 shape
                    checks. Float, enum and erased nominal-tag comparisons
                    remain outside this native subset. Evidence:
                    `docs/evidence/aot-generic-ordering.md`.
                - [x] I preserve tagged argument storage while checking exact
                  host-call parameter tags at consumption. Fresh compiler
                  translation reaches function 569, `exists`, at import 29
                  (`vm_file_exists`, string parameter, bool result), where the
                  classifier rejects an argument-kind mismatch. I test valid,
                  missing and wrong tags without weakening the host ABI. MAC
                  `task_f6b888111b8246f89843e1f945d80936`. Normal and fresh
                  ASan/UBSan suites pass 1,476 AOT and 994 shape checks.
                  Evidence: `docs/evidence/aot-tagged-host-arguments.md`.
                - [x] I retain the compiler's nested aggregate field shapes
                  through `AGG_PACK`. Full compiler acceptance now reaches
                  function 599, `collect_files_dfs`, whose packed result has a
                  hashmap field. I test recursive packing and invalid
                  layouts before rerunning compiler acceptance. MAC
                  `task_f8e9b2d46ce3470f81ed8b06acbea7ed`. Normal and fresh
                  ASan/UBSan runs pass 1,509 AOT and 994 shape checks. Evidence:
                  `docs/evidence/aot-map-aggregate-fields.md`.
                - [x] I reconcile tagged string fields across compiler call
                  shapes without accepting incompatible payloads. After map
                  field support, `check_function` (333), offset 351, conflicts
                  at field 3 of parameter 5 of `symbol_new` (301): string
                  versus tagged value. I test both function orders and actual
                  runtime values. MAC `task_ad09fe6dbbb24d5781e970dca9db299a`.
                  Evidence: `docs/evidence/aot-optional-record-arrays.md`.
                - [x] I retain optional record-field constraints across array
                  returns and tail calls. After argument conversion, function
                  281 offset 769 conflicts at returned element field 3. I test
                  present and absent return paths and reject incompatible
                  payloads. MAC `task_380228e6684346c7b211956940ca4f43`.
                  Normal and fresh ASan/UBSan runs pass 1,523 AOT and 994 shape
                  checks. Evidence: `docs/evidence/aot-optional-record-arrays.md`.
                - [ ] I resolve final aggregate field facts in function 163,
                  `parser_store_union_construct`. Full compiler acceptance
                  now reaches its `AGG_PACK` rejection. I trace the unresolved
                  field rather than inventing an integer representation. MAC
                  `task_a478c928daf246129a8b8fb82da27fd6`.
                  - [x] I defer packed-field validation until graph construction
                    is complete and still reject genuinely unresolved fields.
                    Normal and fresh ASan/UBSan runs pass 1,528 AOT and 994
                    shape checks. Evidence:
                    `docs/evidence/aot-late-packed-field-shapes.md`.
                  - [ ] I verify these fields in the complete compiler graph;
                    its traversal now stops at the array-update gate below.
                - [ ] I reconcile record-array update facts after recursive
                  inference. With deferred packed-field validation, full
                  compiler acceptance reaches `ARR_SET record field
                  representation mismatch`. I preserve incompatible-layout
                  rejection and alias semantics while tracing the exact
                  field. MAC `task_dfd2bd4170e544b9869cb936484845b9`.
                  - [x] I defer unknown flat-field comparisons at `ARR_SET`
                    while retaining recursive element unification and runtime
                    guards. Both function orders preserve alias-visible writes
                    and reject incompatible nested fields. Normal and fresh
                    ASan/UBSan runs pass 1,534 AOT and 994 shape checks. Evidence:
                    `docs/evidence/aot-late-array-update-fields.md`.
                - [x] I preserve boolean element tags in native arrays.
                  `typecheck_local_lets` (305), offset 41, appends a boolean to
                  an explicitly boolean empty array and conflicts in the shape
                  graph. I test literals, append, get/set, calls and tagged
                  values without conflating boolean and integer tags. MAC
                  `task_4fb63e3486c4404fa2f8b9e5ee8124a9`.
                  Normal and fresh ASan/UBSan runs pass 1,572 AOT and 994 shape
                  checks. Evidence: `docs/evidence/aot-boolean-arrays.md`.
                - [ ] I resolve the recursive shape conflict in `env_get_type`
                  (311), offset 273, at a tail call after boolean-array support.
                  I trace source/result fields and test compatible optional
                  records without accepting incompatible payloads. MAC
                  `task_d1cdf01edf9040b2885cb31134688288`.
                  - [x] I isolate nested present/missing string returns in
                    `tests/nanoisa/fixtures/nested_optional_returns.nasm`.
                    The VM executes it; native translation rejects it. I add
                    that fixture to the compiler acceptance gate and report
                    conflicting shape kinds and node IDs in diagnostics.
                  - [x] I implement recursive storage conversion without
                    merging a present string node into its optional wrapper.
                    I test shared payload nodes, call order, recursive graphs
                    and incompatible payloads before clearing the regression.
                    The nested acceptance fixture passes in four call/order
                    combinations. Normal and fresh ASan/UBSan suites pass
                    1,572 AOT and 1,073 shape checks. Evidence:
                    `docs/evidence/aot-directed-storage-conversions.md`.
                - [x] I resolve local record storage joins in
                  `check_match_expr` (319), offset 650. After recursive return
                  conversion, `STORE_LOCAL` still equates optional and string
                  field representations. I test branch and assignment order,
                  source preservation and incompatible payload rejection. MAC
                  `task_0557eb72ec60494983314ecb11a31a36`.
                  Normal and fresh ASan/UBSan suites pass 1,572 AOT and
                  1,073 shape checks; all twelve local-join cases pass.
                  Evidence: `docs/evidence/aot-local-record-storage.md`.
                - [x] I resolve the later `check_expr_node` (323) tail-call
                  string/optional shape conflict at offset 397 after local
                  storage conversion. I trace argument and result constraints,
                  retain exact payload checks and rerun compiler acceptance.
                  MAC `task_5c9df665e6024af3a0cd243dfa6fe8ca`.
                  Eight projected-argument cases pass; normal and fresh
                  ASan/UBSan suites pass 1,572 AOT and 1,073 shape checks.
                  Evidence: `docs/evidence/aot-projected-optional-arguments.md`.
                - [x] I resolve tagged scalar writes to native string arrays in
                  `cg_append` (359), offset 6, after projected optional
                  arguments. I check exact runtime string tags, preserve aliases,
                  reject incompatible payloads and rerun compiler acceptance.
                  MAC `task_ed5f20e9b78d4759b44e7b496b92a2ea`.
                  Normal and fresh ASan/UBSan runs pass 1,670 AOT and 1,073
                  shape checks. Evidence:
                  `docs/evidence/aot-tagged-string-array-writes.md`.
                - [ ] I resolve the final storage-conversion solver conflict
                  after tagged native string-array writes: an optional source
                  attempts to widen an exactly constrained string destination.
                  I identify the originating constraints, retain exact consumer
                  checks and rerun compiler acceptance. MAC
                  `task_146d0626844a4b958bfe0e8697226185`.
                  - [x] I retain all incoming record-local field facts across
                    branch assignments and inference passes. I test both branch
                    outcomes and function orders, and rerun my AOT gates.
                    Normal and fresh ASan/UBSan runs each pass 1,670 AOT and
                    1,073 shape checks. Evidence:
                    `docs/evidence/aot-branch-record-fields.md`.
                  - [ ] I resolve the remaining conversion from `env_get_type`
                    (311), offset 268, after fixing the local-27 join in
                    `check_statement_list` (331), offset 1020. I preserve exact
                    payload constraints and establish full compiler acceptance.
                    - [x] I represent inferred record-string field facts as
                      directed flows at record boundaries and projections.
                      Four VM/native cases pass and four incompatible payload
                      cases are rejected. Full compiler conversion solving now
                      passes; its later packed-field failure remains below.
                      Evidence: `docs/evidence/aot-inferred-record-string-fields.md`.
                - [x] I resolve `type_from_kind` (260), offset 165, where
                  `AGG_PACK` field 0 remains unresolved after inferred record
                  string facts become directed flows. I retain exact payload
                  checks and rerun full compiler acceptance. MAC
                  `task_041fdf3407774b93a24bbaaa30a7bbb9`.
                  I use supported parameter declarations only after caller
                  inference converges. Eight native ABI cases and six tagged
                  argument cases pass; eight missing-declaration cases reject.
                  Normal and fresh ASan/UBSan suites pass 1,670 AOT and 1,073
                  shape checks. Full compiler acceptance reaches the next
                  failure below. Evidence:
                  `docs/evidence/aot-declared-parameter-fallback.md`.
                - [x] I resolve the uncalled `codegen_next_temp` (362), offset
                  28, whose packed field 0 lacks a record-field shape. I inspect
                  available layout metadata and uncalled-function handling
                  without inventing field types. MAC
                  `task_8aaa3f722ce34624a4bb8a16283afa2b`.
                  I test scalar field inference from matching declared nominal
                  constructors, refuse ambiguous evidence, preserve variant and
                  width boundaries, and rerun full compiler acceptance.
                  Six native reconstruction cases and ten rejection cases pass.
                  Normal and fresh ASan/UBSan suites pass 1,670 AOT and 1,073
                  shape checks; full compiler emission reaches the next failure
                  below. Evidence: `docs/evidence/aot-nominal-scalar-fields.md`.
                - [x] I resolve `parser_get_identifier_count` (192), whose
                  record-field-10 projection reaches `ARR_LEN` without an
                  array representation. I preserve runtime element tags rather
                  than guessing storage. MAC
                  `task_9a2a87fb80d94386b096a7b01ec7eaaf`.
                  Twenty runtime-storage, alias and rejection cases pass.
                  Normal and fresh ASan/UBSan suites pass 1,670 AOT and 1,073
                  shape checks. Evidence:
                  `docs/evidence/aot-projected-array-length.md`.
                - [x] I resolve the next emitted record projection in
                  `typecheck_parser` (339), where `AGG_GET` receives a
                  non-record representation. I identify its exact operand,
                  preserve nested field checks and rerun compiler acceptance.
                  MAC `task_f682c0c61d354890a80c28b3ed32918f`.
                  Sixteen nested-read and rejection cases pass. Normal and
                  fresh ASan/UBSan suites pass 1,670 AOT and 1,073 shape checks.
                  Full compiler emission reaches the next failure below.
                  Evidence: `docs/evidence/aot-nested-record-consumers.md`.
                - [x] I resolve the emitted string-store mismatch in
                  `generate_enum_definitions_from_tokens` (394). I trace its
                  source and destination representations, retain exact type
                  distinctions and rerun compiler acceptance. MAC
                  `task_34d2d9345d584ff78fab59261eb112d6`.
                  Thirty-two native string-operation and tag-rejection cases
                  pass. Normal and fresh ASan/UBSan suites pass 1,670 AOT and
                  1,073 shape checks; full compiler emission advances below.
                  Evidence: `docs/evidence/aot-string-consumer-shapes.md`.
                - [x] I resolve projected array reads in `genenv_get_mut`
                  (410), preserving the container shape as well as the boolean
                  element representation, runtime tags and bounds. I rerun full
                  compiler acceptance. MAC
                  `task_3b10cd3d807f44d8bf04a5128b697932`.
                  I include array kinds resolved through instruction shapes in
                  native runtime-helper selection, not only local variables and
                  explicit array constructors.
                  Eight native reads and twenty-four storage/tag/bounds
                  rejection cases pass. Normal and fresh ASan/UBSan suites pass
                  1,670 AOT and 1,073 shape checks. Full compiler emission
                  advances to the failure below. Evidence:
                  `docs/evidence/aot-projected-array-reads.md`.
                - [x] I resolve the array parameter's emitted storage in
                  `gen_call` (440), preserving its runtime element tags when
                  native element storage is unresolved. I rerun full compiler
                  acceptance. MAC `task_9c768255d97544cc9e850ef547b90fc3`.
                  I delay the unknown-to-integer fallback until shapes resolve.
                  Forty-eight generic-array execution and rejection cases pass.
                  Evidence: `docs/evidence/aot-generic-array-parameters.md`.
                - [x] I keep retained generated functions warning-clean under
                  strict C11 compilation without invoking uncalled helpers or
                  weakening compiler flags. I rerun native compiler acceptance.
                  MAC `task_cd4036cffef64fd8a046f88153564b69`.
                  Two function orders pass strict C11 compilation and execution
                  without invoking the trapping helper. Normal and fresh
                  ASan/UBSan suites pass 1,670 AOT and 1,073 shape checks.
                  Full compiler acceptance advances to the runtime failure below.
                  Evidence: `docs/evidence/aot-generic-array-parameters.md`.
                - [x] I resolve the native compiler's artifact-loading abort while
                  compiling `examples/language/nl_hello.nano`. Native C emission,
                  strict C11 compilation and `--help` pass; compiling a program
                  does not. I trace the failing runtime check, add a regression
                  and rerun end-to-end acceptance. MAC
                  `task_fca5000374ef49988da33e5617e703b8`.
                  I traced the abort to `dlopen` in the `path_canonical` adapter:
                  my standard-library artifact requires the missing host symbol
                  `dyn_array_get_string`. Fresh compiler emission reproduces it.
                  I provide an explicit AOT host-runtime link target using my
                  existing array and GC implementation, test real standard-
                  library loading and owned walk-result release, and document
                  native link flags before retrying compiler execution.
                  My real-artifact regression passes loading, copied filename
                  access after foreign release and balanced GC object counts.
                  GC regressions pass; full compiler acceptance passes 16 of 17
                  methods and advances to the parser failure below. Evidence:
                  `docs/evidence/aot-host-runtime-linkage.md`.
                - [x] I preserve token record-field tags in native compiler
                  execution. With the host runtime linked, hello compilation
                  reaches `parser_is_at_end` and aborts on field zero's expected
                  storage kind 2. I trace construction and projection, retain
                  validation and rerun compiler acceptance. MAC
                  `task_45fedd409e1447089dad970396b0a075`.
                  I project unresolved scalars through checked tagged values,
                  including local copies, without guessing an integer type.
                  Forty-eight execution/rejection cases pass. Normal and fresh
                  ASan/UBSan suites pass 1,670 AOT and 1,073 shape checks.
                  Full acceptance passes 17 of 18 methods; native compiler
                  execution remains failing. Evidence:
                  `docs/evidence/aot-unconstrained-scalar-projections.md`.
                - [ ] I bound native record and temporary frame growth during
                  parsing. Fresh native compiler runs exhaust an approximately
                  8 MiB stack about seventeen frames deep while compiling hello.
                  I measure storage/lifetimes, add a bounded-stack regression and
                  rerun acceptance without merely raising the stack limit. I also
                  reconcile the full suite's SIGABRT with standalone SIGSEGV.
                  MAC `task_e81212c768d148639429d1d2be9f826d`.
                  - [x] I allocate record temporary pools per invocation and
                    snapshot returns before freeing them. I check ordinary,
                    self-tail and cross-tail calls under a 2 MiB stack cap,
                    allocation balance and allocation failure.
                    Normal and fresh ASan/UBSan suites pass 1,670 AOT and
                    1,073 shape checks; compiler acceptance passes 18 of 19
                    methods. Evidence: `docs/evidence/aot-scoped-record-temporaries.md`.
                  - [ ] I reduce residual record-local and by-value argument
                    frames. Scoped temporary pools alone leave megabyte-scale
                    frames in `parse_primary` and `generate_expression`. I add
                    deep-expression coverage before claiming general recursion
                  space bounds. MAC `task_ec6559ac267b475daec33dd711c34ec1`.
                  - [x] I explain the differing signals: this host's `make`
                    raises recipe stack limits from 8,176 to 65,520 KiB. The
                    larger-stack run reaches the fixed string-array arena abort;
                    direct runs exhausted the old record-temporary stack pools.
                - [x] I replace the fixed string-array growth arena with checked
                  owned capacity growth. Native compiler execution now reaches
                  `cg_append` and exhausts `nsarr_arena` in `nsarr_push` while
                  emitting the C runtime. I preserve aliases, test sustained
                  append and allocation failure, and rerun compiler acceptance.
                  MAC `task_7b8691dd087e48ea9afdeb89ddcf4640`.
                  I test 70,000 appends, large literals, aliases, borrowed buffers,
                  escaped copies, balanced cleanup and twelve failure cases.
                  Normal and fresh ASan/UBSan suites pass 1,670 AOT and 1,073
                  shape checks; compiler acceptance advances to 19 of 20 methods.
                  Evidence: `docs/evidence/aot-owned-string-array-growth.md`.
                - [x] I replace the fixed scalar-string arena with checked
                  owned storage. Native compiler execution now passes array
                  growth and aborts in `nstr_concat`, called by `cg_build` while
                  emitting the C runtime. I preserve escaped strings, test
                  allocation/size failures and rerun full acceptance. MAC
                  `task_b0c4ad8c9a824e64ab8fd3fa6881146e`.
                  Large strings, escaped values, formatting, cleanup and six
                  negative cases pass. Normal and fresh ASan/UBSan suites pass
                  1,670 AOT and 1,073 shape checks. All 21 compiler acceptance
                  methods now pass. Evidence: `docs/evidence/aot-owned-scalar-strings.md`.
                - [x] I resolve the next compiler field conflict without
                  weakening compatibility checks: function 264 offset 60,
                  `ARR_PUSH` field 0 has string versus integer facts.
                  I preserve explicit record-array literal tags and element
                  fields, including empty literals, with execution and
                  incompatible-element tests before compiler acceptance.
                  Unresolved local field vectors also remain unknown rather
                  than defaulting to integer. The normal suite passes 1,080
                  AOT and 965 graph checks. Evidence:
                  `docs/evidence/aot-record-array-literals.md`.
                - [ ] I diagnose and implement the compiler's hashmap opcode
                  requirements. Acceptance now passes the previously rejected
                  record-result merge and stops at `LOAD_GLOBAL` in function
                  323 (`check_expr_node`) at offset 21. General map and tagged
                  aggregate flow remains incomplete below.
                  - [x] I implement reusable emitted map storage with checked
                    growth, owned keys/values, missing-key results, replacement,
                    deletion and retained lookup values; test both integer and
                    string maps. Normal and sanitizer suites pass 1,106 AOT
                    and 965 graph checks. Evidence:
                    `docs/evidence/aot-map-storage.md`.
                  - [ ] I connect map storage to classification and emission,
                    preserving value kinds, missing-key tags and ownership
                    through locals, branches, calls and returns. Storage tests
                    alone do not implement `HM_NEW` or complete acceptance.
                    - [x] I carry map/key/value facts and emit construction,
                      mutation, presence, length and deletion through calls
                      and aliases. I retain `HM_GET` as an explicit gate until
                      missing-value tags and fetched-value ownership survive
                      compiler data flow.
                      My normal and sanitizer suites pass 1,114 AOT and 980 shape
                      checks. Compiler acceptance reaches `HM_GET` in function
                      268 at offset 32. Evidence:
                      `docs/evidence/aot-map-flow.md`.
                    - [ ] I reclaim unreachable emitted maps and fetched
                      string copies before entry returns, preserving aliases
                      and returned values under
                      bounded-live-memory stress. MAC
                      `task_2f837947d5f24130b401ae433dd8d8c9`.
                    - [ ] I preserve lookup results as tagged values through
                      stack transfers, locals, branches, calls and returns.
                      I test missing versus zero/empty values and retained
                      strings after replacement/deletion. Scalar consumers
                      check tags at consumption; lookup itself does not trap
                      or invent a default. I extend mixed-value joins and
                      return inference before claiming full integration.
                      - [x] I check tagged scalar results at typed function
                        returns, matching VM return-tag validation. I accept
                        present values and trap missing or wrong-tag values at
                        that boundary, without equating optional storage with
                        the declared scalar shape. Normal and sanitizer suites
                        pass 1,226 AOT and 994 shape checks. Evidence:
                        `docs/evidence/aot-tagged-scalar-returns.md`.
                      - [x] I emit tagged lookups through locals, compatible
                        calls and same-representation branch joins, retaining
                        fetched strings after map mutation. I test missing
                        values, casts, tag inspection, scalar consumption and
                        truthiness. Normal and sanitizer suites pass 1,128
                        AOT checks and 990 shape checks. Evidence:
                        `docs/evidence/aot-tagged-map-lookups.md`.
                      - [ ] I preserve boolean versus integer runtime tags
                        across native scalar flow, generic equality and map
                        insertion. I do not infer a runtime tag from their
                        shared C integer representation. MAC
                        `task_811f280202174ac88a501ca3281d5e58`.
                        - [x] I give booleans distinct scalar and shape facts
                          through producers, locals, calls, returns and record
                          fields. I test tag inspection, mixed equality,
                          casts and refusal at integer-only boundaries.
                          Normal and sanitizer suites pass 1,191 AOT and 994
                          shape checks. Evidence:
                          `docs/evidence/aot-boolean-tags.md`.
                      - [ ] I preserve void-valued local reads before stores,
                        including conditional and loop paths. Until native
                        storage carries that tag, I reject potentially
                        uninitialized reads rather than inventing zero values.
                        MAC `task_288b833c300d40d18f6204c0cccba24f`.
                        - [x] I reject potentially uninitialized native local
                          reads using a control-flow definite-assignment
                          worklist, with branch, backedge and dead-code tests.
                          This is a guard, not void-valued local support.
                          Evidence: `docs/evidence/aot-boolean-tags.md`.
                      - [x] I reconcile ordinary and optional string arguments
                        at function boundaries, including forward calls and
                        tail calls. I preserve the present-value shape and
                        reject incompatible payloads; caller order must not
                        decide whether the program compiles. Normal and
                        sanitizer suites pass 1,137 AOT and 990 shape checks.
                        Evidence: `docs/evidence/aot-mixed-lookup-arguments.md`.
                      - [ ] I preserve tagged fields through record packing,
                        extraction, returned record facts and compatible
                        ordinary/optional field joins. I retain missing tags
                        and owned lookup strings instead of unboxing on pack.
                        - [ ] I reconcile ordinary and tagged record argument
                          fields at call boundaries. With scalar global loads
                          enabled, compiler acceptance reaches function 323
                          offset 1761 and conflicts on field 1 of parameter 0
                          of function 274. I preserve source representations
                          while widening the callee's field storage.
                          - [x] I support ordinary/optional string fields at
                            direct and tail record calls, preserving source
                            records and checking payload compatibility. I test
                            both caller orders and function-definition orders;
                            nested and other scalar field joins remain open.
                            Normal and sanitizer suites pass 1,259 AOT and
                            994 shape checks. Evidence:
                            `docs/evidence/aot-record-arguments.md`.
                        - [x] I carry consistently tagged fields through
                          records, nested snapshots and returned record
                          arrays. I defer unresolved projection validation
                          until the complete shape graph is available, then
                          require an actual tagged representation at emission.
                          Normal and sanitizer suites pass 1,142 AOT and 990
                          shape checks. Evidence:
                          `docs/evidence/aot-tagged-record-fields.md`.
                        - [x] I join ordinary and optional string fields in
                          record results using explicit source-to-result
                          compatibility, including tail returns. I test both
                          return paths, missing values and unchanged ordinary
                          source strings; nested field joins remain required.
                          Normal and sanitizer suites pass 1,155 AOT and 990
                          shape checks. Evidence:
                          `docs/evidence/aot-optional-record-results.md`.
                - [ ] I implement typed global load/store data flow, checked
                  slot sizing, VM-compatible initialization and shared
                  cross-function mutation/ownership. I test uninitialized
                  values and full compiler acceptance. MAC
                  `task_bcc4271b0de244c2810c09e004f6cd2e`.
                  - [x] I invoke the module's zero-argument `__init__` before
                    entry, matching VM startup. I test call order, discarded
                    initializer results, initializer failure, invalid arity
                    and an initializer that is also the entry point. Normal
                    and sanitizer suites pass 1,176 AOT and 990 shape checks.
                    Evidence: `docs/evidence/aot-module-initializer.md`.
                  - [ ] I preserve void-before-store, boolean versus integer
                    tags, and empty-array element inference across global
                    initialization and later stores. My compiler's fourteen
                    globals include booleans, strings and arrays; scalar-only
                    storage does not satisfy its requirements.
                    I now pass the earlier empty string array store to global
                    6 in `register_extern_names`; full compiler translation
                    still requires compatible empty-array return inference.
                    - [x] I carry integer and string array handles in tagged
                      globals, preserving alias mutation, void-before-store
                      and void-valued out-of-range reads. I test array
                      length/read/push/set, copied global handles and tag
                      checks. Record arrays and empty-array inference remain
                      required before I complete globals. Normal and fresh
                      ASan/UBSan suites pass 1,294 AOT and 994 shape checks.
                      Evidence: `docs/evidence/aot-array-globals.md`.
                    - [ ] I preserve void-valued out-of-range reads from
                      ordinary native arrays too. Existing untagged helpers
                      abort at lookup; NanoVM yields void and lets consumers
                      decide. I test ignored results, tag inspection, casts,
                      typed consumers and index narrowing. MAC
                      `task_ed0f455484d04f13818362fd857d2889`.
                    - [x] I first carry tagged scalar global loads and stores
                      through initialization, cross-function mutation and
                      saved values. I check slot bounds and retain void before
                      assignment. Aggregate storage and array inference remain
                      required for the full compiler. Normal and sanitizer
                      suites pass 1,241 AOT and 994 shape checks; all 95
                      verifier tests pass. Evidence:
                      `docs/evidence/aot-scalar-globals.md`.
                - [ ] I audit classifier opcode coverage against emission
                  and verifier stack effects, explicitly handling or rejecting
                  each opcode instead of silently skipping unknown effects.
                  MAC `task_f90db79b0f464637a18486c44262c4d3`.
                  - [x] I reject unimplemented classifier instructions at
                    their own offsets, check classifier/emitter case parity,
                    and test representative unsupported instruction families.
                    Normal and sanitizer suites pass 1,102 AOT and 965 graph
                    checks, plus the opcode case-parity test. Compiler
                    acceptance now names `HM_NEW` at function 267 offset 0.
                    Evidence: `docs/evidence/aot-opcode-coverage.md`.
              - [ ] I reclaim unreachable nested-record snapshots during
                long-running execution, with bounded-live-state stress tests
                and alias-safe destruction. Entry-return cleanup alone does
                not bound retained memory. MAC
                `task_d152cc3913f248fb8d1483210e60f00b`.
              - [x] I store record-array fields by reference and preserve
                their element shapes through packing, calls, extraction and
                aliases; I test empty arrays and mixed-field elements.
                Normal and sanitizer suites pass 1,055 AOT checks and 965
                graph checks. Compiler acceptance advances from function 20
                to function 100's nested record packing. Evidence:
                `docs/evidence/aot-record-array-fields.md`.
              - [x] I read resolved field representations from the graph
                during emission without creating new constraints; I test
                missing edges, aliases and existing extraction behavior.
                Normal and sanitizer runs pass 1,048 AOT and 965 graph
                checks. Nested storage remains unfinished. Evidence:
                `docs/evidence/aot-resolved-field-emission.md`.
              - [x] I attach persistent shape variables to production
                classifier values, locals, parameters, results and joins,
                checking compatibility alongside existing representation facts.
                My AOT suite passes 1,048 checks normally and with translator/
                graph ASan/UBSan instrumentation; 952 graph checks pass.
                Evidence: `docs/evidence/aot-production-shape-constraints.md`.
          - [x] I separate record and record-array temporary field facts;
            their independent indices must not overwrite each other. I test
            live string-record arrays across scalar record construction and
            the reverse collision, directly and through both branch paths.
            My AOT suite passes 1,010 checks. MAC
            `task_3673443775f2477c94688b1021d6102a`; evidence:
            `docs/evidence/aot-record-fact-namespaces.md`.
          - [x] I size generated temporary arrays to actual high-water counts,
            allocate emitter record facts and snapshots dynamically, and test
            more than 256 temporaries. I check output-growth arithmetic before
            inserting the resulting declarations. My AOT suite passes 992
            checks; evidence: `docs/evidence/aot-dynamic-temporaries.md`.
          - [x] I select array constructors from emitted representations,
            including direct stack construction without locals and legacy
            inferred array kinds; strict generated-C compilation passes.
- [x] **AOT byte-character conversion.** I preserve the existing C-byte
      `vm_string_from_char` contract, including zero-byte empty text and
      independent storage. I test integer boundaries before compiler acceptance.
      MAC `task_419c47bdc8fc42e4b52eb6af1a0e9a71`.
      - [x] I emit `INT64_MIN` without an out-of-range positive C literal;
        strict C compilation and the signed-extrema arithmetic case pass.
      My AOT suite passes 904 checks. Compiler acceptance remains failing at
      `vm_mktemp_dir` (import 34). Evidence: `docs/evidence/aot-byte-character.md`.
- [x] **AOT shell execution.** I adapt the exact `nl_exec_shell` builtin
      contract, preserving raw system status. I test success, nonzero exit and
      signature rejection before rerunning compiler acceptance.
      I also preserve bounded capture/draining with independent result storage.
      MAC `task_419c47bdc8fc42e4b52eb6af1a0e9a71`.
      My AOT suite passes 842 checks. Compiler acceptance still fails at
      `vm_string_from_char` (import 32). Evidence: `docs/evidence/aot-shell-capture.md`.
- [x] **AOT lexical normalization.** I normalize builtin paths without
      fixed component/output limits, preserving relative parents and roots.
      I test long paths and rerun compiler acceptance.
      MAC `task_419c47bdc8fc42e4b52eb6af1a0e9a71`.
      My AOT suite passes 828 checks. Compiler acceptance still fails at
      `nl_exec_shell` (import 25). Evidence: `docs/evidence/aot-builtin-normalize.md`.
- [x] **Native normalization bounds.** I replace module path truncation and
      generated-native `parts[512]` overflow on leading parents with checked
      dynamic storage. I test both paths beyond their old limits.
      MAC `task_82bd388637824cc889b12204d226e75b`.
      - [x] I use checked dynamic storage for public `path_normalize` and
        generated `nl_os_path_normalize`, with 700-component/parent and
        5,000-byte tests. Evidence: `docs/evidence/native-normalization.md`.
      - [x] I remove the remaining bounded normalization/token/output buffers
        inside `path_relpath`; public normalization does not fix this caller.
        Long shared-prefix and 4,504-byte result tests pass. Evidence:
        `docs/evidence/dynamic-relative-paths.md`.
- [x] **Relative-path anchors.** I define and verify dot, mixed-root and
      unresolved-parent behavior rather than treating all normalized components
      as interchangeable. MAC `task_64696231d8984732a5a1e1c319ca043b`.
      I anchor relative inputs to one dynamically read working directory;
      reconstruction, long-cwd and unavailable-cwd cases pass. Evidence:
      `docs/evidence/relative-path-anchors.md`.
- [x] **AOT identity checks.** I preserve file/destination identity semantics
      for builtin imports, including hard links, missing paths and failed
      lookups. I test absent-destination probe cleanup in private directories.
      MAC `task_419c47bdc8fc42e4b52eb6af1a0e9a71`.
      My AOT suite passes 822 checks. Compiler acceptance remains failing at
      builtin `path_normalize` (import 24). Evidence:
      `docs/evidence/aot-builtin-identity.md`.
- [x] **AOT filesystem predicates.** I adapt builtin file/directory existence
      checks with exact signatures and exercise files, directories, missing
      paths and followed/broken links. I rerun compiler acceptance.
      I also adapt builtin removal/rename and test only private targets.
      MAC `task_419c47bdc8fc42e4b52eb6af1a0e9a71`.
      My AOT suite passes 807 checks. Compiler acceptance still fails at
      builtin `file_compare_identity` (import 22). Evidence:
      `docs/evidence/aot-filesystem-basic.md`.
- [x] **AOT builtin text writer.** I preserve both string arguments and
      return failure on short writes or failed close. I exercise generated
      executables and rerun compiler acceptance before completing this item.
      MAC `task_419c47bdc8fc42e4b52eb6af1a0e9a71`.
      My AOT suite passes 771 checks. Compiler acceptance still fails at
      import 20 (`file_exists`). Evidence: `docs/evidence/aot-builtin-text-writer.md`.
- [x] **Text write error parity.** I fix interpreter and generated-native
      writes/appends that ignore write errors, and VM/module writers that
      ignore close errors. I require injected short-write and close failures
      across these paths, not only successful regular-file writes.
      MAC `task_46cb33d875724130a6f767d998f3014b`.
      Seven production-body fault probes, 771 AOT checks and the text-file
      integration regression pass. Evidence:
      `docs/evidence/text-write-error-parity.md`.
- [x] **AOT builtin text reader.** I preserve streaming text reads, empty
      results on I/O errors and embedded-NUL rejection in generated C. I test
      real files and keep builtin and artifact bindings distinct.
      MAC `task_419c47bdc8fc42e4b52eb6af1a0e9a71`.
      My AOT suite passes 752 checks, including FIFO and injected I/O errors.
      Compiler acceptance remains failing at builtin `file_write` (import 19).
      Evidence: `docs/evidence/aot-builtin-text-reader.md`.
- [x] **AOT scalar filesystem imports.** I adapt the known filesystem
      string, boolean and integer contracts through exact artifact bindings,
      preserving argument order and native return widths. I execute real
      library tests and rerun full compiler acceptance.
      MAC `task_419c47bdc8fc42e4b52eb6af1a0e9a71`.
      My AOT suite passes 710 checks; full compiler acceptance remains failing
      at builtin import 18 (`file_read`). Evidence:
      `docs/evidence/aot-scalar-filesystem.md`.
- [x] **AOT owned filesystem adapter.** I bind `fs_walkdir` to its absolute
      artifact path, check its array ABI and release entry point, copy returned
      strings into AOT storage, then release the foreign result. I test exact
      binding and failure cases before using it in compiler acceptance.
      MAC `task_419c47bdc8fc42e4b52eb6af1a0e9a71`.
      My AOT suite passes 647 checks. Compiler acceptance now stops at
      `path_normalize`; remaining adapters stay open under the same task.
      Evidence: `docs/evidence/aot-owned-walk-adapter.md`.
- [x] **Owned walk-result release.** I add an opt-in C release operation for
      unmodified, exclusively owned `fs_walkdir` results, freeing their copied
      strings before the array. Existing callers retain current behavior.
      I test escaped copies and shared-array refusal before AOT adapter use.
      MAC `task_419c47bdc8fc42e4b52eb6af1a0e9a71`.
      My release probe and directory-walk gate pass (one host-limit skip).
      Evidence: `docs/evidence/walk-result-release.md`. AOT adaptation remains open.
- [x] **AOT artifact boundary diagnostics.** I distinguish unsupported exact
      artifact imports from builtin ABI mismatches and pin the filesystem-array
      boundary against name-only rebinding. This does not implement the native
      array adapter. MAC `task_419c47bdc8fc42e4b52eb6af1a0e9a71`.
      The AOT suite passes 617 checks; One-IR acceptance still fails at the
      artifact adapter. Evidence: `docs/evidence/aot-artifact-boundary.md`.
- [x] **One-IR compiler execution baseline.** I exercise the current full
      compiler through bytecode emission, AOT C translation, native compilation
      and a compiled hello program. I retain bounded subprocess groups and
      exact output checks. The 75-commit `8299138f` branch remains unmerged
      until its useful changes are reconciled with current safety contracts.
      Release parent MAC `task_cffdafd16e641ac417ccfddb962534b9`.
      `make test-one-ir-compiler` now exposes the first blocker: compiler
      bytecode emits, but `nvm2c` refuses the `fs_walkdir` import's host ABI.
      Execution acceptance remains failing under MAC
      `task_419c47bdc8fc42e4b52eb6af1a0e9a71`. Evidence:
      `docs/evidence/one-ir-compiler-baseline.md`.
- [x] **Union resource propagation.** I propagate ownership obligations
      through named union payloads and mixed record/union cycles, preserving
      module-owned lookup. I require classifier tests; generic substitution,
      tuples and control-flow enforcement remain open. MAC
      `task_91ae827be4154eaa8f22698aeecc8cf1`.
      Record/union classifier checks pass; evidence:
      `docs/evidence/union-resource-classification.md`.
- [x] **Nested-record resource classification.** I classify ordinary records
      containing resource records without recursive depth limits, including
      cycles. I test deep graphs and keep full path-sensitive ownership and
      union/tuple/collection propagation separate. MAC
      `task_91ae827be4154eaa8f22698aeecc8cf1`.
      The dedicated unit gate passes cyclic, 300-record and module-owned
      field cases. Evidence: `docs/evidence/resource-classification.md`.
- [x] **Affine parity gate coverage.** I test positive execution and both
      existing ownership rejections on the C seed as well as both bootstrap
      stages. Rejections must carry ownership diagnostics and preserve prior
      output. I replace the predictable temporary path with private fixtures.
      Passing the old gate did not establish C-seed rejection parity.
      The corrected gate fails two C-seed cases: unresolved ownership is
      accepted, and use-after-move fails only a shadow assertion. Both bootstrap
      stages pass the limited cases. Checker acceptance remains open under MAC
      `task_91ae827be4154eaa8f22698aeecc8cf1`; evidence:
      `docs/evidence/affine-parity-baseline.md`.
- [x] **Affine contract branch reconciliation.** I retain the current
      contract, which matches `a08341e5`, and reconcile that head plus the older
      `cb99b86e` contract. This records design ancestry, not implementation
      conformance; the C-seed recovery prototype remains under review.
      MAC `task_4ac22044ffda9f93b336a85573293bc2`.
      Both changed documents match exactly; evidence:
      `docs/evidence/affine-contract-branch-reconciliation.md`.
- [x] **Compact-schema branch reconciliation.** I retain the compact operand
      aliases and tests already integrated by PR #139 while reconciling
      `09187898`. I clarify that schema checks do not establish runtime compact
      encoding support and require the schema gate. MAC
      `task_b7c534c9150ee4bb4b8bca62b37b50dd`.
      Schema generation is current and 33 tests pass. Evidence:
      `docs/evidence/compact-schema-branch-reconciliation.md`.
- [x] **Assembler snapshot branch reconciliation.** I verify that `24ac886b`
      exactly matches integrated tree `dfa1aca3`, retain newer capture rules,
      and rerun its include-phase, restored-input and search-order tests.
      MAC `task_92e6817607cf4071ab614289911a9a41`.
      Three tests pass in 263.852 seconds. Evidence and remaining-head inventory:
      `docs/evidence/assembler-branch-reconciliation.md`.
- [x] **Wrapper-publication branch reconciliation.** I retain the staged
      publication implementation already integrated from `c56c6e7e`, including
      newer callback/array link dependencies, while reconciling `df031115`.
      I require wrapper unit and adversarial publication tests. MAC
      `task_0750c33a06a14dd39baf4d3e77e37a0d`.
      Five unit tests and seven publication methods pass. Evidence:
      `docs/evidence/wrapper-branch-reconciliation.md`.
- [x] **VM shadow completion handshake.** I reject foreign `exit(0)` before
      shadow execution returns, preserving prior bytecode. I require a private
      close-on-exec completion channel as well as a successful child status.
      MAC `task_9d9eefa909be4990be0151bed7439953`; both early-exit regression
      cases fail before the fix, then all 40 bytecode-shadow methods pass.
      Evidence: `docs/evidence/vm-shadow-branch-reconciliation.md`.
- [x] **VM-shadow worker reconciliation.** I compare `a1399362` with my
      current graph-wide shadows, typed math lowering and signature checks.
      I retain dependency shadows by default and current publication guards,
      and require the complete bytecode-shadow test module before recording
      ancestry. MAC `task_4c6ff6e9986a49d6a01701a66b8842d6`.
- [x] **PR #295 reconciliation.** I integrate main's verifier cleanup and
      ownership-propagation regression while retaining my allocation-counted
      rejection tests and unknown-effect cleanup. I require schema and verifier
      gates before recording the reviewed worker head as ancestry. MAC
      `task_69dab2ee6f1a4c96845ba7139cfc360f`.
      Schema, 95 verifier tests and allocation-counted checks pass, including
      the new retain/release path. Evidence: `docs/evidence/pr-295-reconciliation.md`.
- [x] **Record-alias branch reconciliation.** I retain recursive record
      copies before binding replacement rather than restore symbol-scan
      reference counts from `9698ce08`. I retain its direct-call alias
      regression alongside current nested-record shadows and run evaluator
      and environment gates. MAC `task_1ac1c2fd6bc04b5e81932553901b0d4b`.
      Bootstrap, evaluator and 32 environment checks pass. Evidence:
      `docs/evidence/record-alias-branch-reconciliation.md`.
- [x] **Bootstrap branch reconciliation.** I retain behavioral invalidation
      checks and direct self-hosted stage dependencies while reconciling
      `d152181d`. I expose the existing suite as `test-bootstrap-dependencies`
      and run it from quick and full test entry points. MAC
      `task_85191b435c95480d9db52e3671d1e740`.
      Eight behavioral tests pass in 17.126 seconds. Evidence:
      `docs/evidence/bootstrap-branch-reconciliation.md`.
- [x] **String-search branch reconciliation.** I retain my shared byte-offset
      search implementation and broader source fixtures while reconciling
      `07d38702`. I retain its missing first-search null-needle assertion and
      require runtime units, native stages, emitted C and VM execution.
      All focused gates pass; evidence:
      `docs/evidence/string-search-branch-reconciliation.md`.
      MAC `task_a9b152cf5299491694d96a2385527e98`.
- [x] **FFI argument branch reconciliation.** I retain current ABI handling
      and the shared foreign-argument ceiling while reconciling the older
      float-refusal and execution-path argument-limit branches. I require
      mixed-signature execution, VM over-limit rejection and protocol gates;
      I do not restore the obsolete ten-argument dispatch limit or float ban.
      Evidence: `docs/evidence/ffi-argument-branch-reconciliation.md`.
      Release parent MAC `task_cffdafd16e641ac417ccfddb962534b9`.
- [x] **Verifier branch reconciliation and rejection cleanup.** I retain
      current signature-aware stack propagation and valid alternate paths after
      returns while reconciling the older stack/range/container heads. I fix
      unfreed verifier work arrays on rejection and require allocation-counted
      failure tests plus schema, verifier and container gates. MAC
      `task_69dab2ee6f1a4c96845ba7139cfc360f`.
      Allocation-counted rejection/failure/success checks pass, along with
      94 verifier tests, 2632 NanoISA checks, 29 v2 container checks and schema
      verification. I retain both old heads in ancestry without restoring
      their weaker stack or terminator policy. Evidence:
      `docs/evidence/verifier-branch-reconciliation.md`.
- [ ] **Full branch inventory and reconciliation.** I inspect local-only and
      remote heads, not just open PRs. At `1c7105a1`, 21 heads remain outside
      integration ancestry. Six contain only patch-equivalent commits already
      in history; I reconcile those without source replacement. The other 15
      require content review, including affine ownership, verifier/FFI work,
      lease snapshots and the 75-commit 4.6 frontend branch. Evidence:
      `docs/evidence/branch-inventory-1c7105a1.json`. Release parent MAC
      `task_cffdafd16e641ac417ccfddb962534b9` remains open.
      - [x] I retain the six patch-equivalent heads as merge ancestors without
        source replacement. Current verifier and co-process protocol gates
        pass. Evidence: `docs/evidence/patch-equivalent-branches.md`.
- [x] **PR #269 reconciliation.** I retain my converged AOT call facts and
      stronger aggregate rejection checks while reconciling the older flat
      record/variant patch. I require the structured-C suite and keep declared
      layouts, nested fields and linked aggregate metadata open under MAC
      `task_a4fde0d59ad24fe18c285a76ad58c176`.
      The suite passes 609 checks, including an added integer/record parameter
      conflict with an unused argument. I retain production code unchanged.
      Evidence: `docs/evidence/pr-269-reconciliation.md`.
- [x] **PR #267 reconciliation.** I compare its actual external-capture
      failure patch with my newer fail-closed implementation, retain current
      diagnostics and expanded recovery checks, and verify concurrent compiler
      isolation plus cold/warm capture failure before merging its ancestry.
      MAC `task_443e8107d0ff4350999e0d5186a809f1`.
      Its base is already integrated and my newer implementation subsumes the
      patch. Six isolation/capture methods pass in 114.378 seconds; I retain
      current source unchanged. Evidence: `docs/evidence/pr-267-reconciliation.md`.
- [x] **Release-gate string documentation.** I document `str_last_index_of`
      and reconcile the string section count with the builtin registry.
      `tests/check_stdlib_docs.sh` passes for all 156 builtins.
- [x] **MAC shell argument safety.** I quote ledger arguments as data and
      test metacharacters against a fake CLI; titles must not execute shell code.
      Native and VM compilation, dependency/root shadows and execution pass
      exact-argument checks for quotes, substitutions, newlines, Unicode and
      option-like values across create, close, show and list. No live ledger
      writes occur in these tests. Positional data follows `--`; the raw
      `exec_command` API intentionally remains a shell-code execution API.
      MAC `task_c3e8254ad9f8ce84143f59ca306e24fb`.
- [x] **Process command length.** I reject or safely accommodate commands beyond
      the process runner's fixed command buffer, with boundary tests.
      My module runner already uses direct shell invocation and file-backed
      capture. I share that implementation with the VM builtin, interpreter
      and generated native helper; the latter two currently drain pipes
      sequentially. I test long commands and simultaneous stdout/stderr.
      All four paths now share `runtime/process_capture.h`, eliminating the
      VM's 4096-byte command buffer and sequential-pipe readers. Native/VM
      builtin and module tests pass 16 KiB commands and 128 KiB per stream.
      VM tests reject commands beyond the host argument limit without running
      a truncated prefix, and check signal status, null input and NUL output.
      Interpreter, transpiler and MAC boundary gates pass. Strict dispatch
      remains 175 selected, 172 identical, three failures and zero skipped.
      Command deadlines, output quotas and array/string ownership remain
      separate runtime boundaries; this does not establish process isolation.
      MAC `task_c3e8254ad9f8ce84143f59ca306e24fb`.

- [x] **Release-gate launcher capture.** The full suite reaches 219 passing
      implementation tests and one compile failure in
      `tests/unit/test_launcher_capture.nano`. I diagnose its saved compile log,
      preserve child-capture and termination assertions, and rerun the gate.
      Two reproductions exposed shell-normalized exit status 143. I use
      `printf; exec sleep` so the observed PID receives the signal directly.
      Eight consecutive native compilations with dependency/root shadows and
      executable runs pass. The complete suite remains a separate release gate.
- [x] **Launcher termination output.** I drain child pipes again after reaping
      a stopped child. `pm_kill` currently drains only immediately after sending
      SIGTERM, so output produced by a termination handler can be lost before
      descriptor closure. I require a handler-output regression, not a sleep.
      Main's incoming handler fixture fails before publication: it signals
      before the child confirms handler installation. I require a readiness
      handshake, per-child log path, both streams and successful reaping.
      The repaired native launcher compile/shadow/run gate passes. A focused
      bytecode import runs all process-manager dependency shadows, including
      the handler regression, and executes successfully. Full VM launcher
      polling remains the separate aggregate defect below.
- [x] **VM launcher poll aggregates.** The native launcher regression passes;
      bytecode shadows fail in `pm_poll` with `AGG_GET field 2 is unavailable`.
      I trace its representation and require full native/VM launcher coverage.
      The opcode trace shows `pm_new` emits its empty children array as integers;
      pushing a record then reads its pointer back with an integer tag. I retain
      the declared element type on empty array literals in record fields.
      The full VM launcher compile/shadow/run regression now passes. My focused
      record/string/float/bool field test passes compilation, shadows and execution
      on native and bytecode paths. Typechecker, NanoVirt and 272,379 VM checks
      pass. MAC currently rejects my claim with `agent_status_unavailable`;
      verified code completion does not imply ledger closure.
      MAC `task_26736715909643ae7df9851939b75e71`.
- [ ] **5.0 release integration.** I reconcile the audit-contract branch with
      main, preserve original-file diagnostics and immutable native-cache
      behavior, run clean build/tests and documentation gates, then merge and
      publish 5.0. The fleet dispatch hold does not gate this user-authorized
      release. Unfinished roadmap items remain explicitly unfinished.
      - [x] I reconcile PR #283's complete one-commit timeout change into the
        integration branch, retaining launch diagnostics and Make-expanded
        regression coverage. I verify its standalone test entry point and
        record merge ancestry; this does not merge the release into main.
        All three regression methods pass from the repository and an outside
        working directory. [Review evidence](evidence/pr-283-reconciliation.md).
      - [x] I reconcile PR #286's daemon path guard and test target. I retain
        private test paths and compile-time test injection, avoid expanding
        production configuration solely for testing, and require the boundary
        test from the daemon gate before recording integration ancestry.
        All three socket-path boundary cases pass via `make test-vmd-server`.
        [Review evidence](evidence/pr-286-reconciliation.md).
      - [x] I reconcile PR #282's physical-newline fix without counting twice
        through the existing shared scanner. I retain its regression, add
        f-string position assertions, rebuild Stage2 and run module-binding
        tests before recording merge ancestry. Both stages rebuild, their
        smoke checks pass and all nine module-binding tests pass on Darwin.
        [Review evidence](evidence/pr-282-reconciliation.md).
      - [x] I reconcile PR #281's function-value signature checking. I retain
        recursive returned-call checking and AST-owned metadata, check typed
        variable aliases, and run typechecker and interpreter regression gates.
        Both gates pass after rebuilding the compiler stages.
        [Review evidence](evidence/pr-281-reconciliation.md).
      - [x] I reconcile both commits in PR #284, preserving the existing source
        identity guard and replacing predictable-path fixtures with my private
        alias regressions. Stage2 passes all three methods after bootstrap;
        the Stage1 override also passes from outside the repository.
        [Review evidence](evidence/pr-284-reconciliation.md).
      - [x] I reconcile PR #285's C-seed artifact/diagnostic collision guard.
        I require preservation checks for every diagnostic destination and
        detect case aliases through the filesystem, not the OS name. I reject
        dangling diagnostic links rather than treating them as absent entries.
        The dangling-link regression reproduces success before the fix; all
        four destination methods and the compiler-contract gate now pass.
        [Review evidence](evidence/pr-285-reconciliation.md).
        MAC `task_f38c6358bf944c218f179daf1490ebe2`.
      - [x] I reconcile PR #273's full manifest-path change without replacing
        existing source snapshots or atomic publication. I retain its quoting
        regression against the production helper in a private directory and
        run manifest-backed single/multi/private-source path tests.
        The rebuilt `test-module-metadata` gate passes, including the quoting
        test and all five end-to-end path variants.
        [Review evidence](evidence/pr-273-reconciliation.md).
        MAC `task_8c10a946dfa14d92b491fd80f4635187`.
      - [x] I reconcile PR #274's checked interpreter FFI dispatch ancestry.
        I retain the existing implementation, later handler-return propagation
        and dependency-shadow default; I rerun FFI, interpreter and foreign
        language-claim regressions before marking integration complete.
        FFI and interpreter gates pass after rebuild. The source-level foreign
        gate initially failed: `array_get (map [0.0] erf) 0` was inferred as int,
        rejecting the float assertion before shadows. The frontend correction
        below now passes all three foreign claim methods without changing their
        assertions. Full map/backend acceptance remains separate release work.
        MAC `task_17fe744141024df08c2ef3de7599865a`.
      - [x] I reconcile PR #266's 39 transpiler-shadow lexer-call repairs,
        already present in my integration source. I preserve newer code and
        record the head as a merge parent. Explicit parser and extern-emitter
        entry-point tests pass; the full component-execution audit remains
        open because the PR does not implement it.
        [Review evidence](evidence/pr-266-reconciliation.md).
      - [x] I reconcile main's PR #293 squash `ce0e8e95` after integrating
        the worker head and additional fixes. Complete tree comparison with
        `c751d2e8` is identical; I record main's commit as a merge parent and
        retain my tested integration source unchanged.
        [Review evidence](evidence/pr-293-reconciliation.md).
      - [x] I isolate the daemon integration script from user processes and
        shared endpoints. I remove ambient process-name killing, fail selected
        compilation/execution errors and daemon death, and bound waits with
        failure-injection tests. MAC `task_999bf1a1ab96472294660aa2b19cae9a`.
        I first add a shared explicit socket override and a client mode that
        forbids automatic daemon launch. My runner then owns a foreground
        child and private directory, applies deadlines, and checks every
        selected compilation, execution and output comparison. I exercise
        startup, execution, timeout and cleanup failures before the real gate.
        My injected failures and native endpoint checks pass; the rebuilt
        `make test-nanovm-daemon` gate passes all eight selected programs with
        no skips. Evidence: `docs/evidence/daemon-gate-isolation.md`.
      - [x] I keep daemon execution state off bounded client-thread stacks.
        The strict gate exposes a Darwin `SIGBUS` before even a ping is
        handled. The crash report identifies `___chkstk_darwin` in
        `client_thread` and reports that the thread stack size was exceeded.
        I use checked owned VM-state allocation and verify real connections
        and all eight daemon corpus programs. MAC
        `task_ed9fc711786a459abb5bafb69ea33f3e`.
        The rebuilt daemon now answers the readiness ping and executes all
        eight programs successfully. This is not a concurrent-client stress
        test or proof of every allocation-failure path.
      - [ ] I isolate the co-process lifecycle gate before running it: its
        EXIT trap currently kills ambient `nano_cop` and `nano_vmd` processes.
        I replace process-name cleanup with owned handles and private sockets,
        audit masked failures, and test preservation of unrelated processes.
        MAC `task_a02a66101b184e6eaa3e61480079f300`.
        Its claimed crash-recovery case never kills a worker, and global
        process-name counts cannot prove per-client isolation or lazy launch.
        I require observed owned-worker identity and an actual injected crash,
        not merely successful repeated calls or matching ambient counts.
      - [ ] I require semantic rejection evidence in the self-hosted shell
        suite. Its negative-test loop currently counts any compiler failure,
        including timeout or launch failure, as a pass and discards diagnostics.
        I preserve the corpus and distinguish expected diagnostics from
        infrastructure failures with injected regressions. MAC
        `task_4f84d7b8485a467da3909f79e2417233`.
      - [x] I make dispatch-equivalence coverage explicit: unexpected compile
        failures and zero comparisons fail, expected exclusions are reported,
        and compilation/VM executions have deadlines and retained diagnostics.
        I select every root tests/*.nano file with no implicit exclusions.
        Compilation and each VM run have a 60-second process-group deadline;
        byte output and exit status are compared without stripping newlines.
        Synthetic tests cover empty coverage, compile/artifact failures,
        timeouts, signals and output/status mismatches. The strict real run
        reports 173 selected, 157 identical, 16 failed, zero skipped. The gate
        correctly fails; the formerly hidden corpus failures remain below.
        MAC `task_31ecd1c81d6740d0e856a117ce786b6a`.
      - [x] I resolve the 16 compilation/shadow failures exposed by the strict
        dispatch corpus. My expanded corpus passes all 175 verified and
        equivalent programs with zero exclusions. The broader effect lifecycle
        obligations below retain their independent unfinished scope.
        MAC `task_a954001005513e9f99272f3b6275f817`.
        - [x] I infer an unqualified handler from all its operation names,
          requiring one unique effect rather than declaration-order selection.
          I reject duplicate clauses and incorrect handler parameter counts.
          Typechecker regressions cover ambiguity and order independence.
          Typechecker/effects unit tests pass, and both native and VM negative
          suites pass 36/36 with a pinned ambiguity diagnostic and no artifact
          publication. MAC `task_76faf75ded541840965d2d4f20d4484e`.
        - [ ] I implement effect dispatch across native and VM execution.
          - [x] I preserve ordinary module inference context while assigning
            source identity to native effect capture bindings. An importer of
            `src_nano/compiler/error_messages.nano` compiles and runs, and my ten
            native effect checks pass. MAC
            `task_7ecadba95e9445098841f83707a1d7c6`.
          - [x] I classify perform, handle, and await as value expressions in
            my shared parser and verify native handler captures, nesting,
            lexical return, and cleanup. I register imported effect declarations
            before checking module functions and keep native dynamic handler
            state shared across compilation units. I isolate unsafe-block
            cleanup inside its emitted C scope. Three shared native cases and
            ten native/runtime cases pass, including the latter under ASan/UBSan
            with leak detection disabled. My evidence is in
            `docs/evidence/native-effects-linux.md`. MAC
            `task_36491565f7db6038fb0b1591f6164c36`.
          - [ ] I route interpreter handler returns to the lexical function's
            active call, preserving the destination through intervening helper
            calls and cleanup. I test final-expression resumption separately,
            expression ordering, string results and repeated handler unwinding.
            The lexical call destination and scalar call/operator propagation
            now pass interpreter regressions, including nested handlers and
            final-expression resumption. Aggregate constructors, match subjects
            and guards, loop bounds, assertions and other value consumers now
            propagate the return before later work. Tests check interrupted
            string/record arrays preserve caller storage. Evaluator, effects,
            typechecker, parser and transpiler gates pass; the changed evaluator
            and evaluator tests pass ASan/UBSan with other linked objects
            uninstrumented and leak detection disabled. Map, filter and reduce
            now propagate callback returns and discard partial output, with
            twelve static/dynamic-array escape/resumption cases passing.
            One hundred nested synchronous async calls propagate the return,
            allocate exactly 200 private handles and release them all; evaluator
            and scheduler gates pass. The handwritten `coro_spawn`, `coro_done`
            and `coro_result` special paths are not registered by the frontend;
            they are not evidence of a working owned-task API. Their argument
            propagation belongs with the task-lifecycle implementation below.
            General propagation and foreign callback boundaries remain open.
            MAC `task_67e5e620d75a413b99753c7cdbde1f48`.
          - [x] I preserve declared handler parameter metadata for nominal
            field access, typed array reads and function signatures. I test
            valid uses and incompatible body bindings before runtime lowering.
            Nested array and scalar callback signatures retain their declared
            types; handler names do not inherit unrelated outer record metadata.
            Parser, typechecker, effects and evaluator gates pass. This does
            not complete perform argument/result compatibility or dispatch.
            MAC `task_0e712110ced84d10b47bd50d54d434b3`.
          - [x] I preserve handler capture identity in VM lowering. Existing
            CLOSURE_NEW copies captured values and STORE_UPVALUE updates only
            that copy; lowering handlers directly to those closures would not
            preserve interpreter mutations of enclosing locals. I require
            caller/handler read-write visibility, nested routing and scope-exit
            cleanup tests before calling that lowering equivalent.
          - [x] I carry ordered perform arguments through one parser, the AST,
            checking and interpreter dispatch. I evaluate every argument before
            introducing handler bindings, test zero and multiple arguments,
            and preserve nested expression arguments and source order. Native
            signatures remain consistent with the implemented dispatch ABI.
            Parser, typechecker, effects, interpreter and transpiler gates pass.
            The required cross-backend gate now also checks ordered multiple
            arguments, caller-name shadowing and zero arguments. Both native
            and VM cases pass real shadows and executed assertions. Native
            foreign-return boundaries and unsupported nvm2c effect opcodes
            remain explicit. MAC `task_36491565f7db6038fb0b1591f6164c36`.
          - [x] I resolve performs through the frontend effect declarations,
            reject unknown operations and incorrect scalar arguments/arity,
            and report the operation's declared result type in expressions.
            I test statement/expression positions and forward declarations.
            Typechecker, effects and interpreter gates pass; native and VM
            negative compiler suites pass 37/37 with a pinned argument-type
            diagnostic and no artifact publication. This checks scalar base
            types, not full nominal/aggregate signature compatibility.
            MAC `task_6a559f3c44e2f1decb45662b2d81e8df`.
          My initial native emitter ignored handlers and NanoVirt rejected
          effect operations. Both now execute synchronous handlers, captures,
          ordered arguments and lexical returns. `test-effect-execution` runs
          real shadows and state-changing operations on both backends; the
          native and VM evidence records their cleanup and foreign boundaries.
          General interpreter propagation remains separate work below.
          - [x] I connect unqualified handler ASTs to interpreter frames for
            single-argument operations. Interpreter/effects unit gates pass.
            Executed tests check state changes, nearest-handler precedence,
            restoration of the outer handler and an empty stack after repeated
            calls. Native and VM acceptance now pass real shadows and execution;
            my initial void-local and rejected-operation failures are repaired.
          - [ ] I complete operation argument/result handling and control-flow
            rules across backends. My creator chose: a handler-arm `return`
            exits the function containing the handler; the final expression
            supplies the operation result and resumes `perform`. I test this
            across helper calls, nested handlers and early scope cleanup.
            Argument checking still needs nominal identities and recursive
            aggregate/function signatures, including preserved result metadata;
            interpreter scalar paths now preserve lexical return destinations,
            but complete expression propagation and handler break/continue rules
            remain unchecked. These boundaries remain open,
            as does native/VM dispatch itself.
          - [x] I emit void-valued locals without C storage, preserving each
            initializer and assignment side effect. I test inferred/annotated
            bindings, reads and returns with real native and VM execution.
            The same fixture exposes VM STORE_LOCAL underflow after a void
            call; I materialize its void value for storage and discard a bound
            void value when returning from a zero-result function.
            Native/VM compile, shadows and execution pass, alongside native
            transpiler gates and 69 NanoVirt code-generation tests. The effect
            acceptance artifact now compiles natively but fails its executed
            state assertion: native performs are still no-op stubs. VM still
            rejects effect operations. MAC `task_dc995297abc3c89937cd3def9c232ccc`.
        - [x] I fail self/ancestor await cycles and propagate failed awaits
          to running callers. I keep completion/error terminal so a callback
          cannot overwrite failure with a later completion. I test self waits,
          nested cycles, stale handles and both scheduler entry paths.
          Scheduler/interpreter gates pass, as does the standalone scheduler
          ASan/UBSan suite with leak detection disabled. Typed task errors and
          resumable async remain separate work below.
          MAC `task_cae2c16891b91e98fdb4bb9ee2074306`.
        - [x] I retain scheduler handles and completed results until explicit
          release, reject active/pending release and invalid spawn callbacks,
          and reject ID exhaustion without wrapping. Synchronous interpreter
          async calls release private handles after retrieving results. I test
          retention, capacity, reuse and stale IDs; language-level owned task
          values and result ownership across backends remain unfinished.
          Scheduler and interpreter tests pass. The interpreter regression
          executes 130 async calls, checks allocated IDs and verifies private
          slot release; registering a program alone did not exercise its main.
          Standalone scheduler ASan/UBSan tests pass with leak detection off.
          MAC `task_ad179bc283020e72e3ff89b67c789d94`.
        - [x] I keep scheduler slots pinned until their callbacks return,
          even after explicit completion or error. I test spawn and nested
          await after a terminal transition through step and await entry paths.
          Scheduler, optimization and interpreter gates pass. The standalone
          scheduler suite also passes ASan/UBSan with leak detection disabled;
          this does not establish leak freedom or resumable execution.
          MAC `task_859a6d28a6a9923b9a144520bf0b2b3f`.
        - [ ] I implement resumable async execution across native and VM
          backends. My current CPS walker validates only selected syntax;
          it does not create continuations. My scheduler runs callbacks to
          completion and yield is a no-op. Passing transparent scalar awaits
          alone will not complete this item.
          I replace the old coroutine-spawn smoke test that ignores typecheck
          failure with a checked language-level task lifecycle test as part
          of implementing owned task values.
        - [x] I reject non-string format templates before native or bytecode
          emission. I test literals, inferred bindings and function results,
          and require a type diagnostic without publishing an artifact.
          Typechecker tests pass; the focused negative fixture passes both
          CLIs. The full native negative suite passes 35/35; NanoVirt passes
          34/35 and exposes the array-index defect below.
        - [x] I reject non-integer array indices in `at` and `array_get`
          before emission. NanoVirt accepts the string-index negative fixture;
          native rejection alone does not establish frontend type safety.
          MAC `task_fe0c5cdbdb647732ea9084281c24261f`.
          I preserve nested array element metadata through indexed access;
          stricter validation exposed a false rejection of three-dimensional
          arrays when the middle access was incorrectly inferred as `int`.
          Parser/typechecker tests and 69 NanoVirt checks pass. Both negative
          suites now pass 35/35 with a pinned index-type diagnostic. Strict
          dispatch returns to 175 selected, 171 identical, four failures and
          zero skipped; async, effects and formatting remain unfinished.
        - [ ] I reconcile formatting conversions across my interpreter,
          native emitter and VM before claiming variadic formatting parity.
          My interpreter formats whole floats without the native `.0` suffix;
          aggregate substitutions have no consistent conversion contract.
          - [x] I lower VM formatting to a runtime call with a typed string
            array, not native varargs. I preserve placeholder scanning and
            check empty, missing, extra, UTF-8 and long substitutions in
            runtime tests and executable/shadow tests.
            Native and VM compilation, shadows and execution pass. All 21 VM
            builtin tests and 69 NanoVirt checks pass. Strict dispatch reports
            175 selected, 172 identical, three failures and zero skipped;
            async and effects remain. Conversion parity above remains open.
          - [x] I emit native assertion diagnostics as literal text, never as
            printf formats. Formatting assertions exposed `%s`/`%d` in source
            text being interpreted as native conversion directives.
            The integrated native regression compiles and executes a failing
            assertion containing `%s`, `%d`, `%n` and `%%`, preserving them
            literally and returning status 1. Transpiler tests pass.
            MAC `task_108e132bac3db9598c8431fca94108c9`.
        - [x] I preserve inherited fields in VM record spreads, evaluating the
          source once and resolving inherited fields against its own layout.
          I check overrides and unchanged source records on native/VM paths.
          Native execution exposed the same omitted-source defect in my C
          emitter; both emitters now pass compilation, shadows and execution
          for known nominal layouts, including reordered source fields and
          all-fields-overridden cases. My 69 NanoVirt checks and transpiler
          unit tests pass. Strict dispatch reports 175 selected, 171 identical,
          four failures (async, effects and formatting), and zero skipped.
        - [ ] I resolve open-record field access without choosing the first
          nominal layout that contains its name, including reordered fields.
          MAC `task_720689f69816542431a75d2e27d122c7`.
        - [x] I import the filesystem module in the path/walk fixtures instead
          of declaring unbound externs. I assert path results and walk a private
          known-content directory, then verify native/VM default shadows and runs.
          Both fixtures pass with dependency shadows enabled. Traversal checks
          both files without assuming order and removes its private fixtures.
          Strict corpus: 175 selected, 170 identical, five failures, zero skipped.
        - [x] I lower generic map_remove/map_clear with void stack effects and
          alias-preserving mutation, using declared type tags for direct local
          constructors. Int/string map regressions pass native/VM compilation,
          default shadows and execution: deletion, reinsertion, extraction,
          alias-visible clear and repeated empty clear. All 69 code-generation
          tests pass. Strict corpus: 175 selected, 168 identical, seven failures,
          zero skipped. This does not establish all constructor contexts.
        - [x] I carry map constructor key/value types through returns, globals,
          fields, arguments and nested generic contexts instead of defaulting to
          string/int tags. I retain checked scalar tags on each constructor
          and complete record field annotations on the parsed declaration.
          All four int/string pairs pass VM values, including empty extraction.
          See `docs/evidence/map-constructor-contexts.md`.
          MAC `task_f4e1871af407805219770d7620d58349`.
        - [x] I keep native map cleanup inside the binding scope. Match arms
          now emit cleanup before leaving their lexical C block, and my full
          four-pair native/VM map fixture runs without removing selected arms.
          Fresh bootstrap and 36 map/effect/selected-ownership methods pass;
          alias/return controls and 20 opaque finalizers establish the bounded
          behavior in `docs/evidence/native-map-lexical-cleanup.md`.
          I preserved the recovered fleet patch and removed its proposed
          early-return changes. Map deallocation and borrowed-result lifetime
          remain separate obligations below.
          MAC `task_1edadd5eb33a445d9bf6516744bc405e`.
        - [x] I retain map key/value metadata when a native map operation takes
          an ordinary direct call or checked callback result. I capture receiver,
          key and value expressions once in source order. All four scalar pairs,
          nested receivers, returned expressions and invalid-context artifact
          preservation pass native/VM checks. Fresh bootstrap and 24 adjacent
          methods pass; `docs/evidence/native-returned-map-metadata.md` records
          the explicit native-only free boundary. Cleanup/lifetime is unchanged.
          MAC `task_e018b78bc20a47d18619fce55a20e567`.
        - [ ] I specify and lower explicit `map_free` across NanoISA backends.
          Native emission accepts it, but NanoVirt currently reports an undefined
          function while compiling shadows. The native-only free-order control
          in returned-map tests is explicit; I require a lifetime/alias contract
          and paired scalar-tag coverage before claiming portable free support.
          MAC `task_2f848b73acf847a79df68418b9213637`.
        - [ ] I define early-return cleanup with explicit retain/transfer of
          computed or borrowed results, preserving evaluation order and separate
          control-flow paths. Current native HashMap allocations use malloc;
          gc_release does not establish their deallocation. I keep map lifetime
          and opaque/effect cleanup gates separate from lexical visibility.
          MAC `task_195cac35e7704e56805932977512ae02`.
        - [x] I advance my bytecode for-loop index on continue, including
          unconditional and nested paths, without advancing an enclosing loop
          on an inner while continue. My new regression and both former timeout
          fixtures pass shadows and native/dual-dispatch VM execution. All 68
          code-generation unit tests pass. Other corpus failures remain open.
        - [x] I preserve filter result element representation with an empty
          source slice. Integer, float, string, boolean and empty-result checks
          pass native/dual-dispatch VM execution and default shadows, including
          append-after-empty and unchanged-source checks. All 68 code-generation
          unit tests pass. This does not establish callback signature validation.
        - [ ] I represent expression callees in the self-hosted AST and carry
          them through parser, generated runtime transport, checker and emitter.
          Stage2 rejects `((choose) 7)` before map lowering. I require nested
          returned calls, argument evaluation order/once, signature errors,
          shadows and native execution. This is a prerequisite for full map
          parity, not permission to remove returned callbacks from its tests.
          MAC `task_256337a9977f43b2baee7b26ebd66bc7`.
          I reconcile PR #293 with the newer unary-parser call site. Review
          also found computed print arguments still read as identifier IDs,
          and nested function-parameter arrows counted as generic closers.
          I repair those paths and require focused native regressions before
          accepting this prerequisite.
          The first serial gate passes the branch's positive/negative fixtures
          but rejects nested function parameters in generated C. I register
          nested function-pointer typedef dependencies before their users and
          repair arrow handling in the renderer's parameter splitter too.
          After rebuilding both stages, my explicit returned-call gate passes
          all three methods, including nested signatures and argument order.
          Schema regeneration agrees. Self-hosted NanoISA computed calls still
          fail explicitly; broader backend acceptance remains open.
          [PR #293 evidence](evidence/pr-293-reconciliation.md).
        - [ ] I preserve map transform result representation and validate
          collection callback signatures rather than merely checking each
          argument independently. MAC `task_75b340982b6cf797f29b38c1a188aab3`.
          PR #274 reconciliation reproduces a frontend failure in
          `test_qualified_and_returned_foreign_dispatch`: indexing a mapped
          float result is typed as int before the explicit shadow can run.
          I first retain declared transform result types through direct and
          local-bound map results, reject scalar input/signature mismatches,
          and test both indexing aliases. Nominal/nested metadata, empty-result
          runtime representation and complete backend acceptance remain required.
          The frontend correction passes twelve typechecker cases, the full
          typechecker/interpreter gates, and all three foreign claim methods.
          I next allocate interpreter scalar map results from the declared
          transform result, not the source representation; empty results retain
          their tag. I test static/dynamic, empty/nonempty arrays for int, float,
          bool and string results and preserve input values. VM formerly emitted
          `ARR_NEW TAG_INT`; native lowering assumed matching input/output types.
          Both now select scalar result storage from shared declared callback
          metadata, before execution. All 32 scalar input/result backend cases
          pass compilation, default shadows and artifact execution, including
          named/variable/returned callbacks and append-after-empty. All 69
          NanoVirt tests pass. Nominal/nested types and full self-hosted lowering
          still require acceptance before this item can close.
          `make test-selfhost-map-results` selects the same 16 scalar pairings
          on rebuilt Stage2, without removing named/variable/returned callbacks
          or empty-result assertions. The initial run fails all pairings at
          returned-call parsing; this gate must pass before self-host parity.
          The rebuilt gate confirms all 16 failures after bootstrap smoke checks
          pass. [Evidence](evidence/selfhost-map-prerequisite.md).
          After PR #293 integration, parsing is unblocked: the same 16-case
          gate passes int-to-int and fails the other 15 at native compilation.
          My generated `nl_map` still hard-codes `int64_t (*fn)(int64_t)`.
          I retain every pairing and fix that lowering next; a cast would not
          establish callback ABI or result-storage correctness.
          I lower scalar maps at each call site using the transform's declared
          parameter/result types. I evaluate source then transform once, read
          the input representation and allocate/push the result representation
          independently, including empty arrays. I rerun the unchanged Stage2
          matrix and returned-call gate after rebuilding both stages.
          Both stages rebuild, all 16 Stage2 scalar pairings pass, and the
          three returned-call methods remain green. A separate execution trace
          checks source then transform evaluation once and one callback per
          element. The final two-method Stage2 map run passes in 52.268 seconds.
          [Evidence](evidence/selfhost-scalar-map.md). Nominal/nested metadata,
          aggregate layouts and full signature validation remain open.
          I next reject invalid self-hosted map arity, sources and callback
          signatures before C emission, infer the result array from the callback
          result, and reject incompatible direct-map result annotations. My
          negative gate uses source-only compilation so a C compiler failure
          cannot masquerade as a typechecker rejection.
          Baseline: all eight invalid fixtures were accepted and replaced
          their prior C output. After rebuilding both stages, all eight are
          rejected with their specific diagnostic and preserved prior output;
          the inferred `array<float>` fixture runs. Both type-gate methods
          pass in 2.835 seconds. The 16 scalar pairings and execution trace
          remain green in 46.648 seconds. Evidence:
          `docs/evidence/selfhost-map-type-checking.md`.
          All sixteen interpreter combinations pass, including the complete
          rebuilt evaluator gate and a final evaluator-object rebuild/test
          after diagnostic wording changes. Logs:
          `/tmp/nanolang-map-runtime.log`, `/tmp/nanolang-map-runtime-final.log`.
        - [x] I preserve nested native comparison grouping in self-hosted
          code. I parenthesize operands through the shared binary emitter,
          retain the array-type equality shadow, and execute nested equality
          and relational comparison matrices. MAC
          `task_0f64149182a549b5abf7a6dc2547799e`.
          Both stages and the unchanged equality shadow now pass; explicit
          native comparison matrices execute successfully.
        - [x] I remove the self-hosted `array<int>` compatibility wildcard
          and retain a working bootstrap without it.
          Direct map checks do not repair aliases, calls or returns routed
          through `types_equal`. I require rejection tests across those paths
          and a complete bootstrap without this exemption. MAC
          `task_850ac4914d9b4a9cbc34f7f16dd1902c`.
          I test all 12 mismatched scalar array pairs through alias binding,
          calls, returns and reassignment before C emission. I also stop
          `apply_return_type_hint` from treating an empty scalar type-name
          field as unknown element type. Positive cases retain typed empty
          arrays, append and identity calls on all four scalar types.
          The first bootstrap reaches native shadow execution and fails the
          new equality matrix: the emitter flattens nested comparisons into
          a C comparison chain. I preserve the assertion and fix its grouping
          before rerunning this gate.
          The final rebuilt gate passes all 48 scalar mismatch rejections,
          four positive scalar-array programs and native comparison matrices
          in 15.413 seconds. Map typing passes in 3.221 seconds; all scalar
          map pairings and the trace pass in 50.427 seconds. Evidence:
          `docs/evidence/selfhost-array-compatibility.md`.
        - [x] I keep each self-hosted array literal's element references in
          one contiguous span after parsing its children. Nested parsing must
          not replace an outer element with an inner scalar. I require native
          nested-value tests and bootstrap. MAC
          `task_8f09497a0e6e47dd93b68c7d9f017626`. Both stages rebuild and the
          five-method array gate passes in 19.171 seconds. Evidence:
          `docs/evidence/selfhost-nested-arrays.md`.
        - [x] I reconcile main's PR #294 nested-indexing changes with my
          recursive checker, literal-span and empty-storage fixes. I retain
          complete element types in runtime helper selection and require the
          new fixture plus the array/map/returned-call gates after bootstrap.
          Both stages rebuild; array tests pass six methods in 21.496 seconds,
          map typing two in 2.929 seconds, scalar maps two in 48.372 seconds,
          and returned calls three in 8.422 seconds. Evidence:
          `docs/evidence/pr-294-reconciliation.md`.
        - [ ] I compare recursive and nominal self-hosted array element types
          instead of stopping at the element kind. Nested arrays, function
          signatures, enums and unions retain their distinctions across
          aliases, calls, returns and assignment; failed inference must not
          masquerade as an empty array. I require positive execution and
          source-only rejection tests plus bootstrap. MAC
          `task_3c5d8625cec144ba87efd9695239275a`.
          I first preserve nested array element descriptions through literal,
          append and map inference; render those descriptions without replacing
          them with `unknown`; and compare nested element types recursively.
          I extend the source-only alias/call/return/assignment matrix across
          nested scalar arrays and array depths before rebuilding.
          This nested-scalar slice now passes 24 boundary-specific rejections
          and native positive values. Nominal/generic execution and unknown
          compatibility remain open; I do not close the parent item.
        - [x] I preserve each remaining array level when emitting nested
          indexing, rather than selecting an integer read for an inner array.
          I require native shadows and runtime values at multiple depths.
          I also retain nested element storage tags for typed empty literals
          and propagate their child annotations through literal emission.
          MAC `task_f32d71bbad07448c8656843015e72ef3`. I test three-level
          indexing and append to empty nested integer/float arrays through
          native shadows and execution. Evidence:
          `docs/evidence/selfhost-nested-arrays.md`.
        - [x] I make C-seed shadow visibility errors prevent compiler output
          publication. A bootstrap shadow called a private parser helper: I
          printed the error but still produced Stage 1. I require a minimal
          imported-shadow regression with nonzero status and prior-output
          preservation. MAC `task_22bb4774aeb145f3bcd15d60c532a1b1`.
          I now count the access diagnostic and test all four root/dependency,
          bare/bound private-call cases, structured `E009`, unchanged output
          and successful public controls. Ten dependency-shadow methods pass
          in 29.151 seconds; both stages rebuild and all typechecker unit tests
          pass. Evidence: `docs/evidence/shadow-visibility-errors.md`.
        - [x] I provide VM left/right string trimming with the existing four-byte
          whitespace contract, checking empty/all-whitespace input, the opposite
          edge and UTF-8 preservation on native and VM paths.
        - [x] I provide VM str_join with checked length arithmetic and a single
          result allocation, covering empty arrays/elements and separators.
          I also repair interpreter str_join rejection of literal arrays, exposed
          by the new native shadows, and check its length arithmetic.
          The complete expanded string fixture passes native/VM compilation,
          default shadows and execution. All 20 VM-builtin tests, evaluator tests
          and 68 code-generation tests pass. Direct join checks include a
          10,004-byte result and malformed array-header rejection.
        - [x] I preserve literal scalar arrays in interpreter sort/reverse and
          accept them in contains/index_of, with native shadow/run regressions.
          I no longer replace nonempty literals with empty arrays. Integer sort
          and search, string/float/bool reversal, empty inputs and unchanged-source
          checks pass native compilation, default shadows and executable runs.
          Evaluator unit tests pass. VM utilities remain unfinished below.
        - [ ] I implement the missing VM sort/reverse/contains/index_of operations
          and verify element-type and original-array preservation.
          - [x] I share checked scalar sorting across interpreter, native and VM
            paths: integers/bytes/numbers ascending, false before true, strings
            bytewise, NaNs last. I test boundaries and leave the source unchanged.
            The expanded array fixture passes C-seed native and VM compilation,
            default shadows and execution. All 27 dynamic-array tests, 20 VM
            builtin tests, evaluator tests and 69 code-generation tests pass.
            Debug/release allocation-failure tests verify sorting returns failure
            without replacing the source. The strict corpus now reports 175
            selected, 167 identical, eight failures and zero skipped.
          - [ ] I complete compile-time rejection of unsupported sorting types
            and self-hosted-driver parity; runtime scalar tests do not establish
            these contracts.
          - [x] I lower reversal and first-match search to existing array
            instructions, preserving empty-array type and evaluating inputs once.
            I also preserve checked scalar element types when interpreting empty
            literals; defaulting every empty array to integers breaks append after
            reversing an empty string array.
            Native/VM shadows and execution check scalar reversal, integer
            search, empty arrays, source preservation and exactly-once left-to-right
            input evaluation. VM-only float/bool/string searches and all 69
            code-generation tests pass; evaluator tests pass too. Sorting and
            remaining cross-backend type contracts are still open.
          I also align their type contracts: existing sort silently leaves
          nonintegers unsorted, native search takes integer-only values and
          reverse mishandles unsupported element types.
          MAC `task_ee340b82c97b031407fee59f0445d418`.
        - [x] I implement the missing VM file_read_bytes bridge with byte-typed
          storage, binary/empty/missing-file checks and byte-to-string round trips.
          Native/VM compilation, default shadows and execution pass the expanded
          byte fixture. All 18 VM-builtin tests pass, including 8,193 binary bytes,
          empty/missing paths and directory read failure. This does not establish
          identical backend error handling or native-array ownership.
        - [x] I align interpreter/native byte-file storage, non-seekable reads
          and read/close-error behavior with the VM path.
          I share one streaming reader, inject partial-read and close failures,
          and compare every byte from a pipe in real shadows and artifacts.
          Byte storage exposes missing ELEM_U8 handling in interpreter at/array_get;
          I fix that indexing path as part of the same observable contract.
          The pipe regression checks all 8,193 bytes through real native/VM
          shadows and executable artifacts. Injected stream errors discard
          partial bytes and close once. All 21 VM-builtin tests and the
          interpreter/transpiler gates pass. Text readers remain separate.
          MAC `task_84458d97b4635a911eee8f2b90d66673`.
        - [x] I make text file reads safe for streams and read failures.
          I replace unchecked seeks/sizes and partial reads with a shared
          interpreter/native/VM/std-fs implementation.
          MAC `task_a3f451df37b59cefc68dfcf91e5e3f6a`.
          I use the checked byte-stream reader, return empty for embedded NUL
          data instead of truncating it, and preserve each backend's existing
          string ownership. I test pipe input and injected read/close failures
          and text-result allocation failure; binary callers use file_read_bytes.
          All 12 direct/module, native/VM, long/empty/NUL pipe combinations
          pass default shadows and artifact execution. Fault injection verifies
          empty text after read/close failure, NULL after result-allocation
          failure, and one close. Byte-stream, 21 VM-builtin, interpreter and
          transpiler gates pass. General OOM recovery and FFI string ownership
          are not established by these tests.
          - [x] I remove the native work-item formatter's 2,048-byte truncation.
            I size output before allocating. The long text fixture formerly
            emitted an unterminated C string; it now executes exact comparisons.
          - [x] I remove silent string-interpolation limits (64 parts and a
            4,096-byte literal buffer) and bounded assertion-source truncation.
            I require long/interpolated/escaped string and diagnostic regressions.
            Shared lexing already lowers f-strings; native-only interpolation of
            ordinary strings is incorrect and redundant. The lexer's own fixed
            64-part arrays lack bounds checks. I replace their storage, remove
            duplicate native interpolation and preserve literal braces.
            Native and VM execute an 80-interpolation fixture with 5,000-byte
            literal segments. Lexer checks cover selected counts from 0 to 256 around growth
            boundaries. Long escaped assertion diagnostics retain their complete
            tested source spelling. Parser, typechecker, interpreter and
            transpiler gates pass; the standalone lexer also passes ASan/UBSan
            with leak detection disabled.
            MAC `task_041ab4cea1da3fea84d72483cb735c49`.
        - [x] I repair the tail-call fixture's non-language call syntax and
          discarded returns, add bounded shadows, and retain a million-step
          executable test with defined integer arithmetic and asserted results.
          Both VM strategies complete the million-step check; bytecode and
          ordinary native compilation pass the bounded shadows. Native execution
          without TCO crashes, and native --tco fails all six shadows.
        - [ ] I repair native TCO parameter binding, simultaneous argument
          evaluation, exit semantics and typed results. I retain the corrected
          million-step fixture and require successful native execution.
          MAC `task_ab36fda5e6846c45beaf42b3ba819c13`.
          - [x] I repair scalar self-tail returns using collision-free state
            names, argument temporaries, ordinary function returns and an
            in-place body update visible to the interpreter. Seven executable
            `make test-native-tco` cases pass: swaps, early returns, string/float
            results, generated-name collisions, void fallthrough and one million
            optimized calls. The gate is part of `test-opt-passes`; its C unit
            tests also pass. Unoptimized shallow programs remain the comparison.
          - [ ] I extend binding-aware lowering and acceptance to aggregate
            parameters, closures, loops and parameter-shadowing bindings. My
            preflight currently leaves these functions unchanged. Scalar test
            success is not a complete proper-tail-call guarantee.
      - [x] I repair verifier-corpus coverage. Every root `tests/*.nano` source
        must compile and pass `--verify-only`; failures, signals, missing
        artifacts and empty coverage fail the gate. Compilation and verification
        have 60-second process deadlines, and failure logs are retained.
        Synthetic tests exercise failures without relying on diagnostic text.
        The real gate reports 175 selected, 172 verified, three failures and
        zero skipped. Async/effects failures remain in the strict-corpus task;
        fixing coverage does not make the release green.
        MAC `task_b94386786f35a8a28af6dd2ad7bbb42d`.
        - [x] I write subprocess output directly to per-case log files so
          inherited pipes cannot hold the timeout runner open after process
          exit. I test a detached descendant and preserve dispatch comparisons.
          Four verifier-gate tests and three dispatch-gate tests pass. I do
          not claim descendant isolation or an output-storage quota.
        - [x] I add `nano_vm --verify-only`: load and verify, then return
          without loading FFI libraries or executing the module. I test silent
          success, non-execution, unresolved externs, invalid files and option
          conflicts before switching the corpus gate to this command.
          Three CLI regression tests and the verifier unit gate pass. This
          command verifies bytecode structure; it does not resolve foreign
          symbols or establish that runtime execution will succeed. The old
          corpus script remains unfinished above, including its silent skips.
      - [x] I isolate MAC module shadows from live hub writes and replace the
        unresolved capture/status extern pair with the existing process runner.
        Incoming tests hid missing interpreter support as offline success, and
        their VM dependency shadows created four real test tasks. I cancelled
        those verified fixtures, preserving audit records. I require real
        exactly-once counts and fake-CLI assertions without hub side effects.
        `make test-mac-command-boundary` passes native and bytecode compilation,
        dependency/root shadows, execution, both legacy fixtures, stdout/stderr
        and exit-status checks. JSON task IDs use the `id` field rather than
        task-like description text. The merged VM gate passes 272,379 checks;
        the FFI gate passes 25 tests. Full release acceptance remains open.
      - [ ] I integrate the existing AOT runtime and self-hosted NanoISA work
        from `origin/feat/4.6-frontend-contract` before duplicating that backend.
        Its history contains `895c6eda`, `8e781874`, `4d407c9a` and `8299138f`,
        which are absent from this branch. I preserve the newer verifier,
        module identity, callback and aggregate fixes through conflict review;
        rerun executable AOT/bootstrap gates rather than trusting old task text.
        - [x] I restore the compiler-sized local limit from the existing AOT
          branch: 1024 rather than 256. My codegen suite tests 1024 locals and
          rejection at 1025; all 68 tests pass. My complete `nanoc_v06.nano`
          bytecode build now passes with dependency shadows enabled.
        - [x] I identify failed bytecode shadows without per-opcode tracing.
          Assertion failures now report their stack. All 39 bytecode-shadow
          tests pass, including stack output and failed-artifact nonpublication.
          I retain the ten-second deadline. Stack locations still use the root
          source path; this is not original dependency-file provenance.
        - [x] I repair the compiler's `scan_block_for_lists` and
          `substitute_union_field_type` shadows on NanoVM. Empty array arguments
          retain declared parameter element types; empty push receivers infer
          their representation from the first value, including registered
          builtins. Native and VM fixtures test strings, records, floats and
          booleans, nested pushes, shadows and execution. The full compiler's
          default dependency-shadow build now publishes its bytecode.
        - [x] I preserve array result element types through direct call indexing.
          Root and imported function registration retain the declared element
          type; native accessors recover the record name when needed. My native
          and VM fixture checks direct bool/string/float/record call indexing.
          A qualified imported Boolean-array call checks module registration,
          empty arguments and call-result indexing on both backends too.
          Typechecker tests and all 68 NanoVirt tests pass. This does not claim
          general indirect-call or self-hosted inference parity.
          MAC `task_cf8bfc10244bc0fc3488c109278d4cb8` retains the evidence.
        - [ ] I carry that compiler bytecode through native AOT translation.
          The current executable gate rejects its imports with `imports require
          a host ABI; nvm2c refuses CALL_EXTERN`. I integrate the existing AOT
          branch's host support with checked signatures, then compile and run
          the generated compiler. I do not substitute an embedded VM wrapper.
          - [x] I integrate the existing process-argument and environment host
            adapters behind exact builtin import signatures. I reject foreign
            namespaces, co-process/artifact import kinds and type mismatches;
            test native execution, argument bounds and missing environment
            values, including spaced arguments and `--help`. All 608 AOT checks
            pass normally and with generated C and
            harness ASan/UBSan instrumentation; prebuilt objects are not all
            instrumented. Returned host strings remain allocated until exit.
          - [x] I bound standard filesystem directory traversal before relying
            on it for native bootstrap. `walkdir_recursive` follows directory
            symlinks through `stat`, has no visited-identity set and builds paths
            in an unchecked 2048-byte buffer. I define and test cycle handling,
            long paths and failure behavior instead of importing these hazards
            into a second implementation. MAC `task_aa93c6440aa0c87057a3306140bea1fb`.
            The C-seed's emitted `nl_os_walkdir_rec` repeats the cycle/depth
            hazard. I share the corrected walk between the module and native
            emitter and test both entry points.
            The native compile-and-shadow regression also exposes the same
            recursive walk in the C-seed interpreter; it must share the repair.
            All three now share an iterative queue, opened-directory identity
            checks and dynamic path storage. Six filesystem tests pass with
            one host-path-limit skip, including native/VM shadows and execution;
            the five C fixtures also pass ASan/UBSan with the same skip. Eight
            bootstrap-dependency tests pass. Leak checking is not established.
          - [x] I fail safely at native array allocation boundaries before
            changing element ownership. Failed growth currently returns to a
            push that writes beyond the unchanged capacity; capacity products
            and alignment rounding are unchecked. Reserve/clone and first
            struct allocation need the same checked contract. I inject failures
            and test overflow boundaries, not only successful allocation.
            Flat struct cloning also lost its element width and copied into
            delayed, unallocated storage; I preserve width and test independent
            cloned contents. I reject widths beyond the current byte-sized ABI
            before truncation, including builds with assertions disabled.
            I also snapshot an array's borrowed record before a self-append can
            reallocate its storage, and fail explicitly if string copying fails.
            Nine failure cases and positive construction/growth/clone/alias
            cases pass with assertions enabled and disabled, both normally and
            under ASan/UBSan. All 26 existing array tests pass. The filesystem
            integration suite passes with its one host-path-limit skip.
            MAC `task_4f3df48b3bb966a5ec5b7239445d09e4`.
          - [ ] I widen and version native array element-width metadata for
            general record arrays. Its current one-byte width cannot represent
            records above 255 bytes; CUDA, OpenCL and SDL helper code duplicate
            that native layout. I update those boundaries and their ABI tests
            together, not just the core typedef.
            MAC `task_83bd7cd20fc916a0140dd06c19a3e550`.
            - [x] I first remove duplicated GPU/SDL layouts and validate host
              transfer lengths against the declared integer-array storage.
              GPU copies currently accept unchecked byte counts; SDL texture
              upload multiplies unchecked dimensions and does not verify array
              type or texture extent. I test these boundaries with fake drivers,
              keeping the layout unchanged until ABI versioning is implemented.
              CUDA-only, unified CUDA/OpenCL and SDL upload fixtures pass
              normally and with ASan/UBSan. I reject invalid metadata, lengths,
              pixel formats and texture extents before driver transfers. The
              full SDL helper source also passes syntax checking with installed
              SDK headers. These tests do not establish real GPU or display
              behavior, device-pointer provenance or mixed-version ABI safety.
            - [x] I version array-bearing artifact exports per C function and
              check the declaration against the image that supplies that
              function. Unmarked artifacts mean legacy version 1, not whichever
              ABI the host happens to use. I test matching/mismatching versions,
              missing declarations under version 2, and a declaration supplied
              by the wrong dependency before changing the layout.
              Loader fixtures pass normally and under ASan/UBSan; all 26 VM
              FFI tests pass, including rejection before an incompatible
              array-returning function can execute. I keep the current layout
              at version 1. These declarations are trusted C metadata, not
              memory-safety proofs.
            - [ ] I enforce the same version boundary in native C compilation
              and cover array-bearing exports across modules before switching
              to the wider version-2 layout. The VM artifact check alone does
              not protect native-linked calls or legacy logical imports.
              Native fixtures also exposed missing function-type collection
              inside unsafe blocks; I fix and test that prerequisite so taking
              an array-bearing extern as a function value remains covered.
              - [x] I guard C-seed native calls and unqualified extern function
                values using exported declarations and defining-image checks.
                Fifteen shared/static executable cases pass on Darwin, covering
                matching, unmarked and mismatched versions, array parameters
                and results, qualified calls and function values. Native loader
                fixtures also reject wrong-image and missing-v2 declarations.
                I collect function types inside unsafe blocks before emission.
                Native executable and loader fixtures pass under ASan/UBSan;
                all 26 VM FFI tests and eight bootstrap dependency tests pass.
                Filesystem integration passes with one host-path-limit skip.
              - [x] I check legacy VM logical array imports after resolving
                their actual function address. I look up the declaration in
                the process namespace without rebinding the call and require
                the same defining image. Matching/unmarked calls succeed;
                incompatible calls fail before entry and remain failed in the
                descriptor cache. All 26 VM FFI tests pass; loader tests cover
                logical matching, mismatched, missing-v2 and wrong-image
                declarations normally and under ASan/UBSan.
              - [x] I guard self-hosted native array calls and unqualified
                extern function values, preserving local function shadowing.
                Sixteen shared/static executable cases pass with a private
                Stage 1 compiler normally and under ASan/UBSan; nine module
                binding regressions pass. The compiler builds a private Stage 2
                with default dependency shadows; Stage 2 passes the same
                sixteen executable cases. The C-seed fifteen-case suite and
                eight bootstrap dependency tests also pass. I include Linux loader/export
                flags and test their selection in a shadow; actual Linux
                integration still requires its platform gate.
                The complete `src_nano/nanoc_v06.nano` driver also builds to a
                575,024-byte NanoISA artifact with default dependency shadows.
                That establishes bytecode compilation, not the unfinished
                native AOT compiler-bootstrap acceptance gate.
              - [ ] I enumerate and validate module array exports before
                widening the layout. Declarations remain trusted metadata;
                hidden/stripped markers count as missing.
                - [x] I validate UI-widget array metadata, counts and scroll
                  indices before SDL calls; I test invalid inputs and valid
                  prefixes before declaring all three layout markers.
                - [ ] I audit UI geometry narrowing and signed arithmetic,
                  including non-finite scale and mouse-coordinate conversions.
                  - [x] I normalize invalid scales, saturate scaled mouse
                    coordinates before integer conversion, and widen rectangle
                    endpoint comparisons; I test edge values under sanitizers.
                  - [x] I bound slider/progress geometry before narrowing,
                    normalize non-finite fractions, and reject zero-width
                    interaction before division; I test drawing and click paths.
                  - [x] I validate spinner geometry and ranges, clamp initial
                    values, and check centered text coordinates before SDL;
                    I test integer-limit clicks and extreme surface dimensions.
                  - [x] I bound panel/label geometry and clamp color channels
                    before border arithmetic; I test extreme channels and
                    label-surface cleanup when placement is rejected.
                  - [x] I validate button geometry and share checked centered
                    text placement with the spinner; I test rejected geometry,
                    click behavior, extreme surfaces and cleanup.
                  - [x] I validate checkbox/radio coordinates, including the
                    radio border's extra pixel, and check adjacent label
                    placement; I test click/toggle behavior and cleanup.
                  - [x] I format time displays over the full signed int64
                    range and reuse checked label placement; I test signed
                    limits, long durations and rejected placement.
                  - [x] I repair image-button brightness conversion, which
                    casts values above 255 before clamping, and bound geometry.
                  - [x] I check tooltip measurement failures, padded dimensions
                    and cursor offsets before drawing; I test extreme values
                    and rejected surface cleanup.
                  - [x] I bound text-input buffer reads, geometry, text and
                    cursor placement, measurement failures and blink counting.
                  - [x] I validate list/file-selector geometry and row text
                    placement, preserving the caller's clip rectangle; I test
                    coordinate limits, selection, and rejected text cleanup.
                  - [x] I validate closed and expanded dropdown geometry and
                    text placement; I test height multiplication, open/close
                    signals, selection and rejected surface cleanup.
                  - [x] I bound syntax-highlighted code-display geometry and
                    scrolling, check font measurements, preserve full tokens
                    beyond 255 bytes, and restore clipping; I test failures.
                  - [ ] I audit ANSI/code-editor indexing and rendering,
                    including multiline tokens and visual-line scrolling.
                    - [x] I bound ANSI geometry and escape parsing, check
                      measurements and surface placement, saturate offscreen
                      pen advances, and restore clipping; I test malformed SGR.
                    - [x] I bound editor geometry, line/cursor arithmetic and
                      measurements, preserve full tokens and cursor prefixes,
                      and restore clipping on allocation failure.
                  - [ ] I reconcile the text-input API's promised editing and
                    Enter behavior with its current render-only implementation.
                - [x] I make process pipe-spawn result allocation precede pipe
                  creation/fork, close partial pipe setup, check nonblocking
                  configuration, and test failures before declaring its ABI.
                  Normal and ASan/UBSan tests cover constructor/field allocation,
                  either pipe failure, descriptor setup and fork failure, real
                  stdout/stderr capture and callers with closed standard output
                  descriptors. Read ends are nonblocking and close-on-exec.
                - [x] I validate preference playlist saves before opening the
                  output and test rejection without truncating existing files,
                  valid prefix writes, and stream failures.
                - [x] I replace preferences' manually allocated playlist arrays
                  with canonical runtime allocation, preserve long lines, and
                  clean up partial loads on failure; I reject truncated or null
                  preference path inputs before returning a path.
                - [x] I replace process_run's truncated shell/redirection
                  command and reopened temporary paths with descriptor-owned
                  capture, preserving full commands and checking read failures.
                  MAC `task_ed30bee0415e4c1a9959968c9a078d16`.
                  The full command goes to /bin/sh unchanged; anonymous capture
                  descriptors are retained through child exit and checked reads.
                  Normal and ASan/UBSan tests cover compound/6,000-byte commands,
                  72 KB output, null/NUL handling, allocation/setup/read failure,
                  and closed standard descriptors. I declare its array ABI.
                  This synchronous API has no timeout/output quota and cannot
                  roll back command side effects on later capture failure.
                - [x] I declare PEG capture arrays and make capture allocation
                  failure return null without partial results or process exit.
                  I test table growth, element copies, empty/no-match results
                  and snapshot lifetime with injected allocation failures.
                  Normal and ASan/UBSan fixtures pass, as does the existing
                  native PEG regression with default shadows. Successful
                  copied-string ownership remains a separate runtime task.
                - [x] I validate GLEW array upload types, widths, byte bounds,
                  conversion ranges and allocation failure before touching GL.
                  I separate the adapter for fake-driver tests, preserve valid
                  empty uploads, and declare both array ABI markers.
                  Fake-driver normal and ASan/UBSan tests pass. The manifest
                  sources link and export both markers without loading GL.
                  Negative/oversized integer indices and finite values beyond
                  float32 range now leave the buffer unchanged instead of
                  silently narrowing. This does not test a real GL context.
                - [x] I declare the filesystem listing array ABI and replace
                  truncated 1,024-byte joined paths with descriptor-relative
                  stat checks. I test sorted filtering, symlinks, empty/null
                  paths and long directory names without changing symlink
                  following semantics. Copied-string ownership stays separate.
                  The native fixture checks all three layout markers and
                  entries whose joined path exceeds 1,024 bytes; normal and
                  ASan/UBSan runs pass.
                - [x] I repair the filesystem module's remaining static
                  2,048-byte join/parent path truncation and define null-input
                  behavior for its scalar path queries.
                  Normal and ASan/UBSan tests check exact buffer limits,
                  overlong rejection, previous-result aliasing and null queries.
                  Returned path buffers remain borrowed and non-thread-safe.
                  MAC `task_c4177b06a5c741ad557bb32cd2c8b92e`.
                - [x] I declare and test collections key/value/set snapshots
                  and JSON object-key arrays against the canonical layout,
                  including empty and post-source-destruction results. Their
                  copied-string ownership remains a separate runtime task;
                  a layout marker does not certify leak-free ownership.
                  All four version markers and snapshot contents pass normal
                  and ASan/UBSan tests. Test cleanup explicitly frees the known
                  copied elements; runtime ownership is still tracked by MAC
                  `task_93bb44374587a757753418fc28c2095d`.
              - [x] I repair SDL_image array exports before declaring their ABI:
                batch loading accepts raw `char**` and returns raw `int64_t*`,
                while the language declares `DynArray` arguments/results.
                Batch destruction also frees a raw buffer rather than accepting
                the declared array representation. I check supported-format
                results, count bounds, partial failure and repeated cleanup
                with fake SDL entry points and sanitizers before certification.
                MAC `task_5f2879308468ad24921d62eedea9f3e9`.
                - [x] I keep the callback fixture warning-clean under the
                  sanitizer compiler's SDK by bounding its formatted string.
                - [x] I diagnose the full FFI sanitizer suite's Darwin crash
                  in ASan global registration while loading ffi_artifact_first
                  after the SDL cleanup case. The focused SDL case passes;
                  that does not establish a passing full sanitizer suite.
                  MAC `task_a36fd588dd05d06fadc36470a23cdc1b`.
                  A standalone dlopen/dlclose/reload reproducer fails with
                  duplicate ASan global registration under both installed
                  Apple Clang 21 builds, without linking my VM. Homebrew Clang
                  23.1.1 passes the reproducer and all 27 FFI tests with
                  ASan/UBSan. I keep image unload and sanitizer checks enabled;
                  the suite now runs the reload probe before VM dispatch.
                  This checks instrumented fixtures and driver, not a clean
                  sanitizer rebuild of every VM object.
                - [x] I rebuild the complete FFI dependency closure in fresh
                  ASan/UBSan object storage. I remove the redundant block-scope
                  extern declaration of the file-local tuple registry that
                  triggers Clang 23's unused-global diagnostic, and verify
                  native tuple emission as well as the runtime tests.
                  I also remove an unread shadow-test pointer and make the
                  retained-image fixture use the selected object directory.
                  The fresh Homebrew Clang 23.1.1 ASan/UBSan build passes 27 FFI
                  cases, 35 protocol cases, eight protocol fuzz checks and SDL
                  cleanup integration, with UBSan halting on errors. Native
                  tuple and tuple-parameter programs pass with default shadows,
                  and the transpiler unit gate passes. Darwin leak detection is
                  disabled; these runs do not establish leak freedom or Linux
                  behavior.
                - [x] I replace the three raw-buffer implementations with a
                  separately testable canonical-array boundary and ABI markers.
                  Native fake-SDL tests cover count/type/width bounds, allocation
                  failure before acquisition, null and failed paths, prefix
                  cleanup with duplicates, repeated cleanup and format results.
                  I leave array storage under runtime ownership. Cross-array
                  copied handles are not ownership-safe aliases.
                  Normal and ASan/UBSan runs pass. Both production sources
                  pass SDK syntax checks; the manifest-selected sources link
                  and export all three ABI markers without loading SDL startup.
                - [x] I preserve native array mutations across VM FFI calls.
                  My previous marshalling copied VM arrays to `DynArray` without
                  copying mutations back, so clearing texture handles during
                  cleanup did not prevent repeated VM-side destruction. I test
                  copy-back, aliasing and temporary-array cleanup before
                  claiming repeated cleanup across both backends.
                  MAC `task_2d34e62c8b2e0e086045c7d56f6eb66f`.
                  My production SDL_image cleanup adapter now passes direct VM,
                  mailbox and 2,000-element pipe tests with fake SDL destruction:
                  prefix aliases are cleared and repeated calls destroy exactly
                  two distinct handles. Focused normal and ASan/UBSan runs pass;
                  the full normal FFI suite passes 27 tests. Prebuilt VM objects
                  are not wholly instrumented by the focused sanitizer run.
                  - [x] I complete the in-process call frame for scalar/string
                    arrays, including shared argument/return identity, snapshot
                    ownership, UTF-8 and metadata checks, and all-or-nothing VM
                    publication after fallible conversion. Native side effects
                    cannot be rolled back after a failed copy-back.
                    My frame tests pass normally and under ASan/UBSan, including
                    invalid UTF-8, embedded NUL, malformed metadata, allocation
                    failures and 1,000 repeated alias calls. All 27 VM FFI tests
                    pass, including typed array/float calls and rejection of an
                    invalid result before mutation publication. Wrapper gates
                    pass with the new runtime object. This is not direct SDL
                    resource-lifetime integration or co-process parity.
                  - [x] I extend the co-process request/reply protocol to carry
                    array mutations and alias identities. The former value-only
                    transport lost both; call envelopes now preserve them in
                    tested mailbox and pipe paths.
                    - [x] I add a versioned call-value envelope with bounded
                      backward array references and atomic reply application;
                      I test truncated envelopes and changed alias topology.
                      My protocol and fuzz suites pass. I also carry u8 values
                      and reject mismatched serialized array element tags.
                      This codec does not establish transport integration.
                    - [x] I connect that envelope to mailbox and pipe dispatch,
                      preserving ordered array mutations across batched calls.
                      - [x] My mailbox uses call envelopes and resets batch mode
                        for single calls. Array-bearing batches cross in order;
                        scalar batches remain packed. Forked tests check shared
                        argument/result identity, three dependent mutations,
                        repeated clearing and invalid-result rejection.
                      - [x] I migrate pipe requests and replies, update the wire
                        version, and test large isolated payloads without
                        silently falling back to in-process execution.
                        - [x] My version-2 pipe codec uses call envelopes in the
                          parent and standalone worker. The shared handler
                          validates requests and bounds reply allocation at
                          16 MiB. A forked pipe fixture checks 2,000-element
                          aliased arrays, repeated mutations and reply identity.
                        - [x] I connect a large-payload pipe channel to the
                          default mailbox worker without restarting its native
                          state, and bound the full pipe exchange by a deadline.
                          My default worker polls both channels. Tests alternate
                          2,000-element pipe cleanup calls with mailbox counter
                          queries and verify one PID, persistent native counts,
                          and repeated cleanup without double destruction.
                          - [x] My parent pipe exchange uses one monotonic
                            deadline for request/header/body I/O, temporarily
                            enables nonblocking I/O and contains its SIGPIPE.
                            Stalled-write, partial-reply and dead-peer tests
                            pass. Teardown closes instead of writing shutdown
                            into a full pipe, then kills an unresponsive worker
                            after the grace period rather than trusting SIGTERM.
                        - [x] I spill oversized mailbox replies through the
                          worker's pipe without executing the foreign call a
                          second time. A native growth fixture expands one
                          element to 2,000, preserving result identity and PID;
                          its counter proves two requests execute twice, not
                          four times. The receive-only path shares the original
                          deadline. Variable-size batch results use this path.
                          The 16 MiB cap and allocation failures still fail
                          closed after native side effects have already occurred.
            - [ ] I support qualified extern function values consistently with
              qualified calls. `let f: fn() -> array<int> = foreign.probe`
              currently fails with a struct-field diagnostic before native
              publication; this is not working qualified function-value syntax.
              MAC `task_3c147355ed9fcbd45630c41b80db8a85`.
            - [ ] I reconcile C-seed `--target c` with native C generation:
              its separate `c_backend` currently emits raw integer pointers and
              unresolved array helpers for the native array ABI fixture.
              I must implement or explicitly reject unsupported foreign array
              lowering before publishing that source as usable C.
              MAC `task_618cc665c2cfde8725221bd9b3306ecc`.
          - [ ] I define ownership for native array string elements, including
            filesystem walk results. `dyn_array_push_string_copy` allocates
            copies, while the native GC array destructor frees only the backing
            array. I test release and escaped-element lifetimes before changing
            that ownership; freeing borrowed strings in a walker finalizer would
            not establish a safe general array contract.
            MAC `task_93bb44374587a757753418fc28c2095d`.
          - [ ] I integrate the compiler's artifact-backed standard filesystem
            imports without silently rebinding foreign exports by symbol name.
            The current next failure is import 1, `fs_walkdir`, bound to a
            retained `libstd.dylib` artifact. The old branch's name-only host
            replacement does not preserve this identity or the native array
            ABI. I retain checked artifact binding and adapt its typed values;
            remaining builtin file/process adapters and compiler AOT gates
            stay in scope.
        - [x] I integrate native prefix/suffix string operations from that
          branch without importing its heuristic aggregate classifier. I test
          empty, longer, equal, matching and nonmatching strings through emitted
          C and retain the current control-flow and aggregate rejection tests.
          I also integrate fixed-width byte escaping for UTF-8/control literals,
          including a following digit that must not extend a C escape.
          `make test-nvm2c` passes 485 checks, including emitted C compilation
          and execution for 22 prefix/suffix cases. The imported implementation
          comes from `8299138f`; its larger classifier is not merged by this step.
        - [x] I reconcile `ARR_SET` for integer, string and record arrays with
          the current typed classifier, preserving alias-visible mutation,
          bounds checks and rejection of incompatible element representations.
          Record widths need runtime guards because classifier field-kind facts
          do not encode width. Write-only record arrays also exposed an unused
          helper warning under `-Werror`; inline did not suppress it, so I emit
          only the record-array helpers selected by the module's operations.
          `make test-nvm2c` passes 529 checks, including alias-visible writes,
          negative/length/maximum-int indices, incompatible scalar/record fields
          and runtime record-width rejection. My first sanitizer invocation
          passed `CC` only through the environment, which `Makefile.gnu`
          overrides. That run did not establish instrumentation coverage.
          I rerun with an explicit make command-line override below.
        - [x] I integrate self-tail restart without C recursion. I preserve
          simultaneous argument transfer, typed call checks and fresh local
          state; deep scalar/record recursion must run at `-O0` without relying
          on a C compiler's tail-call optimization. Mutual tail calls remain
          separate work until their native stack behavior is checked.
          `make test-nvm2c` passes 547 checks. Deep scalar/string/record calls
          execute 100,000 or more iterations with `-O0` and
          `-fno-optimize-sibling-calls`; malformed arguments/extra stack values
          are rejected, and unreachable self-tail code still compiles cleanly.
        - [x] I verify sanitizer compiler selection for AOT tests explicitly:
          `make CC='cc -fsanitize=address,undefined' test-nvm2c`, with Darwin
          leak detection disabled. Compiler command lines, not log filenames,
          establish instrumentation; prebuilt linked objects remain outside it.
          The explicit override and new `make test-nvm2c-sanitizers` target both
          pass 547 checks with compiler flags present. This supersedes the
          earlier environment-only sanitizer claim for the array-mutation work.
        A read-only merge preview finds 16 conflicting paths across 56 changed
        files. I first reconcile main's three conflicts: preserve checked FFI
        failure reporting and the central GC child-slot walk while incorporating
        the exactly-once MAC command behavior and its regression tests.
- [x] **Release-gate module shadows.** I diagnose and repair the NanoISA
      facade assembly fixtures exposed by default dependency-shadow execution,
      replacing obsolete function headers with explicit `void 0` results.
      The module gate passes; the full release gate belongs to the release
      integration item above.
- [x] **Release-gate emitter shadow.** I initialize the declared local in
      the `nisa_emit_call` fixture before emitting its use, then rerun the
      source-emitter gate (86 checks passed). The complete release gate
      remains part of the integration item.
- [x] **Release-gate byte I/O fixture.** I import the filesystem wrappers
      used by the bytes-conversion shadow instead of assuming undeclared
      `file_write` and `file_remove` functions. Compile and execution pass.
- [x] **Release-gate mutable map shadows.** I test empty maps before mutation
      and shared-handle state after mutation; the native hashmap wrappers
      mutate their opaque handle and do not create persistent snapshots.
- [x] **Release-gate dependency shadows.** I correct the JSON array-length
      spelling and vector normalization's exact floating-point comparison,
      then diagnose nested-array evaluation exposed by the property library.
      I preserve nested literal tags in bytecode lowering as well as inner
      array identity in the interpreter, with a shared executable regression.
- [x] **Release-gate coverage timing.** I provide the generated runtime's
      epoch-millisecond helper to interpreter shadows and test its result.
      Property-counterexample shadows must expect failures, not success,
      and use guaranteed counterexamples where the test requires one.
- [x] **Release-gate OPL shadows.** I escape a quoted validation fixture
      and require the newline skipper to advance past its newline token.
- [x] **Bytecode slice convention.** I lower the third `array_slice`
      argument as a length, evaluate arguments once, and clamp before adding.
      I avoid signed overflow in native and interpreter slicing too. The
      nonzero-start and maximum-int-length regressions pass in both backends.
      MAC `task_efa11df9058a199510c563782734c31a`.
- [x] **Release-gate module identity rejection.** I reject distinct imported
      files with the same declared/fallback introspection identity in the
      C loader, matching the self-hosted contract before publication.
- [x] **Release-gate self-hosted test routing.** I build and select my actual
      stage-one driver for self-hosted CLI tests, instead of testing my C seed
      under a self-hosted label.
- [x] **Nested unary-minus C emission.** I parenthesize nested negation so
      `-(-7)` does not become the C decrement token `--7`.
- [x] **Release-gate VM example coverage.** I repair the compilation and
      dependency-shadow failures reported by `make test-vm-examples` and
      remove eight stale exclusions that now compile to bytecode. I retain
      the default dependency-shadow contract rather than hiding failures.
      MAC `task_7ee12d8737363c126a040fde905a7114`.
      - [x] I traverse match scrutinees and arm-local immutable bindings in
        both closed-purity walkers, plus guards in the C AST that represents
        them. I treat evaluating a resolved function declaration as
        effect-free, while keeping invocation through an unqualified
        function-typed value unknown until function types carry a verified
        closed-purity capability. I keep those higher-order helpers ordinary
        rather than weakening the purity contract. My three-stage component
        build and full bootstrap pass, the shared two-frontend purity contract
        passes five methods, and `make test-vm-examples` lowers all 245
        eligible examples while preserving all four verified exclusions.
      I verify integer/wildcard/guarded match lowering, block-arm values,
      named union/enum signatures, typed array allocation, inferred record
      names, and foreign runtime boundaries with focused regressions before
      rerunning the full example gate.
      I run MAC examples against an explicit offline CLI fixture: their
      shadows must not create tasks in a developer's live ledger.
- [x] **Single logical foreign object per link.** I select one immutable
      generation per physical module directory in each native invocation.
      A deterministic regression uses two NanoLang interfaces to one C object
      with uncached compiler identity. It reproduced duplicate symbols before
      the repair and passes two complete compile/run invocations afterward.
      MAC `task_9b9359bd64dce70ea919f0861ac9d5ba`.
- [ ] **Release-gate native callbacks.** I replace raw bytecode-index-to-C-
      pointer conversion with an ABI-aware bridge that owns callback lifetime,
      asynchronous quiescence, captures/globals, and VM reentrancy. My retained
      VM bridge and Apple dispatch adapters now pass their focused gates.
      Unsupported callback imports still fail closed; SDL_mixer contracts and
      complete release acceptance remain unfinished.
      MAC `task_20f0d57878f248cd8573e6841825152a`.
      The user requires lifetime-safe callbacks before 5.0; native-only
      dispatch is not an alternative release condition.
      - [x] I define a versioned, typed, retained callback handle and test
        owner-thread execution, concurrent submission, reentrancy, final
        release, cancellation, and handles that outlive runtime shutdown.
        `make test-callback-runtime` passes 40,000 cross-thread invocations,
        allocation-failure checks, and deterministic queued cancellation.
        Both binaries pass ASan/UBSan and TSan on Darwin arm64. This is the
        handle runtime, not completed VM or native-adapter integration.
      - [x] I carry callback signatures and adapter contracts through the
        compiler, serialized imports, loader, and verifier; absent contracts
        fail closed rather than inferring behavior from symbol names.
        - [x] I parse strict `callback_adapters` manifest entries, retain
          their metadata with the selected native generation, and bind
          scalar callback signatures from the loaded declaration for both
          production and shadow modules. I reject missing contracts,
          duplicate/unknown fields, unsupported signatures, and NUL-bearing
          manifest strings before native symbol selection. I test manifest
          mutation after capture and document the supported schema.
          NanoVirt passes 65 checks, bytecode shadows 39 tests, and native
          cache/link-response regressions 83 tests. Manifest-parser and JSON
          decoder metadata tests pass ASan/UBSan on Darwin with leak detection
          disabled. Runtime scheduler integration remains unfinished.
        - [x] I encode explicit retained-handle adapter contracts in a
          feature-gated v2 section, referencing existing signatures. I verify
          indices, scalar shapes, parameter coverage, duplicates, policy,
          truncation, and rejection of lossy legacy serialization.
          Canonical assembly retains callback contracts, import kinds,
          and declared function parameters instead of dropping ABI facts.
          `make test-nvm-callbacks` covers round trips and malformed inputs;
          canonical tooling passes 117 checks, the v2 bridge 295, FFI 23,
          VM 272247, and AOT 375. Execution integration remains unchecked
          below and in the parent item.
        - [x] I preserve declared function parameter tags through the
          in-memory module and v2 round trip, including nested functions.
          Unknown legacy tags stay explicitly unknown. Function-table growth
          must fail transactionally rather than returning function zero.
        - [x] I assign nested parameters their own declaration provenance
          and check nested bodies against their own return type. A preceding
          float parameter named `value` must not change a nested integer
          parameter with that name. I test distinct nested return types too.
          A nested body cannot break out of the enclosing function's loop.
          I also preserve callable-local precedence in tail-position calls:
          `return (inner 4)` must invoke the closure with its captures, not
          tail-call a bare function index and lose its captured values.
          MAC `task_63005c07d487b97ee55756fb3373bef7`. NanoVirt passes 64
          checks, the v2 bridge 295, and the verifier 93. Typechecker tests
          include rejected nested return and loop-control violations.
      - [x] I connect owner-thread callback execution to suspended VM
        activations, preserve captures/globals, pump during foreign waits,
        and propagate errors without racing the heap or corrupting frames.
        - [x] I retain callable module identity across linked-module returns
          and indirect dispatch, preserving my 16-byte value layout. I test
          colliding function indices in two linked modules, returned direct
          functions and closures, and root callables passed into a dependency.
        - [x] I invoke a callable at a suspended owner-thread boundary and
          stop at its activation floor. I preserve the caller's frames,
          stack roots, module, instruction pointer, and halt state on normal
          return, assertion failure, allocation failure, and nested activation
          floors. Tests use one or two suspended caller frames.
          VM tests pass 272359 checks; NanoVirt passes 65, the frontend matrix
          16, FFI 23, and AOT 375. This is the activation mechanism, not the
          scheduler or native dispatch integration.
        - [x] I publish retained handles only after resolving the callable's
          owner and checking its recorded parameter/result signature against
          the import contract. I reject unknown signatures and mismatches.
          My VM bridge roots captures, executes through suspended activations,
          latches VM failures, and detaches roots before heap destruction.
          Tests cover 256 foreign-thread calls and a live producer crossing
          shutdown (128 executions followed by 128 cancellations), late
          handles, wrong-thread operations, and failed allocation/publication.
          VM tests pass 272379 checks; the bridge passes ASan/UBSan and TSan
          on Darwin. Automatic safe-point and native-wait pumping remain open.
        - [x] I marshal contracted native calls on the owner, run worker-policy
          calls outside the VM, pump callback requests during waits and at
          bounded instruction intervals, and propagate latched failures.
          I test synchronous callbacks, nested waits, asynchronous completion,
          policy-only waits, and explicit isolated-call rejection.
          FFI tests pass 24 cases, VM 272379 checks, and bytecode shadows 39
          tests including a real foreign-thread callback. The native-call
          bridge passes ASan/UBSan and TSan on Darwin with the documented
          harness settings. These are fixture gates, not dispatch acceptance.
        - [x] I keep a retained adapter's native image resident after VM and
          loader shutdown. A cancelled handle does not make delayed native
          code safe to unload. I test late native completion after teardown
          and document the process-lifetime image retention policy.
          I test failed allocation/open, recovery, and reuse across loader
          shutdown without adding duplicate image references. Those failure
          tests pass ASan/UBSan; the native scheduler and loader pass
          ASan/UBSan and TSan on Darwin.
        - [x] I keep loader registration transactional when copying a module
          name or path fails, and test both failures before retrying resolution.
          The failure fixture passes ASan/UBSan, and interpreter/VM FFI tests
          pass after the repair.
      - [ ] I implement retained dispatch adapters and an explicit COP
        transport policy, then test delayed work and shutdown in both paths.
        - [x] I add explicit retained adapters for queue/group callbacks and
          worker policies for blocking waits and queue destruction. I retain
          handles before asynchronous publication and release them on completion.
        - [x] I preserve transitive captures through intermediate closures:
          a callback nested inside a callback must capture an outer local even
          when only the innermost body reads it. I test this with nested native
          waits and shared array state, not an extra artificial outer read.
          I instantiate anonymous functions at their lexical expressions and
          preserve captures added to suspended parent tables. NanoVirt passes
          67 checks, including returned named/anonymous closure chains; parser
          and codegen cases pass ASan/UBSan. C-native parity is not established.
        - [x] I make queue destruction drain accepted delayed work as well as
          already-enqueued work, check constructor allocation failures, and test
          real captured state, barriers, group notification, timers and shutdown.
          `make test-dispatch-callbacks` passes real dependency shadows and
          isolated-call refusal. The 133-callback native lifecycle fixture
          passes ASan/UBSan and TSan; wrapper allocation failures pass ASan/UBSan.
      - [ ] I pass all six dispatch-dependent examples with dependency
        shadows enabled, sanitizer checks, and the complete clean release gate.
        - [x] I link my retained callback runtime and VM bridge into generated
          native wrappers. The full suite exposed missing objects in the
          wrapper's explicit link list. `make test-wrapper-gen` now passes
          all five linking cases and seven publication tests. The full suite
          remains a separate release gate.
        - [ ] I resolve newly exposed SDL_mixer callback declarations, including
          `Mix_SetPostMix`, through explicit adapter and threading contracts.
          The full example gate exposed six rejected audio/visualizer users;
          I do not exempt unused callback declarations to hide this boundary.
          - [x] I correct the post-mix signature to userdata, borrowed buffer
            and byte count; retain registrations across replacement, clear and
            audio close; and test audio-lock quiescence and late cancellation.
            I preserve the language's integer-zero null spelling for opaque
            foreign parameters while rejecting nonzero integer addresses.
            `make test-mixer-callbacks` passes a real dummy-audio VM capture
            fixture and isolated-call refusal; the lifecycle/failure fixture
            passes ASan/UBSan and TSan. The wider locking audit remains open.
          - [x] I reconcile `null_opaque()` inference and VM representation;
            the mixer fixture exposed its current unknown-type diagnostic.
            I accept its generic opaque type at a nominal opaque parameter and
            its integer-zero representation at the native boundary. VM equality
            now recognizes opaque null versus integer zero; the real callback
            trace reproduced the old inequality. Value/FFI tests cover both
            directions and reject nonzero integer addresses and floating nulls.
          - [ ] I audit remaining mixer/audio-lock operations for owner-thread
            deadlocks while a post-mix callback is registered. I preserve error
            reporting across worker calls and verify native-C callback parity.
            The real NanoLang fixture still fails C-seed callback marshalling
            and reaches the shadow deadline. Source-only C output also hoists
            the lambda without its captured `observed` binding. Native C adapter
            tests do not establish language closure parity; I must integrate
            the existing AOT foundations and keep that acceptance item open.
            - [x] I bind every mixer operation to a typed adapter, preserve
              source API names, convert SDK integer/void returns explicitly,
              and transport the last completed operation's error across workers.
              I exercise channel/music operations with an active post-mix hook,
              missing-file errors, error clearing and native adapter execution.
              Typed adapters and real VM/native-C fixtures pass. My first direct
              sanitizer probes failed before main because the sanitizer loader
              could not find Homebrew's SDL3 library. I now derive its directory
              from pkg-config, set only the test child's loader path, and run a
              headless SDL3-only preflight with the same sanitizer. Missing-library
              refusal is tested before SDL2 loads. `make test-mixer-sanitizers`
              passes the native adapter fixture with ASan/UBSan and TSan.
            - [x] I copy string arguments before callback-aware native calls
              and copy borrowed string results before their worker exits.
              I test worker-local result lifetime and owner-thread VM allocation
              before moving mixer file-loading and error boundaries to adapters.
              All 25 FFI cases pass, including TLS teardown, aliasing results,
              embedded-NUL refusal and cleanup after later argument failure.
              The focused FFI code and harness pass ASan/UBSan and TSan;
              real mixer callbacks and wrapper publication gates still pass.
        - [x] I give the boids graphical shadow a bounded real-frame runner.
          Its current shadow invokes the interactive event loop indefinitely
          and hits the ten-second deadline. I preserve interactive main and
          test rendering/dispatch cleanup through the bounded runner.
          Its two one-frame runs now complete inside the original deadline.
          All six original dispatch examples compile with dependency shadows;
          the complete example gate now passes all 243 eligible programs with
          six validated exclusions. Mixer locking and full release work remain open.
- [x] **Release-gate non-callback example shadows.** I correct nominal record
      lookup on call results, retain SDL_mixer artifact linkage, bound the
      particle rendering smoke test, and verify the entire example tree from
      the repository fixture root. I preserve actual assertions and dependency
      shadows. This continues the VM coverage task above.
- [x] **Release-gate cycle-collector live payloads.** AddressSanitizer caught
      a use-after-free of a module string constant during parser shadows.
      I balance trial-deleted edges exactly once, including record field names,
      and test dead cycles sharing payloads with live roots and record field
      names. The integrated VM suite passes 272,247 checks; parser shadows pass under
      AddressSanitizer and without tracing. The full release gate remains open.
      MAC `task_b53374269a1748c5a56a185a75fb6480`.
- [x] **Follow-up — C arrays of records.** I retain the nominal element
      name and use the dynamic-array representation when lowering record
      literals or record-returning calls inside array literals. Four native/VM
      acceptance methods cover direct and returned record values. PRs #415,
      #444 and #449 complete the repair. MAC
      `task_967a32569524e07e3c97742cf23234e9`.
- [ ] **Follow-up — MAC commands execute once.** My standard-library command
      wrapper captures stdout with one execution, then executes the same
      command again to obtain its status. I replace this with one execution
      and test a counted side effect, output, failure status, and offline use.
      Release tests use an offline fixture so shadows cannot mutate a live
      task ledger. MAC `task_5f807ded474a473ca5776018c32c636f`.

- [x] **5.0 / C-seed nested-array indexing.** I preserve recursive array
      type metadata while parsing and registering locals and parameters, then
      select the correct getter at each nested `at`. The C seed passes all
      16 native acceptance cases; the nested-array case also passes Stage 2.
      I allow explicit compiler selection and retain compilation diagnostics
      in the acceptance runner. Remaining Stage 2 failures are tracked below.
      MAC `task_b2c8c93fb14a4f089573edec76b99aab`.
- [x] **Sanitizer build isolation.** I make the AOT sanitizer target rebuild
      instrumented objects instead of reusing normal objects when only `CC`
      changes. I test warm-cache behavior before claiming translator coverage.
      MAC `task_f3df199b025042e0b1d83484cd104ed3`. My warm-cache run passes
      1,048 AOT and 952 shape checks with fresh instrumented objects, leaving
      normal translator artifacts unchanged. Evidence:
      `docs/evidence/aot-sanitizer-build-isolation.md`.
- [x] **Chronicle branch reconciliation.** I review main `1a5fed53` and
      worker `633acda1`, retain their identical README chronology update, and
      integrate both histories without replacing compiler work. I verify the
      merged diff and record ancestry under release task
      `task_cffdafd16e641ac417ccfddb962534b9`. Both heads are ancestors;
      source and tests are unchanged. Evidence:
      `docs/evidence/chronicle-branch-reconciliation.md`.
- [x] **AOT temporary directories.** I adapt `vm_mktemp_dir` with checked
      template allocation and exclusive creation. I test unique, independent
      paths and failure inside private roots before compiler acceptance.
      MAC `task_419c47bdc8fc42e4b52eb6af1a0e9a71`.
      My AOT suite passes 913 checks. Compiler acceptance clears imports and
      stops at function 18's `ARR_NEW` element kind. Evidence:
      `docs/evidence/aot-temporary-directories.md`.
- [ ] **Compiler AOT array shapes.** I identify the full compiler's array
      element shapes and support their construction and data flow without
      guessing a scalar representation. I verify complete compiler execution.
      MAC `task_419c47bdc8fc42e4b52eb6af1a0e9a71`.
      - [x] I construct explicitly tagged struct arrays and reject incompatible
        record-field representations on append instead of overwriting facts.
        My AOT suite passes 922 checks; evidence:
        `docs/evidence/aot-tagged-record-arrays.md`.
      - [ ] I complete compiler stack and array/aggregate shape support through
        full compiler acceptance. Function 20 (`parser_init_ast_lists`) now
        reaches unsupported array-valued fields in its 75-field aggregate.
        - [x] I allocate classifier stacks and branch snapshots from checked
          function bounds. My AOT suite passes 924 checks, including deep
          branch snapshots and incompatible heights. Compiler acceptance now
          reaches the 75-field aggregate limit; emitter storage and aggregate
          limits remain separate. Evidence:
          `docs/evidence/aot-dynamic-classifier-stack.md`.
        - [x] I allocate emitter operand storage and independent branch
          snapshots dynamically, including array-literal scratch storage and
          output. Both 75-value branch paths and 200-element integer/string
          literals execute correctly; 936 AOT checks pass. Evidence:
          `docs/evidence/aot-dynamic-emitter-stack.md`.
        - [ ] I remove fixed temporary-count limits and support the compiler's
          aggregate widths and array-valued fields without guessing their
          representation. I require full compiler translation and execution.
          - [x] I derive record field width from decoded module construction,
            use checked dynamic classifier/emitter facts, and emit matching
            value-copy record storage. I test wide mixed-field records through
            calls and branches before enabling array-valued fields. My normal
            and translator-instrumented AddressSanitizer suites each pass
            1,029 checks. Evidence: `docs/evidence/aot-module-record-width.md`.
          - [x] I preserve integer-array and string-array field representations
            through record packing, calls, locals and extraction, with tests
            for empty/nonempty arrays and copied records. Record-array fields
            still require nested shape facts before full compiler acceptance.
            My AOT suite passes 1,043 checks. Evidence:
            `docs/evidence/aot-scalar-array-fields.md`.
          - [ ] I carry recursive aggregate shape constraints through AOT
            calls and joins. MAC `task_9c850e94e5a74b6f8941622e2872af23`.
            - [x] I verify a dynamic, cycle-safe shape constraint graph with
              deep nesting, recursive records, shared children and conflicts.
              Its 952 checks pass normally and under ASan/UBSan. Evidence:
              `docs/evidence/aot-shape-constraints.md`.
            - [ ] I connect those facts to classification and C emission,
              preserve record-array element shapes, and pass full compiler
              acceptance. A standalone graph test does not complete this gate.
              - [x] I preserve nested record values using immutable snapshots
                whose lifetime outlasts returning functions, release owned
                snapshots after entry returns, and test calls, copies and
                repeated construction under sanitizers.
                My sanitizer suite passes 1,057 AOT and 965 graph checks.
                Compiler acceptance now reaches conflicting inferred field
                kinds rather than unsupported nested storage. Evidence:
                `docs/evidence/aot-nested-record-values.md`.
              - [ ] I resolve the compiler's conflicting nested-field
                inference with function/instruction diagnostics and focused
                regressions before relying on those shapes for reclamation.
                - [x] I implement `CAST_INT` stack effects and emission;
                  skipping it currently misclassifies a parsed tuple index as
                  a string. I test conversion boundaries and mixed call sites.
                  The sanitizer suite passes 1,060 AOT and 965 graph checks.
                  Compiler acceptance advances to function 170's array-kind
                  conflict. Evidence: `docs/evidence/aot-cast-int.md`.
                - [x] I preserve explicitly constructed record-array kinds
                  while nested element facts remain unknown. Function 170 at
                  offset 925 previously passed inferred `narr_t` to parameter 1
                  of function 141, which requires `nrarr_t`.
                  I keep unknown element facts unknown and stop `ARR_LEN`
                  from guessing integer elements. Normal and sanitizer suites
                  pass 1,065 AOT and 965 graph checks. Evidence:
                  `docs/evidence/aot-delayed-array-facts.md`.
                - [x] I propagate record-array result field facts through
                  calls and returns instead of supplying integer defaults.
                  Normal and tail-return regressions preserve mixed fields.
                  The sanitizer suite passes 1,070 AOT and 965 graph checks.
                  Evidence: `docs/evidence/aot-record-array-return-fields.md`.
                - [x] I resolve the next compiler field conflict without
                  weakening compatibility checks: function 264 offset 60,
                  `ARR_PUSH` field 0 has string versus integer facts.
                  I preserve explicit record-array literal tags and element
                  fields, including empty literals, with execution and
                  incompatible-element tests before compiler acceptance.
                  Unresolved local field vectors also remain unknown rather
                  than defaulting to integer. The normal suite passes 1,080
                  AOT and 965 graph checks. Evidence:
                  `docs/evidence/aot-record-array-literals.md`.
                - [ ] I diagnose and implement the compiler's hashmap opcode
                  requirements. Acceptance now stops at function 280 offset
                  487: ordinary and tagged strings conflict at parameter 0 of
                  function 282 (`type_from_string`).
                  - [x] I implement reusable emitted map storage with checked
                    growth, owned keys/values, missing-key results, replacement,
                    deletion and retained lookup values; test both integer and
                    string maps. Normal and sanitizer suites pass 1,106 AOT
                    and 965 graph checks. Evidence:
                    `docs/evidence/aot-map-storage.md`.
                  - [ ] I connect map storage to classification and emission,
                    preserving value kinds, missing-key tags and ownership
                    through locals, branches, calls and returns. Storage tests
                    alone do not implement `HM_NEW` or complete acceptance.
                    - [x] I carry map/key/value facts and emit construction,
                      mutation, presence, length and deletion through calls
                      and aliases. I retain `HM_GET` as an explicit gate until
                      missing-value tags and fetched-value ownership survive
                      compiler data flow.
                      My normal and sanitizer suites pass 1,114 AOT and 980 shape
                      checks. Compiler acceptance reaches `HM_GET` in function
                      268 at offset 32. Evidence:
                      `docs/evidence/aot-map-flow.md`.
                    - [x] I reclaim unreachable emitted maps and fetched
                      string copies before entry returns, preserving aliases
                      and returned values under
                      bounded-live-memory stress. MAC
                      `task_2f837947d5f24130b401ae433dd8d8c9`.
                      I trace live map identities and fetched string copies at
                      loop back-edges, reclaim only owners absent from locals
                      and the operand stack, and retain entry cleanup as the
                      final safety net. Repeated construction stays bounded.
                    - [ ] I preserve lookup results as tagged values through
                      stack transfers, locals, branches, calls and returns.
                      I test missing versus zero/empty values and retained
                      strings after replacement/deletion. Scalar consumers
                      check tags at consumption; lookup itself does not trap
                      or invent a default. I extend mixed-value joins and
                      return inference before claiming full integration.
                      - [x] I emit tagged lookups through locals, compatible
                        calls and same-representation branch joins, retaining
                        fetched strings after map mutation. I test missing
                        values, casts, tag inspection, scalar consumption and
                        truthiness. Normal and sanitizer suites pass 1,128
                        AOT checks and 990 shape checks. Evidence:
                        `docs/evidence/aot-tagged-map-lookups.md`.
                      - [ ] I preserve boolean versus integer runtime tags
                        across native scalar flow, generic equality and map
                        insertion. I do not infer a runtime tag from their
                        shared C integer representation. MAC
                        `task_811f280202174ac88a501ca3281d5e58`.
                - [ ] I audit classifier opcode coverage against emission
                  and verifier stack effects, explicitly handling or rejecting
                  each opcode instead of silently skipping unknown effects.
                  MAC `task_f90db79b0f464637a18486c44262c4d3`.
                  - [x] I reject unimplemented classifier instructions at
                    their own offsets, check classifier/emitter case parity,
                    and test representative unsupported instruction families.
                    Normal and sanitizer suites pass 1,102 AOT and 965 graph
                    checks, plus the opcode case-parity test. Compiler
                    acceptance now names `HM_NEW` at function 267 offset 0.
                    Evidence: `docs/evidence/aot-opcode-coverage.md`.
              - [ ] I reclaim unreachable nested-record snapshots during
                long-running execution, with bounded-live-state stress tests
                and alias-safe destruction. Entry-return cleanup alone does
                not bound retained memory. MAC
                `task_d152cc3913f248fb8d1483210e60f00b`.
              - [x] I store record-array fields by reference and preserve
                their element shapes through packing, calls, extraction and
                aliases; I test empty arrays and mixed-field elements.
                Normal and sanitizer suites pass 1,055 AOT checks and 965
                graph checks. Compiler acceptance advances from function 20
                to function 100's nested record packing. Evidence:
                `docs/evidence/aot-record-array-fields.md`.
              - [x] I read resolved field representations from the graph
                during emission without creating new constraints; I test
                missing edges, aliases and existing extraction behavior.
                Normal and sanitizer runs pass 1,048 AOT and 965 graph
                checks. Nested storage remains unfinished. Evidence:
                `docs/evidence/aot-resolved-field-emission.md`.
              - [x] I attach persistent shape variables to production
                classifier values, locals, parameters, results and joins,
                checking compatibility alongside existing representation facts.
                My AOT suite passes 1,048 checks normally and with translator/
                graph ASan/UBSan instrumentation; 952 graph checks pass.
                Evidence: `docs/evidence/aot-production-shape-constraints.md`.
          - [x] I separate record and record-array temporary field facts;
            their independent indices must not overwrite each other. I test
            live string-record arrays across scalar record construction and
            the reverse collision, directly and through both branch paths.
            My AOT suite passes 1,010 checks. MAC
            `task_3673443775f2477c94688b1021d6102a`; evidence:
            `docs/evidence/aot-record-fact-namespaces.md`.
          - [x] I size generated temporary arrays to actual high-water counts,
            allocate emitter record facts and snapshots dynamically, and test
            more than 256 temporaries. I check output-growth arithmetic before
            inserting the resulting declarations. My AOT suite passes 992
            checks; evidence: `docs/evidence/aot-dynamic-temporaries.md`.
          - [x] I select array constructors from emitted representations,
            including direct stack construction without locals and legacy
            inferred array kinds; strict generated-C compilation passes.
- [x] **AOT byte-character conversion.** I preserve the existing C-byte
      `vm_string_from_char` contract, including zero-byte empty text and
      independent storage. I test integer boundaries before compiler acceptance.
      MAC `task_419c47bdc8fc42e4b52eb6af1a0e9a71`.
      - [x] I emit `INT64_MIN` without an out-of-range positive C literal;
        strict C compilation and the signed-extrema arithmetic case pass.
      My AOT suite passes 904 checks. Compiler acceptance remains failing at
      `vm_mktemp_dir` (import 34). Evidence: `docs/evidence/aot-byte-character.md`.
- [x] **AOT shell execution.** I adapt the exact `nl_exec_shell` builtin
      contract, preserving raw system status. I test success, nonzero exit and
      signature rejection before rerunning compiler acceptance.
      I also preserve bounded capture/draining with independent result storage.
      MAC `task_419c47bdc8fc42e4b52eb6af1a0e9a71`.
      My AOT suite passes 842 checks. Compiler acceptance still fails at
      `vm_string_from_char` (import 32). Evidence: `docs/evidence/aot-shell-capture.md`.
- [x] **AOT lexical normalization.** I normalize builtin paths without
      fixed component/output limits, preserving relative parents and roots.
      I test long paths and rerun compiler acceptance.
      MAC `task_419c47bdc8fc42e4b52eb6af1a0e9a71`.
      My AOT suite passes 828 checks. Compiler acceptance still fails at
      `nl_exec_shell` (import 25). Evidence: `docs/evidence/aot-builtin-normalize.md`.
- [x] **Native normalization bounds.** I replace module path truncation and
      generated-native `parts[512]` overflow on leading parents with checked
      dynamic storage. I test both paths beyond their old limits.
      MAC `task_82bd388637824cc889b12204d226e75b`.
      - [x] I use checked dynamic storage for public `path_normalize` and
        generated `nl_os_path_normalize`, with 700-component/parent and
        5,000-byte tests. Evidence: `docs/evidence/native-normalization.md`.
      - [x] I remove the remaining bounded normalization/token/output buffers
        inside `path_relpath`; public normalization does not fix this caller.
        Long shared-prefix and 4,504-byte result tests pass. Evidence:
        `docs/evidence/dynamic-relative-paths.md`.
- [x] **Relative-path anchors.** I define and verify dot, mixed-root and
      unresolved-parent behavior rather than treating all normalized components
      as interchangeable. MAC `task_64696231d8984732a5a1e1c319ca043b`.
      I anchor relative inputs to one dynamically read working directory;
      reconstruction, long-cwd and unavailable-cwd cases pass. Evidence:
      `docs/evidence/relative-path-anchors.md`.
- [x] **AOT identity checks.** I preserve file/destination identity semantics
      for builtin imports, including hard links, missing paths and failed
      lookups. I test absent-destination probe cleanup in private directories.
      MAC `task_419c47bdc8fc42e4b52eb6af1a0e9a71`.
      My AOT suite passes 822 checks. Compiler acceptance remains failing at
      builtin `path_normalize` (import 24). Evidence:
      `docs/evidence/aot-builtin-identity.md`.
- [x] **AOT filesystem predicates.** I adapt builtin file/directory existence
      checks with exact signatures and exercise files, directories, missing
      paths and followed/broken links. I rerun compiler acceptance.
      I also adapt builtin removal/rename and test only private targets.
      MAC `task_419c47bdc8fc42e4b52eb6af1a0e9a71`.
      My AOT suite passes 807 checks. Compiler acceptance still fails at
      builtin `file_compare_identity` (import 22). Evidence:
      `docs/evidence/aot-filesystem-basic.md`.
- [x] **AOT builtin text writer.** I preserve both string arguments and
      return failure on short writes or failed close. I exercise generated
      executables and rerun compiler acceptance before completing this item.
      MAC `task_419c47bdc8fc42e4b52eb6af1a0e9a71`.
      My AOT suite passes 771 checks. Compiler acceptance still fails at
      import 20 (`file_exists`). Evidence: `docs/evidence/aot-builtin-text-writer.md`.
- [x] **Text write error parity.** I fix interpreter and generated-native
      writes/appends that ignore write errors, and VM/module writers that
      ignore close errors. I require injected short-write and close failures
      across these paths, not only successful regular-file writes.
      MAC `task_46cb33d875724130a6f767d998f3014b`.
      Seven production-body fault probes, 771 AOT checks and the text-file
      integration regression pass. Evidence:
      `docs/evidence/text-write-error-parity.md`.
- [x] **AOT builtin text reader.** I preserve streaming text reads, empty
      results on I/O errors and embedded-NUL rejection in generated C. I test
      real files and keep builtin and artifact bindings distinct.
      MAC `task_419c47bdc8fc42e4b52eb6af1a0e9a71`.
      My AOT suite passes 752 checks, including FIFO and injected I/O errors.
      Compiler acceptance remains failing at builtin `file_write` (import 19).
      Evidence: `docs/evidence/aot-builtin-text-reader.md`.
- [x] **AOT scalar filesystem imports.** I adapt the known filesystem
      string, boolean and integer contracts through exact artifact bindings,
      preserving argument order and native return widths. I execute real
      library tests and rerun full compiler acceptance.
      MAC `task_419c47bdc8fc42e4b52eb6af1a0e9a71`.
      My AOT suite passes 710 checks; full compiler acceptance remains failing
      at builtin import 18 (`file_read`). Evidence:
      `docs/evidence/aot-scalar-filesystem.md`.
- [x] **AOT owned filesystem adapter.** I bind `fs_walkdir` to its absolute
      artifact path, check its array ABI and release entry point, copy returned
      strings into AOT storage, then release the foreign result. I test exact
      binding and failure cases before using it in compiler acceptance.
      MAC `task_419c47bdc8fc42e4b52eb6af1a0e9a71`.
      My AOT suite passes 647 checks. Compiler acceptance now stops at
      `path_normalize`; remaining adapters stay open under the same task.
      Evidence: `docs/evidence/aot-owned-walk-adapter.md`.
- [x] **Owned walk-result release.** I add an opt-in C release operation for
      unmodified, exclusively owned `fs_walkdir` results, freeing their copied
      strings before the array. Existing callers retain current behavior.
      I test escaped copies and shared-array refusal before AOT adapter use.
      MAC `task_419c47bdc8fc42e4b52eb6af1a0e9a71`.
      My release probe and directory-walk gate pass (one host-limit skip).
      Evidence: `docs/evidence/walk-result-release.md`. AOT adaptation remains open.
- [x] **AOT artifact boundary diagnostics.** I distinguish unsupported exact
      artifact imports from builtin ABI mismatches and pin the filesystem-array
      boundary against name-only rebinding. This does not implement the native
      array adapter. MAC `task_419c47bdc8fc42e4b52eb6af1a0e9a71`.
      The AOT suite passes 617 checks; One-IR acceptance still fails at the
      artifact adapter. Evidence: `docs/evidence/aot-artifact-boundary.md`.
- [x] **One-IR compiler execution baseline.** I exercise the current full
      compiler through bytecode emission, AOT C translation, native compilation
      and a compiled hello program. I retain bounded subprocess groups and
      exact output checks. The 75-commit `8299138f` branch remains unmerged
      until its useful changes are reconciled with current safety contracts.
      Release parent MAC `task_cffdafd16e641ac417ccfddb962534b9`.
      `make test-one-ir-compiler` now exposes the first blocker: compiler
      bytecode emits, but `nvm2c` refuses the `fs_walkdir` import's host ABI.
      Execution acceptance remains failing under MAC
      `task_419c47bdc8fc42e4b52eb6af1a0e9a71`. Evidence:
      `docs/evidence/one-ir-compiler-baseline.md`.
- [x] **Union resource propagation.** I propagate ownership obligations
      through named union payloads and mixed record/union cycles, preserving
      module-owned lookup. I require classifier tests; generic substitution,
      tuples and control-flow enforcement remain open. MAC
      `task_91ae827be4154eaa8f22698aeecc8cf1`.
      Record/union classifier checks pass; evidence:
      `docs/evidence/union-resource-classification.md`.
- [x] **Nested-record resource classification.** I classify ordinary records
      containing resource records without recursive depth limits, including
      cycles. I test deep graphs and keep full path-sensitive ownership and
      union/tuple/collection propagation separate. MAC
      `task_91ae827be4154eaa8f22698aeecc8cf1`.
      The dedicated unit gate passes cyclic, 300-record and module-owned
      field cases. Evidence: `docs/evidence/resource-classification.md`.
- [x] **Affine parity gate coverage.** I test positive execution and both
      existing ownership rejections on the C seed as well as both bootstrap
      stages. Rejections must carry ownership diagnostics and preserve prior
      output. I replace the predictable temporary path with private fixtures.
      Passing the old gate did not establish C-seed rejection parity.
      The corrected gate fails two C-seed cases: unresolved ownership is
      accepted, and use-after-move fails only a shadow assertion. Both bootstrap
      stages pass the limited cases. Checker acceptance remains open under MAC
      `task_91ae827be4154eaa8f22698aeecc8cf1`; evidence:
      `docs/evidence/affine-parity-baseline.md`.
- [x] **Affine contract branch reconciliation.** I retain the current
      contract, which matches `a08341e5`, and reconcile that head plus the older
      `cb99b86e` contract. This records design ancestry, not implementation
      conformance; the C-seed recovery prototype remains under review.
      MAC `task_4ac22044ffda9f93b336a85573293bc2`.
      Both changed documents match exactly; evidence:
      `docs/evidence/affine-contract-branch-reconciliation.md`.
- [x] **Compact-schema branch reconciliation.** I retain the compact operand
      aliases and tests already integrated by PR #139 while reconciling
      `09187898`. I clarify that schema checks do not establish runtime compact
      encoding support and require the schema gate. MAC
      `task_b7c534c9150ee4bb4b8bca62b37b50dd`.
      Schema generation is current and 33 tests pass. Evidence:
      `docs/evidence/compact-schema-branch-reconciliation.md`.
- [x] **Assembler snapshot branch reconciliation.** I verify that `24ac886b`
      exactly matches integrated tree `dfa1aca3`, retain newer capture rules,
      and rerun its include-phase, restored-input and search-order tests.
      MAC `task_92e6817607cf4071ab614289911a9a41`.
      Three tests pass in 263.852 seconds. Evidence and remaining-head inventory:
      `docs/evidence/assembler-branch-reconciliation.md`.
- [x] **Wrapper-publication branch reconciliation.** I retain the staged
      publication implementation already integrated from `c56c6e7e`, including
      newer callback/array link dependencies, while reconciling `df031115`.
      I require wrapper unit and adversarial publication tests. MAC
      `task_0750c33a06a14dd39baf4d3e77e37a0d`.
      Five unit tests and seven publication methods pass. Evidence:
      `docs/evidence/wrapper-branch-reconciliation.md`.
- [x] **VM shadow completion handshake.** I reject foreign `exit(0)` before
      shadow execution returns, preserving prior bytecode. I require a private
      close-on-exec completion channel as well as a successful child status.
      MAC `task_9d9eefa909be4990be0151bed7439953`; both early-exit regression
      cases fail before the fix, then all 40 bytecode-shadow methods pass.
      Evidence: `docs/evidence/vm-shadow-branch-reconciliation.md`.
- [x] **VM-shadow worker reconciliation.** I compare `a1399362` with my
      current graph-wide shadows, typed math lowering and signature checks.
      I retain dependency shadows by default and current publication guards,
      and require the complete bytecode-shadow test module before recording
      ancestry. MAC `task_4c6ff6e9986a49d6a01701a66b8842d6`.
- [x] **PR #295 reconciliation.** I integrate main's verifier cleanup and
      ownership-propagation regression while retaining my allocation-counted
      rejection tests and unknown-effect cleanup. I require schema and verifier
      gates before recording the reviewed worker head as ancestry. MAC
      `task_69dab2ee6f1a4c96845ba7139cfc360f`.
      Schema, 95 verifier tests and allocation-counted checks pass, including
      the new retain/release path. Evidence: `docs/evidence/pr-295-reconciliation.md`.
- [x] **Record-alias branch reconciliation.** I retain recursive record
      copies before binding replacement rather than restore symbol-scan
      reference counts from `9698ce08`. I retain its direct-call alias
      regression alongside current nested-record shadows and run evaluator
      and environment gates. MAC `task_1ac1c2fd6bc04b5e81932553901b0d4b`.
      Bootstrap, evaluator and 32 environment checks pass. Evidence:
      `docs/evidence/record-alias-branch-reconciliation.md`.
- [x] **Bootstrap branch reconciliation.** I retain behavioral invalidation
      checks and direct self-hosted stage dependencies while reconciling
      `d152181d`. I expose the existing suite as `test-bootstrap-dependencies`
      and run it from quick and full test entry points. MAC
      `task_85191b435c95480d9db52e3671d1e740`.
      Eight behavioral tests pass in 17.126 seconds. Evidence:
      `docs/evidence/bootstrap-branch-reconciliation.md`.
- [x] **String-search branch reconciliation.** I retain my shared byte-offset
      search implementation and broader source fixtures while reconciling
      `07d38702`. I retain its missing first-search null-needle assertion and
      require runtime units, native stages, emitted C and VM execution.
      All focused gates pass; evidence:
      `docs/evidence/string-search-branch-reconciliation.md`.
      MAC `task_a9b152cf5299491694d96a2385527e98`.
- [x] **FFI argument branch reconciliation.** I retain current ABI handling
      and the shared foreign-argument ceiling while reconciling the older
      float-refusal and execution-path argument-limit branches. I require
      mixed-signature execution, VM over-limit rejection and protocol gates;
      I do not restore the obsolete ten-argument dispatch limit or float ban.
      Evidence: `docs/evidence/ffi-argument-branch-reconciliation.md`.
      Release parent MAC `task_cffdafd16e641ac417ccfddb962534b9`.
- [x] **Verifier branch reconciliation and rejection cleanup.** I retain
      current signature-aware stack propagation and valid alternate paths after
      returns while reconciling the older stack/range/container heads. I fix
      unfreed verifier work arrays on rejection and require allocation-counted
      failure tests plus schema, verifier and container gates. MAC
      `task_69dab2ee6f1a4c96845ba7139cfc360f`.
      Allocation-counted rejection/failure/success checks pass, along with
      94 verifier tests, 2632 NanoISA checks, 29 v2 container checks and schema
      verification. I retain both old heads in ancestry without restoring
      their weaker stack or terminator policy. Evidence:
      `docs/evidence/verifier-branch-reconciliation.md`.
- [ ] **Full branch inventory and reconciliation.** I inspect local-only and
      remote heads, not just open PRs. At `1c7105a1`, 21 heads remain outside
      integration ancestry. Six contain only patch-equivalent commits already
      in history; I reconcile those without source replacement. The other 15
      require content review, including affine ownership, verifier/FFI work,
      lease snapshots and the 75-commit 4.6 frontend branch. Evidence:
      `docs/evidence/branch-inventory-1c7105a1.json`. Release parent MAC
      `task_cffdafd16e641ac417ccfddb962534b9` remains open.
      - [x] I retain the six patch-equivalent heads as merge ancestors without
        source replacement. Current verifier and co-process protocol gates
        pass. Evidence: `docs/evidence/patch-equivalent-branches.md`.
- [x] **PR #269 reconciliation.** I retain my converged AOT call facts and
      stronger aggregate rejection checks while reconciling the older flat
      record/variant patch. I require the structured-C suite and keep declared
      layouts, nested fields and linked aggregate metadata open under MAC
      `task_a4fde0d59ad24fe18c285a76ad58c176`.
      The suite passes 609 checks, including an added integer/record parameter
      conflict with an unused argument. I retain production code unchanged.
      Evidence: `docs/evidence/pr-269-reconciliation.md`.
- [x] **PR #267 reconciliation.** I compare its actual external-capture
      failure patch with my newer fail-closed implementation, retain current
      diagnostics and expanded recovery checks, and verify concurrent compiler
      isolation plus cold/warm capture failure before merging its ancestry.
      MAC `task_443e8107d0ff4350999e0d5186a809f1`.
      Its base is already integrated and my newer implementation subsumes the
      patch. Six isolation/capture methods pass in 114.378 seconds; I retain
      current source unchanged. Evidence: `docs/evidence/pr-267-reconciliation.md`.
- [x] **Release-gate string documentation.** I document `str_last_index_of`
      and reconcile the string section count with the builtin registry.
      `tests/check_stdlib_docs.sh` passes for all 156 builtins.
- [x] **MAC shell argument safety.** I quote ledger arguments as data and
      test metacharacters against a fake CLI; titles must not execute shell code.
      Native and VM compilation, dependency/root shadows and execution pass
      exact-argument checks for quotes, substitutions, newlines, Unicode and
      option-like values across create, close, show and list. No live ledger
      writes occur in these tests. Positional data follows `--`; the raw
      `exec_command` API intentionally remains a shell-code execution API.
      MAC `task_c3e8254ad9f8ce84143f59ca306e24fb`.
- [x] **Process command length.** I reject or safely accommodate commands beyond
      the process runner's fixed command buffer, with boundary tests.
      My module runner already uses direct shell invocation and file-backed
      capture. I share that implementation with the VM builtin, interpreter
      and generated native helper; the latter two currently drain pipes
      sequentially. I test long commands and simultaneous stdout/stderr.
      All four paths now share `runtime/process_capture.h`, eliminating the
      VM's 4096-byte command buffer and sequential-pipe readers. Native/VM
      builtin and module tests pass 16 KiB commands and 128 KiB per stream.
      VM tests reject commands beyond the host argument limit without running
      a truncated prefix, and check signal status, null input and NUL output.
      Interpreter, transpiler and MAC boundary gates pass. Strict dispatch
      remains 175 selected, 172 identical, three failures and zero skipped.
      Command deadlines, output quotas and array/string ownership remain
      separate runtime boundaries; this does not establish process isolation.
      MAC `task_c3e8254ad9f8ce84143f59ca306e24fb`.

- [x] **Explicit Stage 2 acceptance.** I reconcile the earlier main-only
      result (11 of 16 cases) with my integration compiler. Fresh Stage 2
      selection passes all 20 runner checks, including loop, recursion,
      let/set, infix, match, import-path and CLI cases. I retain that earlier
      failure history without presenting it as my current result. Evidence:
      `docs/evidence/main-reconciliation-pr297.md`.
      MAC `task_4d134cb7dd9c401c9aa8926cddbdeef3`.

- [ ] **Nested-array shadow evaluation.** I support nested dynamic arrays
      in the C evaluator so native nested-array regression programs can also
      execute their full behavior inside shadows. The current evaluator
      rejects those arrays with `Unsupported array element type`.
      MAC `task_23e8d93323aa4af392384aa096189509`.

- [x] **Local hook migration.** I preserve and disable retired Beads shim
      hooks in this checkout so commits no longer invoke the removed ledger.
      MAC `task_a6600695ae154212a97e1b012c0dff20`.

- [x] **Local integration recovery.** I finish the interrupted rebase onto
      current main, preserve applicable compiler, module-header and shadow
      changes, and retain upstream removals. `make test-quick` and
      `make test-c-backend` pass; both changed examples compile and run
      with the C seed and rebuilt Stage 2. SDL helper headers pass a C
      syntax check. Native bootstrap binaries differ; I claim passing
      smoke and parity checks, not a fixed point.
      MAC `task_52371d21b5174f2aba3acc13074d28f0`.

- [x] **5.0 / self-hosted nested-array indexing.** I preserve every remaining
      array level while inferring and emitting nested `at` calls, and test both
      intermediate array values and the scalar value through native execution.
      MAC `task_f32d71bbad07448c8656843015e72ef3`.
- [x] **Interpreter reliability.** I return an owned string from record field
      access so call-frame cleanup cannot free record-owned storage. Evaluator
      coverage checks direct returns, local reassignment, 100 repeated reads,
      and shadow execution. MAC `task_de992b992c064ceb917bda531312f531`.
- [x] **4.4 release.** I merge the 4.1–4.4 product branch (`feat/forth-core-suite`)
      to `main`, close superseded PRs with evidence, and leave 5.0 / Standard
      System / conflicting Forth-IDE work unmerged. I do not merge MAC lease
      branches that are already contained in the product branch.
      Squash-merged as PR #243 (`3c28a3cf`). Closed #237, #238, #240, #241,
      #244, #245. Left #242 (NanoISA 5.0 side quest) open.
      MAC `task_b19f5e408d373ec2bf1efa6d54e4d366` (supersedes failed
      `task_7b6e8957be5b4abf8b0ea4751fd9b635`).
- [x] **4.1 leftover.** I publish the precise Forth 2012 system label. Passing
      Jackson files is evidence, not a Standard System or Core banner.
      `docs/FORTH_STANDARD_SYSTEM.md`, `make test-forth-jackson`.
      MAC `task_6087b948f1c9a7672420b4e1ea72bd35`.
- [x] **4.4 follow-up.** `make module-self-test` on Ubuntu GCC 15 stops at
      `modules/sdl_term/mvp.nano`: it calls `SDL_KeyCode` as a function and
      uses extern SDL/term calls outside `unsafe`. `make examples` still
      succeeds. Fix the MVP so every module MVP compiles.
      Compile-only smoke with externs in `unsafe`. `make test-sdl-term-mvp`.
      MAC `task_d35028a1b27f45948ffccb351fc7bd4a`.
- [x] **4.5 / Phase 19.** Effects to deployment policy, trap journal and
      deterministic replay, then observability and provenance. This is part
      of the public-release bar, not work after a 4.4 tag.
      `docs/NSI_EFFECTS.md`, `make test-nsi-policy test-nsi-journal test-nsi-obs`.
      MAC `task_2bd5c3128983683b78134ad9257b7d3b`,
      `task_36c1ce545c4d12a0a2c520dd146ea5f1`,
      `task_860edd8c08843bf90ad9559d0b821cef`.
- [x] **4.4 release.** `test_interpret_file_refill` called `mkstemp` twice on
      one buffer. The first call consumes the `XXXXXX` template; Linux then
      returns `EINVAL` (`FAIL: mkstemp unknown`). Restore the template before
      each call.
- [x] **4.4 release.** GCC `-Werror=stringop-truncation` on
      `strncpy(g_project_root, exe_path, PATH_MAX-1)` in `src/main.c` failed
      the AddressSanitizer job. Copy with an explicit length instead.
- [x] **4.4 release.** GCC 15 `-Werror=stringop-truncation` on
      `strncpy(..., FORTH_PATH_MAX-1)` in `src/forth/forth_session.c` when
      source and destination are both `FORTH_PATH_MAX`. Copy with an explicit
      length. `make test-forth-session` passed on Ubuntu GCC 15 (`ubuntu.local`).
      MAC `task_5e86f861c17e44859bc371a8d0c4b27d`.
- [x] **4.4 release.** The same truncation warning on `strncpy` into
      `PATH_MAX` project-root buffers in `src/nano_main.c` (AddressSanitizer
      job), `src/repl_main.c`, `src/lsp_server.c`, and `src/dap_server.c`.
      Use `safe_strncpy`.
- [x] **4.4 release.** GCC `-Werror=stringop-truncation` on
      `strncpy(q->call_id, cid, 31)` in `src/nsi_runtime.c` when both arrays
      are 32 bytes (`make test-nsi-runtime`). Copy with an explicit length.
- [x] **4.4 release.** GCC `-Werror=format-truncation` on
      `snprintf(tiny, 8, "%zu", n)` in `src/nsi_fabric.c`
      (`make test-nsi-fabric`). The decimal form of `size_t` needs 21 bytes.
- [x] **4.4 release.** YAML in `.github/workflows/ci.yml` had two `run:`
      keys on the sanitizer install step, so GitHub never started CI on
      `ab2bf349`. Restore the AddressSanitizer step.
- [x] **4.4 release.** Code Coverage `make test` hits `TEST_TIMEOUT` (1800s)
      during Jackson word-set REFILL under `-fprofile-arcs`, then Forth PTY
      liveness fails (`banner/prompt never appeared`) because gcov stalls
      the REPL. Build and Test and Memory Sanitizers already passed those
      suites. Skip word-set REFILL, PTY, and IDE smoke under coverage;
      keep the Jackson pin / INCLUDE-gap test. Coverage job succeeded on
      `10e97f12` in PR #243.
      MAC `task_d10e63b263724e4ab4ae75dd4a281529`.
- [x] **Public release — docs, deck, user guide.** I update README,
      CONTRIBUTING, PERSONA, indexes, `docs/RELEASE_4.5.md`, the user
      guide (including a Secure Runtime chapter), and the developer deck
      for the 4.1–4.5 bar. Last public GitHub Release was `v4.0.0`. I am
      a language and a secure runtime. I do not claim a Forth Standard
      System, GNU Emacs, a kernel, or that the system is
      internationalized. Landed in PR #247 (`a96d24d3`).
      MAC `task_d2f80f3c3ca6ee5f81070545a5a2ac5f`.
- [x] **Public release — test pipeline and tag `v4.5.0`.** After the
      docs/deck/guide land, I run `make test`, CI, and
      `make release-docs-check`, then tag one public cut covering 4.1–4.5.
      `./scripts/release.sh 4.5.0`. Annotated tag `v4.5.0` on `ef32c833`
      (PR #248). GitHub Release:
      https://github.com/jordanhubbard/nanolang/releases/tag/v4.5.0
      I do not start 4.6 or 5.0.
      MAC `task_f3049ae5d8389ecc96ff28754e52b31b`.
- [x] **Public release — `release.sh` waits for checks and tags the
      squash commit.** `gh pr checks --watch` exited 1 with "no checks
      reported" on PR #248 before GitHub created runs. After squash,
      local `main` still had the pre-squash commit, so
      `git pull --ff-only origin main` could not land. Retry until
      checks exist; `git reset --hard origin/main` before the tag.
      `tests/test_release_workflow.sh`.
- [x] **Public release — Google Workspace publish.** In-place update of
      the existing Slides and Doc after `v4.5.0`. Read-back: 16 slides,
      16 notes, 34 headings. Preview URLs are in `docs/presentation/`.
      LinkedIn copy is `docs/LINKEDIN_4.5.md`.
      MAC `task_d0a1fe6953d749c195141ba921c5cc24`.
      Recorded on `docs/record-4.5-google-publish`.
- [x] **4.6 / Phase 21 — shared frontend contract.** Every frontend emits
      verified `.nvm` v2, uses the same verifier, NSI, capabilities,
      FFI isolation, debug, profiler, and `nvm2c`, and keeps language
      work in desugar/typecheck. Bounded goals are published before a
      language starts. Frontend-private opcodes fail closed. NanoLang
      and Forth already accept a shared library. Scheme, ML, Actor, Dataflow, Object, Shell, and Logic are
      implemented as bounded laboratory frontends. The frontend matrix is
      `docs/FRONTEND_MATRIX.md` / `make test-frontend-matrix`.
      `docs/NANOISA_FRONTEND.md`, `src/nanoisa/frontend.c`,
      `make test-frontend-contract`, `make test-scheme`, `make test-ml`,
      `make test-actor`, `make test-dataflow`, `make test-object`,
      `make test-shell`, `make test-logic`.
      MAC `task_e62d1cd35b49296604012df95de7911b`.
- [x] **4.6 / Phase 21 — laboratory languages.** Scheme, ML, Actor,
      Dataflow, Object, Shell, Logic, then the frontend matrix.
      NanoLang stays my native language.
      `make test-scheme`, `make test-ml`, `make test-actor`,
      `make test-dataflow`, `make test-object`, `make test-shell`,
      `make test-logic`, `make test-frontend-matrix`.
      `docs/FRONTEND_MATRIX.md`.
      MAC `task_6647a64cc76edac6e3d0f62c228d98c4` (Scheme),
      `task_3eed929292a80ed58dd3a8db1ed701b6` (ML),
      `task_0850b9adc62c593b8e4e180e070efcee` (Actor),
      `task_9cb85a523c197b9e2c80ddcffe9ed31a` (Dataflow),
      `task_90023c92e9fb3841aab9fcc71d8cf90d` (Object),
      `task_ee91ee94749200ab6309e6c05df3dd61` (Shell),
      `task_69fc7f6660a1976f10606a42d78fd264` (Logic),
      `task_92c497c72b7aa1fc993d666f66843759` (matrix).
- [x] **5.0 release — trustworthy incremental C builds.** I generate and
      include compiler dependency files for every C object so a changed or
      deleted header invalidates each translation unit that used it. My
      regression proves an indirect header is recorded and that changing a
      recorded dependency makes `make` rebuild the affected object. Verified
      with `make rebuild`, `make test-make-header-dependencies`, and
      `make test-quick`. GitHub #211.
      MAC `task_9f8a6bf48d4c4117b1a551ee35c0b055`.
- [ ] **5.0 audit — language contract and claims.** I reconcile PERSONA,
      README, specification, guide, and compiler behavior; version one
      executable language contract and test each claim on the C seed,
      self-hosted compiler, VM, and AOT where applicable. I distinguish
      project policy, implementation, and proof. MAC
      `task_2b291a75ca2840519d47e08bf991c021` tracks this audit program.
      On 2026-09-12, MAC marked that task failed after a dispatcher-managed
      agent lease expired without captured evidence; it requests manual
      ledger repair. My repository audit and remaining roadmap work stay
      active. This is not a failed compiler or test result.
      My bootstrap reporting now names the checks performed; canonical
      artifact equality and semantic correctness remain separate.
- [x] **5.0 contributor contract — current compiler and verification rules.**
      I reconcile CONTRIBUTING with current shadow execution, backend scope,
      verifier limits and fuzzing practice. `make test-release-gates` currently
      rejects its staleness relative to verifier/fuzzing changes. Its claims
      that shadows prove correctness, only externs are exempt, and tests run
      in the final binary also contradict the tested contract. I repair those
      claims and rerun the gate without a release acknowledgement override.
      I now distinguish the compiler paths, separate shadow test processes,
      default dependency selection, warning exemptions, formal correspondence,
      abstract reference balance and parser fuzzing limits. Sixteen language
      claim methods, all ninety-three verifier tests and all twenty-two release
      gate methods pass on Darwin (2026-09-13), without an acknowledgement.
      The four parser seeds also replay successfully with the installed LLVM
      libFuzzer target; this is not a mutation campaign or leak-freedom claim.
      MAC `task_bb0635cd76e04f898ea7886298377a01`.
- [x] **5.0 release freshness — preserve Git paths.** I repair the shared
      release helper's stripping of the leading porcelain status space, which
      truncates the first unstaged path and still calls edited CONTRIBUTING
      untouched. I use NUL-delimited Git output for committed and working paths,
      preserve both rename endpoints, and test real temporary repositories
      with unstaged edits, renames, spaces and newline-containing filenames.
      Real Git tests now retain the first unstaged path and both rename
      endpoints before and after commit, including literal arrows, leading/
      trailing spaces and CR/LF names. Raw Git output avoids text-mode newline
      conversion. All twenty-two release-gate methods pass.
      MAC `task_04bd7f1a9f5f4396b7ba9fb14dd5d370`.
- [x] **5.0 contract claims — typing and shadow execution evidence.** I
      characterize inferred locals, explicit function boundaries, immutable
      assignment, boolean conditions and shadow handling on the C seed,
      Stage 2 and bytecode frontend. I reconcile specification, canonical
      style and persona with those observations, distinguishing intended
      language rules and project shadow policy from current enforcement.
      Five test methods cover 22 compiler cases, including the specification's
      unsafe math wrapper on the C seed. `make test-language-claims` passes
      with the existing tools (2026-09-12). README, persona, canonical style
      and the specification now agree about this observed shadow gap; the
      specification remains a draft with broader coverage explicitly open.
- [x] **5.0 document-pair acceptance — inspect the artifacts.** I replace
      the regeneration script's unconditional acceptance record with checks
      of the actual PPTX and DOCX: slide count and geometry, text-frame
      intersections, speaker notes, heading hierarchy and explicit placeholder
      and credential patterns. I preserve both artifact paths in the manifest
      and test malformed artifacts and nonzero gate exits. Mechanical acceptance
      does not replace rendering or visual inspection. The documented verifier
      was missing; the renderer printed overlaps without failing. I now run
      the actual verifier from regeneration, invalidate prior acceptance before
      authoring, and preserve both artifact paths with the actual slide count.
      Ten tests pass, including real builder success and authoring-failure
      recovery in isolated outputs. The retained pair passes mechanical checks
      with sixteen slides and thirty-four headings. The full release-gate suite
      still rejects CONTRIBUTING staleness, tracked above; no override is used.
      MAC `task_8708bf8104604c278ad846f2b7f476a9`.
- [x] **5.0 developer deck — shadow enforcement claims.** I replace the
      slide-two `PROOF` label and the claim that an unshadowed function cannot
      compile in `docs/presentation/build_deck.py` with my tested policy,
      warning and exemption boundary. I regenerate and visually verify the
      local artifacts and add a claim regression check before publication.
      MAC `task_467da218e63ea9836c2fa6412c9026b0`.
      I also reconcile the companion narrative's shadow statement and its
      incorrect `gcd` example, which returns its first argument while claiming
      a result of six. The shared claim ledger's open #211 statement is stale.
      Full source review also finds claims of whole-program proof, complete
      ownership and production runtime security, plus a bounds example missing
      the offset precondition. I reconcile these in the same unpublished 5.0
      development draft, retain labeled historical measurements, share one
      executable example between both artifacts, and test the extracted code.
      The local pair is regenerated; extracted DOCX code passes on all three
      backends, and false assertions preserve prior outputs. Mechanical checks
      pass. The host's soffice launcher points to a missing application; I used
      LibreOffice 24.2.7.2 in a disposable Ubuntu container instead. I inspected
      all sixteen slides and five narrative pages, fixed the split command
      block, and gave the closing section its own page. Code excerpts stay
      together, with an artifact regression check. QA records exact artifact
      hashes and renderer scope; neither member is externally published.
- [x] **5.0 imported shadows — selection characterization.** I test direct
      and transitive imports with failing dependency shadows on the C seed,
      Stage 2 native driver and bytecode CLI, then compile that dependency as
      the root to check rejection. This measures selection, not a complete
      imported-shadow policy. On Darwin, all three paths skip the failing
      imported shadow at both depths, run the root assertion using its helper,
      and reject that dependency's shadow when compiled as root (12 compiler
      cases in `test_imported_shadow_selection`). My six language-claim tests
      pass. I use distinct wrapper names to isolate selection from the open
      namespace defect below; this is not Linux or FFI acceptance evidence.
      MAC `task_53197aae0a914dfcaafe490af1a18d3a`.
- [x] **5.0 C-seed shadows — remove foreign-call skipping.** I execute
      explicit root shadows regardless of foreign calls in the tested function
      or shadow body. I test passing and failing assertions with unreachable
      foreign calls and retain the broader FFI error and isolation gates.
      Six C-seed compiler cases cover calls in function bodies, unsafe blocks
      and shadow bodies. Seven language-claim tests and all 95 evaluator tests
      pass on Darwin with the rebuilt C seed. Bootstrap Stage 1/2 smoke and
      no-C-seed checks pass; native binaries still differ. This does not
      establish foreign error propagation, isolation or Linux acceptance.
      The post-bootstrap rerun passes all 16 language-claim/native-shadow tests;
      all nine interpreter FFI tests also pass.
      MAC `task_53197aae0a914dfcaafe490af1a18d3a`.
- [x] **5.0 interpreted shadows — foreign failures are test failures.** I
      propagate failed symbol resolution and marshaling into shadow results
      instead of returning an ordinary void value that a test can ignore.
      My checked call API separates failure from a successful void return.
      I test ignored missing-symbol and unsupported-float calls, JSON failure
      locations and preservation of existing output, with a resolved native
      integer-call control. Ten FFI tests cover malformed metadata, mismatched
      value tags, uninitialized dispatch, recovery, introspection and void.
      Those tests and all 95 evaluator tests pass on Darwin and Linux GCC;
      the nine focused compiler cases pass on both. My Darwin bootstrap smoke
      and no-C-seed checks pass; native binaries still differ. The prior `erf`
      result came from unsupported native dispatch, which remains work below.
      This is shadow failure accounting, not general interpreter exception
      propagation, foreign isolation or proof of the declared C signature.
      The final Darwin language/native-shadow rerun passes all 17 tests.
      MAC `task_48a773d48f4843ea92039b6b4dd381e3`.
- [x] **5.0 interpreter FFI — signature-correct native dispatch.** I replace
      integer-returning pointer casts and unaligned byte buffers with libffi
      calls and aligned typed slots. My fixed-arity scalar/pointer calls now
      handle doubles, bools and void correctly in the tested signatures.
      Eleven FFI tests include real mixed native signatures, pointer/string
      results and ten-argument integer/float calls; these and 95 evaluator
      tests pass on Darwin arm64 and Linux arm64 GCC. Ten focused compiler
      cases pass on both; my full Darwin language/native-shadow suite passes
      all 17 tests. Bootstrap smoke/no-C-seed checks pass; native binaries differ.
      Editor-session tests and shared-session library builds pass on both.
      Linux ASan/UBSan passes with the bridge and driver instrumented; other
      objects and libffi are not instrumented, and leak detection is disabled.
      My Makefile, Linux CI package lists and build guide record the libffi
      dependency. Workflow YAML parses; hosted CI has not run in this check.
      Array arguments, non-opaque aggregates and variadic calls are outside
      this fixed-arity boundary. Native declarations remain trusted, not proved.
      MAC `task_b6f54605c59c4e1abff628422c77922f`.
- [x] **5.0 shadow execution — consistent compiler/runtime enforcement.** I
      make failing shadow assertions observable on every supported compiler
      path. My C-seed driver runs shadows during compilation; I measure the
      self-hosted and VM paths before claiming consistent execution or order.
      I preserve the requirement for useful tests; documenting a gap does
      not satisfy it.
      The failing-shadow characterization is now rejected by the C seed,
      bytecode CLI and Stage 2 native driver. The later transitive-default
      work implements my creator's imported-shadow policy, dependency-first
      execution, explicit root-only opt-out and production separation.
      Source-only emission deliberately does not execute tests. I re-ran
      85 methods across C-seed imports, native shadows, bytecode shadows,
      native shadow emission, supervision and language claims on Darwin:
      all pass in 281.286 seconds with no skips. These tests include failing
      assertions, deadlines, prior-output preservation, diamond order and
      source-only selection. Complete language/backend parity remains open
      under its own acceptance items; this closes shadow execution, not that
      broader claim.
      MAC `task_53197aae0a914dfcaafe490af1a18d3a`.
- [x] **5.0 imported shadow prerequisite — function ownership.** I repair
      same-named pure wrappers across qualified imports, preserving module
      ownership in lookup and rejecting true local duplicates. I test root,
      imported and transitive calls through compiler execution; the broader
      namespace-isolation gate below remains open for foreign declarations
      and colliding module identities.
      I now preserve pure function ownership in C-seed lookup, both interpreted
      call paths, module C emission and bytecode registration/lowering. My
      three-level wrapper and same-named private-helper fixture executes with
      inferred and declared module names on Darwin and Linux. True local
      duplicates are rejected. At that checkpoint Stage2 still failed the
      fixture's useful root shadow; the following gate owns its implementation.
      That Stage2 gate now passes the pure wrapper fixture on both platforms;
      foreign-name and nominal isolation remain in the broader gate.
      The 63 codegen tests, evaluator, typechecker, environment and 28 bytecode
      shadow tests pass on Linux. Darwin bootstrap smoke tests pass, but its
      Stage1 and Stage2 binaries still differ. This is not full module isolation.
      All 48 Darwin language-claim, native-shadow and bytecode-shadow methods
      pass, including the explicit Stage2 boundary characterization; Linux
      passes the 39 applicable language-claim and bytecode-shadow methods
      without a self-hosted compiler build.
      MAC `task_4c6ff6e9986a49d6a01701a66b8842d6`.
- [x] **5.0 Stage2 declaration ownership prerequisite.** I retain each input
      file's start in merged source and reject duplicate non-extern function
      declarations within that owner. I distinguish these from same-named
      declarations in different files, test root and imported duplicates,
      and preserve prior output on rejection. This does not resolve calls.
      My old Stage2 accepted duplicate declarations at all three
      tested depths. The rebuilt compiler rejects them in native and C-source
      modes, preserves the prior bytes and reports `M0001` in JSON. Root and
      direct-import rejection also runs across all three compilers. A repeated
      compatible extern declaration remains accepted. My owner helper tests
      cover empty input and empty-file boundaries. Darwin bootstrap smoke and
      no-C-seed checks pass; native binaries still differ.
      All 40 language-claim, native-shadow, CLI, import-path and build-isolation
      methods pass on Darwin with Stage2. No Linux bootstrap was run in this
      increment.
      MAC `task_9706ba834a2444a7a613d701ba5eceb8`.
- [x] **5.0 Stage2 physical string emission.** I escape physical newline and
      carriage-return bytes in plain string literals without double-escaping
      source escape spellings. I execute a multiline imported function and
      preserve its value; a multiline global is also checked at runtime.
      Darwin and Linux pass. Interpolation escaping remains a separate boundary.
      MAC `task_634c83a18dbb4a47ae1597a07549cca5`.
- [x] **5.0 Stage2 physical token positions.** I count physical newlines
      consumed inside literals before locating subsequent tokens. I reproduce
      owner drift across merged-file boundaries, test escaped versus physical
      newlines, and resolve the introspection lexer call in both bootstraps.
      I check subsequent token lines and columns; this is not complete original
      source provenance or interpolation-internal location coverage.
      MAC `task_9c6fae270c964a25b8949c5929e2f686`.
- [x] **5.0 Stage2 emitter lexical environments.** I prevent inner bindings
      from mutating the outer `GenEnv` arrays. My selective-alias fixture
      exposed an inner local leaking into resolution after the block. Copying
      the environment arrays fixes ordinary blocks, loop and match bindings;
      those cases execute through Stage2 on Darwin and Linux.
      MAC `task_123d7299a3824f0f9383e5de4401360a`.
- [x] **5.0 standalone Linux fixture flags.** I compile small runtime-free
      emitted fixtures at `-O2`, as in my CLI source-output test. GCC otherwise
      retains unused static array helpers whose runtime objects this small
      harness does not link. I retain the `NDEBUG` assertion controls; this
      does not establish general unoptimized, runtime-free C linkage. My
      native driver links runtime sources; these small harnesses do not.
      MAC `task_bc02b0a4d83141ceb2153abc107f2f92`.
- [x] **5.0 Stage2 inactive binding state.** I let library callers use my
      transpiler without driver initialization. My standalone emitter test
      exposed a NULL-array read in the new binding table. I guard inactive
      lookups and verify all nine standalone emitter methods through C-seed
      and Stage2 compilation on Darwin and Linux.
      MAC `task_4949e598ba254bd2a4f7f7dfa8abf5d9`.
- [x] **5.0 Stage2 syntax-aware function alias binding.** I replace raw
      `apply_all_aliases` source rewriting with name resolution that preserves
      local bindings, strings, comments and field labels. Inspection found
      that the old replacement scanned those bytes without syntax context.
      I retain import declarations for the binding pass and test qualified,
      selective and reused aliases with distinct importer identities. Both
      platforms pass; selective type aliases remain in the broader gate.
      MAC `task_9706ba834a2444a7a613d701ba5eceb8`.
- [x] **5.0 Stage2 unresolved qualified calls.** I reject qualified calls
      without an importer binding before native or C-source publication,
      including calls inside aggregates. Before the repair, my controls
      accepted a missing member and an unknown alias by calling a root
      function with the same short name. Six cases now preserve prior output
      and report `M0002` in JSON on Darwin and Linux.
      MAC `task_4d3c007828e445c5bf748c0b6f6bb44e`.
- [x] **5.0 Stage2 pure module function identity.** I preserve declaration,
      call, alias and shadow ownership instead of flattening same-named
      functions into one scope. My three-level wrapper now executes its root
      shadow on all three compiler paths with inferred and declared names.
      Checker and emitter both resolve function references, including function
      values and aggregate calls; checker-only rewriting would miss children.
      Nine binding methods and the existing driver/emitter suites total 58
      passing methods on each of Darwin and Linux. Both bootstraps pass smoke
      and no-C-seed checks; native binaries still differ. The new Make target
      runs the binding suite against the bootstrapped compiler.
      These sequential tables and merged offsets do not establish reentrancy,
      original-file provenance, private visibility, conflicting-alias policy,
      nominal type identity or foreign-name isolation. The broader gate remains
      open; same-basename tests retain distinct declared introspection names.
      MAC `task_9706ba834a2444a7a613d701ba5eceb8`.
- [x] **5.0 imported module failure propagation.** I reject failed loads
      before publishing bytecode. An in-progress cache marker must not turn
      a duplicate-definition error into successful compilation. I cover
      direct and transitive invalid imports and preserve existing output.
      Twenty compiler cases per platform on Darwin and Linux cover duplicates, wrong
      return types, malformed syntax, missing files and cycles through C-seed
      and bytecode drivers. A completed cached load returns an AST; I no longer
      accept a NULL result merely because a loading marker exists.
      MAC `task_60c31b34d18c4e698194a9dd1ba691c4`.
- [x] **5.0 importer-scoped namespace prerequisite.** I distinguish identical
      import aliases declared by different modules in the C environment and
      its consumers. I establish inferred or declared ownership before processing
      imports and restore the caller on success or failure. Six compiler cases
      exercise reused aliases through C-seed, VM and Stage2 on Darwin and Linux;
      the baseline failed both C-seed and VM cases. All 60 driver-test methods,
      32 environment checks, and the evaluator, typechecker and code-generation
      gates pass on both platforms. Both bootstraps pass smoke and no-C-seed
      checks; native binaries still differ. Type alias visibility is tested separately
      from colliding nominal identity, which remains open. These owner names
      are not canonical path identities; dependency-shadow selection remains open.
      MAC `task_8064e9156d1246ada3673b4014802040`.
- [x] **5.0 packaged interpreter link dependency.** I link wrappers with
      libffi. The generation-retention regression exposed unresolved `ffi_call`
      and type symbols after checked interpreted foreign dispatch gained that
      dependency. The packaged-wrapper regression and wrapper-generation gates
      now pass on Darwin and Linux. MAC `task_309856ab8480437aab712e39001d2dd9`.
- [x] **5.0 transitive shadow execution by default.** I run dependency shadows
      by default across C-seed, VM and self-hosted native compilation,
      exercise direct, transitive and diamond imports exactly once, preserve
      ownership and prior output on failure, and document execution order,
      source-only behavior and side effects. My creator chose this default;
      I retain `--root-shadows-only` as an explicit opt-out. C-seed and VM
      now use this default, as does Stage2 with canonical source paths.
      MAC `task_53197aae0a914dfcaafe490af1a18d3a`.
      - [x] I canonicalize resolved imports before cache lookup, graph
        registration, owner inference and C forward declarations. I reproduced
        duplicate typechecking and mismatched prototypes through equivalent
        paths. Dot segments and symlinks now pass C-seed/VM fixtures, including
        dependencies relative to a symlink target and exactly-once VM shadows.
        My public resolver remains unchanged; Stage2 adopts physical-path
        identity in the later checkpoint below.
        MAC `task_8ae80d436b1342b3b25ed8839f8694fd`.
      - [x] I select dependency shadows by default in my bytecode driver,
        retain `--test-imports` and add `--root-shadows-only`, check and compile
        dependency shadows in owner context, execute a deduplicated graph
        dependency-first and root-last, and test failure publication boundaries
        and production separation. I do not let selected dependencies inherit
        the root's unsafe context. At this checkpoint C-seed and Stage2 parity
        remained required.
        Darwin runs all 103 shadow/cache/language test methods successfully;
        Linux runs the same suite with eight platform-specific skips. Environment,
        typechecker, code-generation and wrapper gates pass on both platforms.
        Both bootstraps pass smoke and no-C-seed checks; native binaries differ.
        MAC `task_e74afc2a7ccb4b16941f5d34f314998d`.
      - [x] I repair previously skipped bootstrap shadows as behavioral tests.
        The first failure compared a `HashMap` with integer zero. I check its
        observable collection behavior, replace filesystem shadows' fixed
        temporary paths and repository-cwd assumptions with private fixtures,
        and keep transitive selection enabled through both bootstraps.
        MAC `task_601070f4e9904943b1e2ca1ebeea5f4f`.
      - [x] I repair the interpreter array failure exposed by default bootstrap
        shadow execution. I reproduce the unsupported-element diagnostic and
        subsequent crash, add a minimal regression, and verify the default
        without suppressing dependency tests.
        MAC `task_fdb1477733a44a9fb9ef1881abf4757c`.
      - [x] I isolate interpreted shadows from production compiler state.
        Checked shadow locals contaminated C lowering; I discard their extra
        metadata before lowering and execute shadows in a child. A parent
        deadline and completion handshake reject crashes, hangs and early
        `exit(0)` without replacing prior output. I retain graph-wide JSON and
        mark interrupted runs incomplete. This is not a security sandbox.
        MAC `task_795f49013a5d4ac294c6e25e0363f7ec`.
      - [x] I repair interpreted record alias lifetimes. Full bootstrap shadows
        reached an invalid free when reassigning `env_cur` in nested transpiler
        calls. Record bindings now copy before replacement, including nested
        records. Alias, self-assignment and callee regressions pass; complete
        aggregate reclamation remains outside this repair.
        MAC `task_1ac1c2fd6bc04b5e81932553901b0d4b`.
      - [x] I repair remaining bootstrap contracts exposed by full shadow
        execution: substring search in array type extraction, interpreted
        static append and dynamic indexed writes, obsolete C-name and boolean
        expectations, invalid union fixtures, and cwd-dependent module-cache
        tests. I verify behavior instead of removing failing assertions.
        MAC `task_ef3a7b277592408988a2b4dbd5642034`.
      - [x] I preserve declared signatures when checking indirect calls in
        shadows. Previously unchecked float function-variable and returned
        function calls were incorrectly typed as integers. I retain valid FFI
        regressions and reject invalid indirect arities and scalar argument
        types. Complete aggregate and higher-order signature checking remains
        separate work. MAC `task_4027a8e02a744178b8f99d83a92395d7`.
      - [x] I apply dependency-shadow selection and the default to C-seed,
        preserving owner context, aggregate shadow JSON and prior output on
        failure. I verify library shadows and bootstrap without bypassing the
        default. Both platforms pass bootstrap smoke and no-C-seed checks;
        native binaries differ, so I do not claim equivalence.
        My 157-method shadow/cache/native-driver suite passes on macOS and
        Linux (eight platform-specific skips on Linux). Evaluator, typechecker,
        environment, parser, transpiler and 63 VM codegen tests pass on both.
        The 101 evaluator tests also pass macOS ASan/UBSan with the changed
        environment, evaluator, parser and typechecker instrumented at O1;
        other linked objects are uninstrumented and leak detection is disabled.
        MAC `task_b331d5925b504d379e21b24d0710d03c`.
      - [x] I require a completion handshake from native shadow entry points.
        A dependency calling `exit(0)` previously passed the native supervisor
        without finishing the selected tests. I reject early exit and exec
        replacement, retain crash/deadline handling, and test the production
        supervisor and driver publication boundary.
        MAC `task_32e7e4b9a93d49dfa4e15d7c83ed1678`.
      - [x] I fail timeout wrappers when their requested command cannot start.
        A `noexec` scratch mount exposed zero exits from failed Perl `exec`
        calls and false bootstrap success messages. I test missing and
        non-executable commands, normal exit propagation and deadlines for
        each Makefile timeout wrapper.
        MAC `task_e0f2c52ae84842f084d49cff3394d7e1`.
      - [x] I cover eight inline Makefile timeout wrappers as well as the four
        shared macros. Their bare Perl exec returned zero when launch failed.
        I extract all twelve programs, expand them through Make, and test
        launch errors and their diagnostics, child statuses and deadlines.
        MAC `task_9b6d05452683ff53cc92d98f94bbd945`.
      - [x] I propagate child failures through enclosing shell recipes:
        `test-nsi-runtime` cleanup preserves a failed compiler status, and
        `shadow-check` stops on an earlier failed iteration or Git selection.
        I test actual Make-expanded recipes with injected failures and success
        controls. All three recipe/timeout tests and nine NSI runtime tests pass.
        MAC `task_e92f56e81dbe4cc88a7c06d18df6f29c`.
      - [x] I repair self-hosted lowering exposed by full bootstrap shadows:
        physical-path extern declarations, explicit imports used by shadows,
        record-array literal and call-argument element types, inferred append
        results losing their input array type, and plain strings incorrectly
        interpolated from their braces. I retain the dependency default and
        verify minimal source regressions before accepting bootstrap.
        MAC `task_c1dcaff808164681b70a093c9ca75113`.
      - [x] I apply the same default, flags and dependency order to Stage2,
        establish canonical source identity in its merger, and verify library
        shadows, native publication and bootstrap. Source-only C emission stays
        non-executing. All 164 shadow/cache/native-driver/supervisor/timeout
        regression methods pass on macOS and Linux, with eight Linux skips.
        Both full bootstraps execute dependency shadows and pass installed
        smoke and no-C-seed checks. Native stage binaries still differ; this
        is not a semantic-equivalence claim. Opt-in `NANO_SHADOW_TRACE` prints
        each generated native shadow target before execution.
        MAC `task_be1b5b951f7a499da6b995bed61871f9`.
- [x] **5.0 self-hosted source preservation through path aliases.** I compare
      device/inode identities before diagnostics or shadows execute, checking
      artifact and diagnostic paths against the root and loaded dependencies.
      I reject relative, symlink and hard-link aliases for native and source-only
      output, and fail closed when existing file identity cannot be checked.
      All 24 alias combinations, syntax-error diagnostics and uncheckable-path
      regressions pass within 44 driver tests on Darwin and Linux. Both
      bootstraps pass with dependency shadows enabled. This checks stable
      filesystem entries, not concurrent hostile namespace replacement.
      MAC `task_507117859a0642f4882b7015296d8af6`.
- [x] **5.0 separate artifact and diagnostic destinations.** I prevent
      diagnostics from replacing a published artifact. I reject equal paths,
      symlinks and hard links, including initially nonexistent equal paths,
      while preserving prior artifacts in a stable filesystem. I do not claim
      atomic publication or protection from concurrent namespace replacement.
      Source identity checks do not cover
      this pair. MAC `task_f38c6358bf944c218f179daf1490ebe2`.
      - [x] I reject existing-file aliases by inode and initially missing files
        with identical destination spellings, before diagnostics or
        shadow execution. Twelve native/C-source collision cases, distinct
        outputs, syntax-error preservation and dangling links pass within 57
        driver/emitter tests on each of Darwin and Linux. Both bootstraps pass
        with dependency shadows enabled; strict C warning checks also pass.
      - [x] I honor filesystem name equivalence for initially missing names.
        An exclusive empty-directory probe at the absent output path lets the
        filesystem resolve the other spelling; I remove it before continuing.
        Four Darwin native/C-source case and Unicode alias cases fail on the
        baseline and pass after repair. Distinct names remain usable on Linux.
        Injected creation, lookup and cleanup failures stop the helper; cleanup
        failure or a killed compiler can leave the empty probe directory.
        Both rebuilt bootstraps and 59 regression methods per platform pass.
        My 18-test CLI gate now includes the probe lifecycle test. This checks
        stable filesystem entries, not concurrent hostile replacement.
        MAC `task_a6419f0b3c764b6d9e3036cf13ae0be8`.
- [x] **5.0 bootstrap runtime dependency invalidation.** I conservatively
      track Nano/C/header/manifest files and directory membership under compiler,
      runtime and library roots, excluding hidden cache contents and non-source
      artifacts. The baseline missed 67 edit/invalidation checks. Eight
      dependency tests now cover all self-hosted stages, additions, deletions,
      cache exclusions and no-op builds on Darwin and Linux. Ordinary bootstraps
      and 30 selected build/CLI tests pass on each platform. A header-only change
      in the Linux repository copy advances all self-hosted stage stamps; the
      following unchanged build preserves them. Darwin's unchanged build also
      preserves its stamps. Directory tests use `/.` to avoid old Make wildcard
      behavior and collisions with phony target names. This is conservative
      modification-time invalidation, not a toolchain/content-addressed key.
      MAC `task_85191b435c95480d9db52e3671d1e740`.
- [x] **5.0 self-hosted string-search builtin lowering.** My shared search
      header supplies first/last byte-offset search to both C emitters and the
      interpreter/VM string primitives. I register last-search in the shared
      frontend and rename the typechecker's private character helper to avoid
      a runtime-name collision. Eighteen assertions cover overlaps, empty and
      missing needles, longer needles and UTF-8 byte offsets through C-seed,
      Stage1, Stage2, separately compiled emitted C, and VM shadows/production.
      Both bootstraps, interpreter/typechecker/string/VM-builtin unit gates and
      55 CLI/bytecode-shadow regressions per platform pass on Darwin and Linux.
      These helpers consume NUL-terminated strings, not embedded-NUL byte spans
      or Unicode character positions. The baseline native output lacked both
      runtime declarations; passing bootstrap helpers alone had missed it.
      MAC `task_a9b152cf5299491694d96a2385527e98`.
- [x] **5.0 interpreted indexed-read alias.** I route `array_get` to the
      same reader as `at`. My FFI map fixture exposed that typing accepted the
      alias while interpreted shadow execution reported an unimplemented
      builtin. Static and dynamic integer-array assertions for both names
      pass, along with indexed reads of mapped floats, on Darwin and Linux.
      MAC `task_a6d2c314d94542a48f11a2bac05af8c2`.
- [x] **5.0 qualified and indirect interpreted FFI.** I route the separate
      `call_function` path through checked foreign dispatch. Before the repair,
      qualified and returned-function calls lost successful foreign results and
      ignored failures escaped shadow reporting. Eight compiler cases per
      platform now cover qualified, returned, variable and `map` callback calls,
      valid floating results, ignored missing symbols, shadow failure locations
      and preservation of prior output. Native API tests check void-call side
      effects, bool/string results, invalid count/storage/type rejection and
      unchanged interpreter bindings. I retain the invoking source call for
      builtin callbacks; host calls without a source node do not invent one.
      Darwin and Linux each pass 97 evaluator tests, 11 FFI tests and 59
      driver/emitter/binding methods. Both bootstrap smoke and no-C-seed checks
      pass; native binaries still differ. Darwin ASan/UBSan also passes all 97
      evaluator tests with production `eval.c` and the test driver instrumented;
      other linked objects, libffi and the native fixture are uninstrumented,
      and leak detection is disabled. This sequential call context does not
      establish reentrancy, complete interpreter error propagation, exact foreign
      library isolation or safety of declarations supplied for native symbols.
      MAC `task_17fe744141024df08c2ef3de7599865a`.
- [x] **5.0 audit defect — borrowed record strings in interpreted shadows.**
      My native-emitter build exposed SIGABRT in `eval_call` cleanup. A field
      read borrowed record-owned string storage that a callee's local then
      released. I copy string field values before returning them. My small
      `nl_shadow_struct_string_lifetime.nano` regression now compiles and runs;
      all 95 evaluator and ten GC-struct tests pass, including repeated reads
      and local mutation. This does not establish general aggregate ownership.
      MAC `task_de992b992c064ceb917bda531312f531`.
- [x] **5.0 native shadows — string builder result handling.** I preserve
      `array_push`'s returned array in `cg_append`. My interpreter converts an
      empty static array to a new dynamic array on its first push, so ignoring
      that result leaves the original accumulator empty. I test generated
      source from an interpreted shadow and from compiled emitters.
      The interpreted emitter shadow and nine native-emitter integration
      tests pass with rebuilt C-seed and Stage 2 tools.
      MAC `task_589709f18d304e6bb828a56583be7755`.
- [x] **5.0 native shadows — separate C test entry.** I add a self-hosted
      transpiler entry that emits selected shadow bodies as independent void
      functions and calls them from a private C test entry. I preserve the
      production entry, allow shadows to call NanoLang main, isolate local
      scopes and test emitted C by compiling and executing it. This is a
      prerequisite, not driver enforcement: selection, shadow typechecking,
      bounded execution and publication ordering remain required below.
      `make test-native-shadow-emitter` passes nine tests, building emitters
      with the C seed and Stage 2 and executing their generated C. It covers
      assertion failures with `NDEBUG`, helper assertions, independent locals,
      selected suffixes and invalid selection, calls to main, and production
      output without the harness. My bootstrap smoke and no-C-seed checks
      pass; the native binaries still differ (2026-09-12).
- [x] **5.0 shadow typing — lexical environments.** My self-hosted checker
      registered non-parameter local lets as globals and shared
      mutable symbol arrays across function and block scopes. I classify lets
      from statement ownership, copy environments at scope boundaries and
      test sibling shadows, branches, parameters and undeclared names. I also
      check assertions inside unsafe blocks instead of skipping them.
      My rebuilt Stage 2 rejects seven malformed-shadow cases before C-source
      publication, including scope leaks with no native compiler available.
      This is bounded lexical coverage, not a claim of full checker soundness.
      MAC `task_a16bb6229928482080df35058afb5a52`.
- [x] **5.0 native shadows — driver enforcement.** I typecheck root shadows,
      compile and supervise their private test executable, and reject errors
      before publishing native output. I test imported helper visibility,
      root/import selection, malformed shadows, output preservation, traps
      and deadlines. Source-only output retains its no-native-toolchain
      contract and explicitly reports that shadows were not executed.
      All nine native-driver tests pass on rebuilt Stage 2, including a
      foreign call cancelling an alarm: my parent still enforces ten seconds.
      My test executable uses private staging and sends shadow stdout to
      stderr. Foreign calls retain user privileges; this is not a sandbox.
      My bootstrap smoke/no-C-seed checks, nine emitter tests, five language
      claim tests, all 28 four-path contract rows and nine CLI tests pass
      (2026-09-12). Native bootstrap binaries still differ.
- [x] **5.0 bytecode shadows — verified test module before publication.** I
      lower root-file shadows as separate zero-argument NanoISA functions and
      execute a private test entry before publishing bytecode or wrappers.
      I verify the test module and bound its execution in a child process.
      Production output retains its original entry and omits the harness.
      I test assertion failure, helper calls, main shadows, output preservation,
      test-only sources, traps and timeouts. Imported-shadow policy remains
      part of the open enforcement item above.
      My ten integration tests pass, including a foreign call cancelling the
      child alarm: the parent still enforces the ten-second deadline.
- [x] **5.0 execution contract — standalone VM exit value.** I propagate a
      successful integer entry result as the standalone `nano_vm` process
      status, matching `nano_virt --run` and native execution. The new main
      shadow regression exposed that standalone execution discarded `7` and
      exited zero. I test 0, 7, -1 and 256 through both execution paths and
      retain trap failures as nonzero;
      daemon-mode result propagation requires separate verification.
- [x] **5.0 array literal typing — validate annotations before stamping.**
      My shared checker overwrites an array literal's inferred element kind
      with a let/set annotation without comparing them. I reject incompatible
      literal elements before assigning metadata, preserve empty-array typing,
      and test both C-seed and VM rejection plus successful matching literals.
      Nested generic identity and non-literal array assignment remain separate.
      MAC `task_4c6ff6e9986a49d6a01701a66b8842d6`.
- [x] **5.0 function signatures — initialized nested return metadata.**
      Checking factory arguments uses an uninitialized temporary signature's
      nested return pointer, producing either a crash or a false mismatch.
      I initialize the record and borrow the declared nested return signature.
      I also initialize parser signature cleanup state before error paths.
      I test matching/mismatched factories and incomplete signatures.
      Repeated nested-factory execution and rejected mismatches/incomplete
      signatures pass. The original function-factory example compiles and
      runs, and the parser-recovery gate passes with parser/lexer ASan/UBSan
      instrumentation (linked support objects are not instrumented).
      MAC `task_4c6ff6e9986a49d6a01701a66b8842d6`.
- [x] **5.0 VM shadow typing — check bodies before lowering.** I run the
      shared statement checker on root shadow bodies before bytecode emission
      so local inference, array element metadata and malformed statements
      receive the same checks as function bodies. I test inferred string/int
      arrays, sibling names and rejected malformed shadows before publication.
      A same-named integer parameter still overrides a shadow-local float
      during lowering; I re-establish checked local metadata at each let and
      retain the executable collision regression.
      This does not complete the shared checker's lexical-scope architecture.
      My expanded bytecode-shadow suite passes, including string/int array
      inference, malformed statements, parameter/local metadata collisions
      and output preservation. The original array-inference example now
      compiles and executes its shadows and production entry.
      MAC `task_4c6ff6e9986a49d6a01701a66b8842d6`.
- [x] **5.0 VM numeric builtins — min/max operand preservation.** Running
      the repaired float example exposes `min(2.5, 7.8)` returning `7.8`.
      Both builtins rotate the stack incorrectly and discard the first
      operand. I preserve both values, compare copies, and select the correct
      original. I test both branches, equality, negative int/float values and
      left-to-right exactly-once evaluation through shadows and product runs.
      MAC `task_4c6ff6e9986a49d6a01701a66b8842d6`.
- [x] **5.0 VM numeric builtins — typed absolute value.** My `abs` lowering
      always emits integer negation. I select float operations for float
      operands and test negative, positive and zero operands through actual
      shadow and product execution, retaining integer boundary semantics.
      Negative/positive/zero int and float cases and integer minimum wrapping
      pass. Min/max assertions cover both orders, equality, negative values
      and observable evaluation order. The original float example executes
      with `abs(-3.14) = 3.14` and `min(2.5, 7.8) = 2.5`.
      MAC `task_4c6ff6e9986a49d6a01701a66b8842d6`.
- [x] **5.0 VM FFI source-path lookup.** I retain the directory of a `.nano`
      import when resolving its shared library, including absolute paths and
      relative nested modules. I test real foreign calls in shadows and
      standalone bytecode, with default and explicit build-cache locations,
      and preserve an existing artifact when the foreign assertion fails.
      All four path/cache combinations fail before the repair and pass after
      it, including compiler-run and standalone execution. My 20 shadow
      tests, 63 codegen tests and 18 FFI unit tests pass (2026-09-12).
      Missing-library builds and general module packaging remain separate.
      MAC `task_4c6ff6e9986a49d6a01701a66b8842d6`.
- [x] **5.0 module dependencies — explicit installation authority.** My
      module builder can invoke system package managers during compilation,
      including when discovering a missing `pkg-config`. Before extending
      automatic library builds to the VM, I make installation opt-in with
      `NANO_ALLOW_PACKAGE_INSTALL=1`. I test missing/available dependencies,
      missing tooling, legacy and registry installation paths, and explicit
      opt-in without invoking a real package manager. I retain warm-cache
      dependency checks and document that this is not a build sandbox.
      My intercepted-command regression fails against the previous builder
      and passes with the repair. Missing/available dependency and recovery
      checks pass through the C seed with a warm cache; rebuilt VM tools and
      all 20 shadow tests pass (2026-09-12). No real package manager is invoked
      by the policy regression; actual installation success is not claimed.
      MAC rejects direct `open` to `completed` closure; I leave the task open
      with implementation evidence rather than fabricate an agent claim.
      MAC `task_d6f0090c3b8345f7a4bd75dbd154202a`.
- [x] **5.0 VM imported C libraries — build before publication.** I build
      imported manifest-backed C libraries before root shadows or bytecode
      publication, using my existing builder without transpiling NanoLang
      imports to C. I reject shared-source and shared-link failures rather
      than returning successful build information, and resolve the manifest's
      library name at VM load time. I test cold/warm builds, transitive imports,
      differing source/library names, failed builds preserving output, and
      recovery. A rejected empty library was then accepted as a cache hit;
      I require a regular nonempty library on both fresh and cached paths.
      This does not finish shell-argument or cache-transaction work.
      All 23 shadow tests, 63 codegen tests, 18 FFI tests, package-installation
      policy and C-seed warm-cache dependency/recovery gates pass. The original
      cold-cache datetime example gets past library loading but fails on an
      opaque signature, tracked below. I have not rerun full example coverage
      in this increment (2026-09-12).
      MAC `task_4c6ff6e9986a49d6a01701a66b8842d6`.
- [x] **5.0 C-library cache — content invalidation and retry.** I treat a
      missing, unreadable or mismatching content-hash record as a cache miss,
      not permission to trust timestamps. I include shared-only C sources
      and their header dependencies. I test same-timestamp source, header,
      manifest and shared-source changes, damaged cache records and failed
      link recovery through actual compilation and execution.
      All nine scenarios fail before the repair and pass afterward: five
      same-timestamp content changes, three hash-record damage cases and a
      failed-link retry. All 28 shadow tests, 63 codegen tests, 18 FFI tests,
      package-installation policy and C-seed dependency/recovery gates pass
      (2026-09-12). Cache transaction and toolchain identity work stays open.
      MAC `task_443e8107d0ff4350999e0d5186a809f1`.
- [x] **5.0 ordinary Clang C retained translation units.** For `.c` sources without
      custom or pkg-config compiler flags, I capture preprocessing into private
      files, compile those files, and bind reuse evidence to their bytes.
      I cover ordinary, multiple and shared-only sources, restored edits,
      diagnostics and capture/compile failure recovery. Other compiler modes
      retain their existing behavior pending a separate supported-mode audit.
      Compiler identification requires a successful version query identifying
      Clang; GCC's implicit PCH selection must not silently change. The broad
      snapshot item remains open for other compilers and configured modes.
      All four new methods pass normally and with ASan/UBSan on the production
      builder and linked support sources. Darwin passes 28 shadows, 53 cache
      tests, four snapshot methods, five wrapper link tests, seven wrapper
      boundary tests, 63 codegen tests, 19 FFI tests and dependency gates.
      GCC 12 `-O3 -Werror` passes the four Linux linker methods and explicitly
      skips the Clang snapshot suite (2026-09-12). I retain existing behavior
      for other compiler modes; this is not full snapshot or release acceptance.
      MAC `task_443e8107d0ff4350999e0d5186a809f1`.
- [x] **5.0 shadow gate C-seed prerequisite.** My shadow gate invokes
      `bin/nanoc_c` for array type-contract checks but does not build it.
      A clean Linux run fails five subcases with `FileNotFoundError`. I add
      the actual compiler prerequisite and verify the gate builds it from
      an environment where it is absent, without requiring an earlier gate.
      The Linux rerun confirms absence, builds the seed through this target,
      and passes all 28 shadow methods plus the remaining integration gates
      (2026-09-12).
      MAC `task_443e8107d0ff4350999e0d5186a809f1`.
- [x] **5.0 ordinary GCC C retained translation units.** I verify GCC's
      implicit PCH selection before extending retained compilation. I use
      preprocessing that exposes PCH use, retain ordinary translation units,
      and withhold snapshot reuse when preprocessing references external PCH
      bytes. I test cold/warm restored edits, PCH appearance/removal, original
      results, diagnostics, failures and recovery on Linux.
      The baseline fails on GCC 12: a newly usable `.gch` leaves the old
      generation cached while fresh compilation returns the PCH's result.
      Plain `-E` misses that selection; `-fpch-preprocess` exposes it.
      All six snapshot methods pass on GCC 12 normally and with ASan/UBSan
      on the production builder and linked support sources. Linux passes
      28 shadows, 45 cache methods (8 platform-specific skips), four Linux
      linker methods, six snapshot methods, five wrapper link tests, seven
      wrapper boundaries, 63 codegen tests, 19 FFI tests and dependency gates.
      Darwin passes its full corresponding gates, with the four ordinary
      snapshot methods and two GCC-specific skips (2026-09-12). Retained PCH
      contents and configured modes remain open; the full snapshot item is
      not complete.
      MAC `task_443e8107d0ff4350999e0d5186a809f1`.
- [x] **5.0 configured C snapshots — common scalar flags.** I extend retained
      translation units to an explicit set of optimization, language-standard,
      debug and warning flags plus simple macro/include tokens. I keep
      preprocessing-only flags out of retained compilation, test restored
      edits and warm reuse under `-Werror`, and preserve unknown flag fragments
      on the original path. I test active platform selection rather than
      letting inactive flags choose the compiler mode. Pkg-config fragments,
      shell quoting and other configured modes remain separate work.
      Three restored-edit placements fail before the repair and pass afterward.
      All 33 scalar spellings create retained input and reuse records on Clang
      21 and GCC 12; six unknown forms keep original compilation. Ten snapshot
      methods pass on GCC, with two GCC-specific skips on Darwin. GCC ASan/UBSan
      covers the nine-method suite and added warning-error method; the stronger
      reuse-record assertion is rerun normally on both platforms. Full Darwin
      and Linux shadow/cache/wrapper/codegen/FFI/dependency gates pass
      (2026-09-12). This does not complete configured-mode or snapshot acceptance.
      MAC `task_443e8107d0ff4350999e0d5186a809f1`.
- [x] **5.0 configured C snapshots — literal flag fragments.** I decode
      shell-literal words, quotes and escapes without evaluating expansions,
      commands or globs. I use that boundary for captured pkg-config flags
      and manifest fragments, preserve preprocessing-only options and flag
      order, and test against actual shell/compiler arguments. I reject
      ambiguous or oversized decoding for snapshot eligibility, preserve the
      original fallback, and verify cold/warm results and package consistency.
      Eleven snapshot methods pass on GCC, including ASan/UBSan; Darwin passes
      nine with two GCC-specific skips. Full Darwin/Linux compiler, VM, cache,
      wrapper and dependency gates pass. A final Darwin framework-backed package
      regression and strengthened phase assertions pass separately (2026-09-12).
      MAC `task_443e8107d0ff4350999e0d5186a809f1`.
- [x] **5.0 assembler-input snapshot evidence.** I test inline assembly that
      reads an external binary through `.incbin`, changes that binary only
      during compilation, and restores its bytes and timestamps. I compare
      cold, warm and independent fresh results on Clang and GCC before choosing
      assembler-input capture. A retained C translation unit alone does not
      establish that those later inputs were retained.
      Against `5259507d`, Clang 21 and GCC 12 both reuse 43 while independent
      fresh compilation returns 42, in local and shared caches. The binary is
      absent from reuse records; its bytes, size and mtime are restored.
      `--assembler --require-consistent` fails on both platforms (2026-09-12).
      This is a reproduced defect, not assembler snapshot acceptance.
      MAC `task_443e8107d0ff4350999e0d5186a809f1`.
- [x] **5.0 Clang retained-assembly trial.** I exercise normal Clang `-S`
      output as a candidate retained input. I compare executed direct and
      retained builds, including nested assembler includes, binary ranges,
      missing inputs and configured flags. I remove original assembler inputs
      before assembling the retained output. This trial must establish the
      pipeline boundary before production integration; it is not that integration.
      Four Clang 21 subcases pass: direct/nested assembler reads with default
      and optimization/debug/strict-warning flags. Retained output executes 42
      after deleting all original inputs; changed direct compilation executes
      43 and missing-input capture fails. GCC 12 keeps `.incbin` in its `-S`
      output and cannot use this boundary unchanged (2026-09-12).
      MAC `task_443e8107d0ff4350999e0d5186a809f1`.
- [x] **5.0 Clang retained-assembly integration.** I capture supported Clang
      builds with `-S`, hash and retain those assembly bytes, and compile them
      without C-only flags. I preserve dependency/diagnostic evidence and test
      restored assembler inputs through the production cache, permanent changes,
      capture failures and existing configured builds before marking this complete.
      Fourteen snapshot methods pass on Darwin Clang 21 (two GCC-specific skips),
      including ASan/UBSan on the production builder and support sources. Full
      Darwin and Linux GCC compiler/VM gates pass; Linux Clang 14 passes snapshot,
      cache and linker tests. Warm validation now performs C code generation;
      I report assembly-capture and object-build counts separately (2026-09-12).
      MAC `task_443e8107d0ff4350999e0d5186a809f1`.
- [x] **5.0 GCC retained-object trial.** I test whether compiler object output
      is reproducible across private output paths with supported flags, includes
      assembler-read bytes, and can be linked after deleting original inputs.
      I compare changed and restored inputs and execution results. I distinguish
      this compiler-output boundary from a source-file snapshot and report the
      full-compilation cost before deciding production integration.
      Four GCC 12 subcases pass, including nested macro/include reads and debug
      flags. Both platforms pass the 15-method snapshot suite with three
      compiler-specific skips each. Median cJSON capture time is 214 ms for
      object generation versus 12 ms for preprocessing in the Linux fixture;
      this is not an incremental compilation speedup (2026-09-12).
      MAC `task_443e8107d0ff4350999e0d5186a809f1`.
- [x] **5.0 GCC compiler-output validation integration.** I bind reuse to the
      actual objects compiled from retained C and validate them against fresh
      private captures. I preserve cold compilation failures and old generations,
      test restored/permanent assembler input changes and cleanup, and report
      the extra compilation work. This closes an output-consistency gap, not
      the separate requirement for complete source-file snapshots.
      Seventeen snapshot methods pass on GCC 12 with two Clang-only skips,
      normally and with production-builder/support ASan/UBSan. Darwin passes
      with five GCC-specific skips. Full compiler/VM gates pass on both
      platforms; compiler-count assertions now include validation work. The
      restored assembler edit yields cold 43 without reuse evidence, then
      warm/fresh 42. Temporary success/failure cleanup and cold failure without
      retry are tested (2026-09-12).
      MAC `task_443e8107d0ff4350999e0d5186a809f1`.
- [x] **5.0 GCC literal assembler-file capture.** I emit assembly from retained
      C, recursively copy literal `.include` and `.incbin` inputs into the
      private generation, and assemble only those copies. I preserve assembler
      offset/count expressions and bind reuse to captured bytes. I test restored
      and permanent edits, nested macro includes, relative and quoted paths,
      capture fallback and cleanup. Unsupported syntax retains object validation;
      this bounded path does not close general assembler-input acceptance.
      Nineteen snapshot methods pass on GCC 12 (two skips) and Apple Clang
      (six skips); GCC ASan/UBSan passes with production builder/support
      instrumentation. Both bootstraps and 111-method cache/bytecode/link
      suites pass, with ten platform skips each. Final allocation/MRI guards
      pass the full snapshot suites again, plus leak-enabled copier boundaries.
      Restored literal binary/include edits now yield cold/warm/fresh 42;
      macro-argument fallback still yields cold 43 without reuse, then 42.
      Dependency-rebuild and FFI gates pass on both hosts (2026-09-13).
      MAC `task_443e8107d0ff4350999e0d5186a809f1`.
- [x] **5.0 assembler read-boundary trial.** I test capture/replay at GNU
      assembler file opens, after macro expansion, without parsing its source
      language. I retain exact input bytes and file-name boundaries, replay
      with originals replaced or absent, and reject unrecorded reads. I record
      the supported executable/host boundary and integration costs before
      selecting a production mechanism. This trial alone is not integration.
      GNU as 2.40 on Linux arm64 passes four macro/path subcases and a
      deterministic repeated-path edit, normally and with UBSan on the capture
      helper. Replay matches object bytes and executes retained 42/4243 after
      originals are deleted; absent copies and unrecorded inputs fail. Names
      containing spaces, quotes, dollar/hash, backslash and newline survive
      length-delimited records. Production snapshots remain unchanged; their
      19-method suite passes on both hosts. Darwin skips the Linux-only trial
      (2026-09-13). MAC `task_443e8107d0ff4350999e0d5186a809f1`.
- [x] **5.0 assembler capture record lifecycle.** I replace the trial's
      append-only records with a versioned, bounded, sealed capture format.
      I preserve failed opens and ordered repeated reads, validate copy contents,
      reject interrupted/truncated captures, and require replay completion.
      I build the helper from runtime sources and exercise it with real GNU as
      before connecting it to the production builder. The integration gate
      below remains open until compilation actually selects the helper.
      Eight Linux methods pass against the built helper and with helper UBSan,
      including real GNU-as macro/path replay, ordered failures and repeated
      reads, killed capture, malformed records, changed/missing/symlink copies,
      FIFO replacement and stale-completion rejection. The original trial and
      19 production snapshot methods still pass on both hosts with their
      platform skips. Private replay storage remains trusted; this is not
      hostile-writer isolation or authenticated evidence (2026-09-13).
      MAC `task_443e8107d0ff4350999e0d5186a809f1`.
- [x] **5.0 daemon socket-path length validation.** I reject a Unix-domain
      socket name that cannot fit before unlink/bind, rather than truncating it.
      Linux GCC 12 `make install` failed its `-Werror` build at the truncating
      copy in `vmd_server_run`. The current UID-derived path is short; three
      injected boundary cases test exact fit and oversized-name rejection,
      preserving existing files, on Darwin and Linux. Linux installation now
      passes (2026-09-13). MAC `task_8babad5374294e02ab0e8147a6d717ed`.
- [x] **5.0 GNU assembler capture/replay integration.** I bind a supported
      assembler executable to an isolated read-capture helper, preserving
      ordered repeated reads and exact path bytes. I package and identify the
      helper, scope loader settings to the assembler child, handle failed reads
      and truncated/partial records, and bind object generation and reuse to
      the completed capture. I test macro/conditional inputs, restoration during
      final assembly, missing inputs, driver/tool selection, cleanup and failure
      recovery. I do not promote the single-threaded trial's trusted records
      or its two stdio hooks into a general file-access guarantee.
      Linux selects read replay after literal capture cannot represent an input,
      for dynamic ELF64 little-endian GNU as 2.40. Macro restoration now yields
      cold/warm/fresh 42 with reuse, at six object assemblies across cold/warm
      capture, replay and validation. Both bootstraps and 121-method regression
      sets pass (ten Linux/twenty Darwin skips). All 21 snapshot methods pass
      with production-builder/support ASan/UBSan on GCC. Expanded checks cover
      ignored helper loads, shared-only and inactive inputs, lookup failure,
      adjacent discovery, cleanup and recovery. Linux install includes the
      helper; other assembler variants and general read coverage remain open
      in the acceptance gate below (2026-09-13).
      MAC `task_443e8107d0ff4350999e0d5186a809f1`.
- [x] **5.0 GNU assembler variant acceptance.** I exercise sealed capture,
      replay and the production cache with a second GNU toolchain, including
      restored macro inputs and failure recovery. I widen the supported version
      set only after those gates pass, retaining rejection of unidentified
      executables and untested versions. I record compiler, assembler and host
      versions; this does not close arbitrary compiler-mode snapshots.
      GCC 13.3 / GNU as 2.42 on Ubuntu 24.04 Linux arm64 passes 22 snapshot
      methods (two skips), eight capture-record and two replay-trial methods,
      including helper UBSan. GCC 12.2 / GNU as 2.40 on Debian bookworm arm64
      passes those same gates. I accept exact first-line 2.40/2.42 tokens;
      lookalike versions and later-line matches are rejected. GCC 13.3 exposed
      a discarded diagnostic `write` result; I handle partial writes/EINTR
      without suppressing warnings. The Linux bootstrap and 122-method
      bytecode/cache/link/snapshot/helper suite pass (ten skips). General
      assembler reads and other variants remain open. Darwin bootstrap and
      22 snapshot methods pass with nine skips (2026-09-13).
      MAC `task_443e8107d0ff4350999e0d5186a809f1`.
- [x] **5.0 GNU assembler read-mode acceptance.** I exercise real include
      search failures, nested includes, repeated macro expansion, binary
      offset/count expressions, listing/debug and dependency-output modes.
      I remove captured originals and introduce earlier search candidates
      before replay, then compare object bytes and auxiliary outputs. I record
      the tested toolchain and keep arbitrary syscall coverage separate.
      All nine capture-record methods pass normally and with helper UBSan on
      GCC 12.2 / GNU as 2.40 (Debian bookworm) and GCC 13.3 / GNU as 2.42
      (Ubuntu 24.04), both Linux arm64. The matrix checks actual repeated
      payload bytes, nonempty listings, and dependency output, not only object
      equality. The 22 production snapshots still pass on GNU as 2.40 with
      two expected skips. No production code changed (2026-09-13).
      MAC `task_443e8107d0ff4350999e0d5186a809f1`.
- [x] **5.0 Clang assembler macro-argument capture.** I verify that retained
      assembly resolves file arguments after macro expansion, including nested
      includes and inactive missing inputs. I remove the original files before
      replay and check the resulting library's payload, rather than inferring
      capture from textual directives. I exercise plain and optimized/debug
      compiler flags; broader assembler variant coverage remains below.
      All six replay cases pass on Apple Clang 21 ARM64: direct 42, changed
      43, and retained 42 after originals are deleted. The production cache
      fixture also preserves cold/warm/fresh 42 and actual generation reuse
      for macro arguments under local and shared caches. All 43 snapshot
      methods pass with eleven platform skips; the expanded cache case passes
      separately after the full run started. No production code changed.
      MAC `task_443e8107d0ff4350999e0d5186a809f1`.
- [x] **5.0 selected external-assembler expansion.** I query the actual
      external assembler invocation and test its assembly-text output after
      macro expansion. I require replay after deleting original files, literal,
      nested and macro inputs and plain/debug flags. I preserve the selected
      backend and its target/assembly arguments; text capture changes output
      format/path and temporary-label naming, with identical object bytes after
      ordinary replay. I integrate private capture and warm validation; I do not substitute a different assembler
      merely because it accepts the fixture.
      The initial selected-backend trial replays plain literal/nested/macro
      inputs with identical object bytes, but debug output emits repeated empty
      symbol names and fails reassembly. I must resolve that backend output
      mode before admitting it to production, not strip debug information.
      I test temporary-label naming only during text capture and require
      reassembled object bytes to equal direct assembly; the backend warns
      that retaining temporary symbols during object assembly can change
      semantics, so that object-output option is not an acceptable shortcut.
      Six trials include all byte values and a temporary-label relocation.
      Production plain/debug restored macros yield 42/42/42 with actual
      local/shared reuse; retained files replay after originals are deleted.
      Both full gates pass 194 methods; the separately added failed-query
      restored-edit control exposes the remaining umbrella defect below.
      MAC `task_443e8107d0ff4350999e0d5186a809f1`.
- [x] **5.0 external-assembler stale cache.** I repair the measured
      `-fno-integrated-as` cold/warm/fresh 43/43/42 mismatch on Apple Clang 21
      under local and shared caches. Both publish a reuse record despite no
      retained assembler inputs. Flag exclusion is not containment. I bind
      final compilation and warm validation to retained inputs while preserving
      external assembler selection, and test literal/nested/macro inputs,
      deletion, restored edits and failure recovery. The existing literal
      copier now serves production literal capture: the restored-edit fixture
      returns 42/42/42 with actual local/shared reuse, private payload bytes,
      and the external selector on all three object compilations. Nested
      literal reads, replacement and failed-build recovery are tested separately.
      Fixed-width octal debug data and selected-backend macro expansion now
      pass their acceptance work. At v31, forcing an admitted external driver's
      dry-run query to fail still compiled uncaptured source and produced
      43/42/42 without reuse under both caches. My v32 context refuses that
      build before object compilation. I distinguish an unsupported mode from
      a failed admitted capture; the latter cannot publish an uncaptured result.
      I keep unsupported source/flag modes distinct from admitted external
      capture. I test refusal before any object compilation, cold-cache absence,
      preservation of a previous generation on warm failure, both cache roots,
      diagnostics and recovery to a newly captured reusable generation.
      All 197 gate methods are covered by passing Darwin runs (fifteen skips)
      and one Linux full-gate run (twenty-three skips). Three focused Darwin
      methods pass ASan/UBSan with leak detection disabled. My user-guide build
      and validation pass; broader assembler-input capture remains open below.
      MAC `task_443e8107d0ff4350999e0d5186a809f1`.
- [x] **5.0 escaped assembler debug data.** I distinguish fixed-width octal
      byte escapes in data-string directives from macro and filename expansion.
      I preserve accepted assembly bytes, reject malformed and ambiguous forms,
      and check direct/replayed output, deleted originals, restored edits,
      optimized/debug nested inputs and actual warm reuse. I do not treat this
      lexical repair as general macro-input capture.
      All 190 full-gate methods pass on ARM64 Darwin and Linux (fifteen and
      sixteen platform/configuration skips). Three capture boundary/replay
      methods also pass ASan/UBSan with leak detection and explicit diagnostic
      checks. I preserve all 256 byte values through each supported directive,
      including numeric data inside a named-parameter macro. Apple external
      `-O2 -g` restored edits now yield 42/42/42 with actual local/shared reuse.
      MAC `task_443e8107d0ff4350999e0d5186a809f1`.
- [x] **5.0 admitted snapshot failure consistency.** I remove the remaining
      path from failed GCC assembly capture to preprocessed C with live
      assembler reads, and from failed ordinary capture to live C. The existing
      missing-helper GCC regression expects cold/warm/fresh 43/42/42. I require
      failure before final object compilation, preservation of a previous
      generation, useful diagnostics, and recovery under both cache roots.
      Successful literal and macro capture must still produce reusable 42/42/42
      results. Unadmitted source/flag compatibility remains separate; refusal
      does not complete broader snapshot support.
      My v33 context and diagnostic reporting pass the GCC/Linux 201-method
      full gate (twenty-four skips), Darwin's 200-method full gate (sixteen
      skips) and the added Linux-only method's Darwin skip. Linux Clang passes
      all 59 existing snapshot methods plus the new plain/debug external-macro
      regression: four local/shared cases yield reusable 42/42/42. Four focused
      Linux methods pass ASan/UBSan with leak detection enabled. My user-guide
      build and validation pass. `docs/SOURCE_SNAPSHOT_EVIDENCE.md` records scope.
      MAC `task_1251873d28334037b9c02b640d3669c6`.
      MAC rejects `open` to `completed`; ledger closure needs an ownership
      repair. The task description records the verified repository evidence.
- [x] **5.0 assembler filename spelling acceptance.** I compare direct compiler
      results with retained builds for spaces, quotes, backslashes and UTF-8 in input
      filenames. I test ordinary Clang, external Clang and GCC where applicable,
      local/shared caches, restored edits, permanent replacement and deletion.
      I distinguish native assembler rejection from capture rejection; a passing
      simple-path fixture does not establish filename fidelity.
      Five spellings pass with native baseline 42, actual reuse, replacement,
      missing-input failure preserving the previous generation, and recovery.
      Forced edits and deletion during final compilation yield 42/42/42 after
      restoration. GCC 12.2 and Clang 14 Linux pass all 62 snapshot methods
      (fifteen and nineteen skips); Apple Clang 21 passes seven targeted
      methods. Final external-selector assertions pass separately on both
      Clang hosts. No production changes were needed. Full evidence is in
      `docs/SOURCE_SNAPSHOT_EVIDENCE.md`; broader snapshot support remains open.
      MAC `task_fa906fc996d843d1bb4cbefc72503f95`.
      MAC rejects completion while unclaimed; the task records the evidence
      and needs a verified agent identity for ledger closure.
- [x] **5.0 assembler include-search capture.** I carry literal assembler
      include-search arguments through the correct compiler phases and retain
      the selected input bytes. I measure `-Wa` and `-Xassembler` spellings,
      ordered search directories, metadata/platform/package flags, restored
      edits and new earlier candidates, failure/recovery and actual reuse on
      Clang and GCC. Low-level helper replay alone does not establish production
      support when the module flag classifier declines these arguments.
      Apple Clang 21 (ordinary/external) and GCC 12.2 both reproduce
      cold/warm/fresh 43/43/42 with actual stale reuse under both cache roots.
      My v34 phase selection repairs the admitted flag fragments to 42/42/42
      with actual reuse, preserves native C-header precedence, and keeps
      strict-warning link jobs free of assembler options. The linker grammar
      query uses a private C input instead of a trailing language reset.
      Darwin and GCC/Linux full gates each pass 206 methods (seventeen and
      twenty-four skips); Linux Clang passes all 65 snapshot methods with
      nineteen skips. Two leak-enabled ASan/UBSan methods pass. The user guide
      builds and validates; details and limits are in
      `docs/SOURCE_SNAPSHOT_EVIDENCE.md`.
      MAC `task_92e6817607cf4071ab614289911a9a41`.
      MAC rejects completion while unclaimed; a verified agent identity is
      still needed for ledger closure.
- [x] **5.0 paired compiler flag fragments.** I make adjacent metadata flag
      fragments preserve the same operand pairing and capture semantics as
      their joined spelling, independently of transport-size thresholds.
      I measure restored assembler inputs first, then test invocation-owned
      normalization, C/assembler/linker operands, split/joined argv, malformed
      forms, failure/recovery and local/shared reuse before documenting support.
      I reproduced Apple Clang 21 rejecting split assembler include arguments:
      metadata loading rewrote bare `-I` as an empty relative C include path.
      My v35 operand tracking and owned-fragment normalization repair this;
      restored inputs now yield 42/42/42 with actual reuse.
      An unclassified fragment must not disable existing include fallback for
      later literal flags; I test that compatibility boundary explicitly.
      Darwin and GCC/Linux full gates each pass 211 methods (seventeen and
      twenty-four skips); Linux Clang passes all 70 snapshot methods with
      nineteen skips. Four leak-enabled sanitizer methods and an isolated
      twelve-case Darwin rerun pass. I record an initial reader timeout without
      claiming its cause in `docs/SOURCE_SNAPSHOT_EVIDENCE.md`. The updated
      guide builds and validates. Broader assembler-input capture remains open.
      MAC `task_8500a5ea65584cc3a5cfb02a7812b16b`.
      - [x] I reconcile PR #287's snapshot with my integration history,
        verify its exact tree identity and rerun paired-fragment normalization,
        operand ownership, restored-input reuse and phase/recovery tests before
        retaining its actual head as a merge ancestor.
        Its entire tree matches ancestor `3fb98917`; four focused methods
        pass in 121.173 seconds. I retain newer source unchanged. Evidence:
        `docs/evidence/pr-287-reconciliation.md`.
      MAC rejects completion while unclaimed; a verified agent identity is
      still needed for ledger closure.
- [x] **5.0 GNU alternate-macro production capture.** I first measure restored
      assembler input changes with `--alternate`, using source that requires
      its macro grammar. I admit its literal `-Wa` and `-Xassembler` spellings
      through phase-specific capture, including paired metadata fragments and
      include-search combinations. I verify native semantics, cold/warm/fresh
      results, actual reuse, and failure recovery on supported GNU assemblers
      before accepting the change. I retain explicit limits for other backends.
      MAC `task_8c4127e1aeea4325acda9bca51eacf76`.
      The baseline is 43/43/42 with stale reuse; the repair produces 42/42/42
      with retained reads and reuse on GCC 12/GNU as 2.40, Clang 14/external
      GNU as 2.40, and GCC 13/GNU as 2.42. The 214-method Linux gate, full
      73-method Clang snapshot suite, Darwin include-path matrices and three
      leak-enabled sanitizer methods pass. I record exact scope and timings
      in `docs/SOURCE_SNAPSHOT_EVIDENCE.md`. Arbitrary assembler modes remain
      open below.
      MAC rejects completion while unclaimed; ledger closure still needs a
      verified agent identity.
- [x] **5.0 absolute native-source cache paths.** I use the same absolute or
      module-relative source path for compilation and cache hashing. My new
      mixed-source recovery test finds zero hashes and perpetual rebuilds for
      absolute sources because recording and validation prepend the module
      directory. I verify unchanged reuse and edits for ordinary/shared paths,
      and reject path-format overflow instead of hashing truncated names.
      MAC `task_61fc127a19be474ea666c678c45f4780`.
      Both cache paths now use a bounded resolver and reject failed source
      hashes. Absolute ordinary/shared assembler reuse and hash-path
      equivalence, edits and overflow checks pass on GCC and external Clang.
      MAC rejects completion while unclaimed; a verified agent identity is
      still needed for ledger closure.
- [x] **5.0 Apple `.s` source-kind semantics.** Apple Clang 21's native driver
      query selects assembler-with-cpp preprocessing for lowercase `.s` too.
      My new capture classifies it as raw solely by suffix. I reproduce this
      with preprocessing-required `.s`, then preserve selected-driver default
      semantics and explicitly disable preprocessing for retained text replay.
      I verify integrated and external paths before claiming source parity.
      MAC `task_75da2d51a32c4e10a4d9ef6832abc96f`.
      The external path now preserves native preprocessing with cache context
      v38; eight ordinary/shared, local/shared-cache replacement/deletion cases
      return 42/42/42 with reuse. Native integrated reports and the standalone
      text trial confirm the same default; production integrated admission
      remains unfinished below.
      MAC ledger closure is unresolved: an external worker marked the task
      failed after receiving an unrelated checkout. My local implementation
      and validation evidence are recorded; I did not override that history.
- [x] **5.0 standalone assembler debug parity.** I preserve native debug
      sections and source provenance through standalone-source capture/replay.
      Selected integrated text expansion reproduces ordinary objects, but
      data-only `-g` replay fails with undefined section-end labels on Apple
      Clang 21 and Debian Clang 14. I also audit debug-option ownership when
      copying retained assembler bypasses C lowering. I do not equate payload
      equality or omission of debug flags with native debug parity.
      MAC `task_da2ca60a6acf481ab6d9e3f2fd276a31`.
      At `8f9069c3`, native GCC 12 and Apple Clang 21 external `.s`/`.S`
      objects have debug sections and source provenance; production unit
      objects have neither, despite returning 42 and reusing generations.
      `tests/characterize_assembler_debug.py --require-debug` reproduces this
      under both cache roots. Adding a `nop` to the integrated text trial
      does not resolve Apple's undefined section-end label. The defect is
      therefore not confined to an empty text section in that tested backend.
      A candidate now reproduces byte-identical native objects and decoded
      debug data for simple `.s`/`.S` units on GCC 12 and Apple Clang 21
      external assembly, even with the original unit temporarily deleted.
      It retains the original basename, remaps private directories through
      the selected assembler, and supplies debug flags during final assembly.
      A flat-path alternative with separate directory/basename mappings fails
      GNU raw `.s` provenance in both cache roots; preprocessed sources pass.
      I retain that counterexample as `--candidate --flat`. Integration must
      respect my flat-only publication barrier and descriptor-relative cleanup:
      adding persistent nested directories without changing those contracts
      is not a valid implementation.
      Production context v39 now gives standalone units their debug selectors
      during assembly and GNU read capture, preserving selector order and
      excluding them from final assembly of C-generated text. Directory maps
      make private-object validation stable. This is a partial repair:
      original raw-source basenames, Darwin source-directory alias spelling,
      nested-source provenance and complete integration remain open. Debug
      mapping also requires representable map-source paths (no `=`). I retain
      strict characterization failures instead of accepting section counts
      as complete provenance evidence.
      The new debug macro-read regression found Darwin warm rebuilds in all
      four suffix/cache cases. Cold/warm debug diffs exposed cwd-relative
      private filenames after expansion. Explicit relative-directory mapping
      restores reuse in that regression; it does not restore original line
      locations or basenames.
      I now integrate per-unit original-basename aliases: copy retained bytes
      through descriptor-relative regular-file checks, use the same alias for
      capture/replay, and remove private alias directories before publication.
      I test same-basename units, failed builds, substituted symlinks and warm
      validation without weakening the flat-generation publication barrier.
      Context v40 implements those aliases. Simple GNU `.s` and `.S` objects
      now match native bytes and decoded debug data in both cache roots, with
      reuse and no published alias directories. Darwin basename, debug-section
      and reuse checks pass; the initial lexical-path control differs from my
      physical-root policy, audited below. Macro-expanded source locations
      still need work. I keep this parent open.
      I audit the Darwin spelling comparison against my existing physical-root
      policy before changing metadata: native controls must identify which
      source argument they used. I retain the lexical comparison and add a
      physical-source control plus alias-to-physical warm-reuse checks.
      Policy-matched simple `.s`/`.S` controls now establish byte-identical
      native objects on both hosts, without changing production metadata or
      normalizing source paths in debug dumps. The physical control still
      rejects Darwin macro expansion: native line 5 becomes expanded line 10,
      with an added checksum for expanded text. That is genuine provenance
      loss, distinct from the import-alias policy.
      Before adding a source-location rewriter, I compare native object capture
      from retained pre-expansion units against the physical native control.
      I test debug data, whole-object bytes and execution after unit inputs are
      removed. Any integration must still satisfy capture timing, dependency
      observation, validation and failure-recovery contracts; object equality
      alone does not establish those guarantees.
      The native-object candidate now matches physical native object bytes
      and debug data on both hosts for `.s`/`.S`, both cache roots, and a
      nested macro/binary include. Linking and execution return 42 after the
      original unit, retained alias, macro include and binary are removed.
      I must integrate this serialization boundary without dropping retained
      input identity, restored-input checks, dependency observation or cache
      recovery. The candidate is not a production repair.
      I integrate native unit capture before changing final object use: retain
      the pre-expansion identity and expanded observation, fingerprint the
      selected native object, and verify exact debug/object identity and warm
      validation. Final object transport and post-capture mutation tests remain
      a separate required step; retaining an unused object is not the repair.
      I also correct shared-unit debug mapping under Darwin's lexical `/var`
      alias: macro expansion can retain the private staging directory in the
      final object's line table and defeat warm reuse. I compare actual line
      tables and require recovery followed by reuse for both source groups
      and cache roots before accepting native capture.
      Context v41 now retains native objects with exact physical-control
      debug/object identity across eight Darwin suffix/cache/macro cases.
      I fixed the overlapping-map priority and verified ordinary/shared
      capture-failure recovery, staging cleanup and subsequent warm reuse.
      Six focused Darwin methods and the full 227-method Linux target pass;
      three instrumented Linux lifecycle methods report no sanitizer errors.
      Final objects still come from expanded text, so this parent stays open.
      I next route final native unit objects through a bounded, no-follow
      copy in both fresh builds and validation, without a dummy assembler
      invocation. I require physical native debug/object equality, explicit
      post-capture input mutation evidence, and failed-copy recovery before
      accepting this transport.
      Context v42 now uses captured native objects for selected Apple external
      macro units, with the same executor in fresh builds and warm validation.
      Production debug data and whole-object bytes match all eight physical
      native controls. Eight post-capture mutation cases keep changed binary
      and missing macro inputs through linking and still return 42, then reuse
      the generation. Failed-copy recovery and single-source units also pass.
      Integrated admission and broader provenance coverage remain open; this
      does not establish atomic input observations or an input-read inventory.
      I next admit integrated Clang standalone units through selected native
      capture, preserving platform preprocessing and C siblings. I compare
      complete native objects/debug data, test post-capture nested mutations
      and recovery, and verify both Apple Clang and Linux Clang before treating
      integrated support as complete. Driver-report parsing remains bounded
      and literal; unsupported reports must fail capture without live fallback.
      Initial Linux Clang 14 raw `.s` capture retained private paths despite
      driver debug maps, breaking object equality and warm reuse; `.S` passed.
      I must preserve native raw-source provenance without enabling C
      preprocessing or weakening the equality/reuse checks.
      Mixed integrated modules also expose a C-sibling regression: retained
      C lowering drops assembler include-search flags that native compilation
      uses. I preserve those flags in the retained integrated lowering phase
      and test the native control, runtime value and warm reuse on both hosts.
      Context v43 now admits integrated units on tested Apple Clang 21 and
      Debian Clang 14. Linux retained stdin plus the original logical filename
      restores raw-source identity without preprocessing or line rewriting.
      Integrated retained-C lowering preserves assembler search flags. Native
      object/debug, post-capture mutation and capture/copy/report-recovery
      checks pass; wider source-provenance coverage remains open.
      I next extend native debug controls to instruction-emitting nested
      macros and explicit `.file`/`.loc` directives, including quoted and
      Unicode source names, before closing the provenance parent.
      I measure these controls without normalizing source paths or line rows,
      across `.s`/`.S`, both cache roots, integrated Clang and external
      assembly. Explicit-location fixtures must name their logical debug
      source rather than mistaking its absence from the physical source for
      missing debug information.
      The expanded 48-case matrix passes on each host, including both quote
      forms, implicit locations and explicit logical source locations. I
      retain complete native-object/debug equality and warm reuse. Context
      v44 closes the remaining `=`-path requirement below through selected
      Clang stdin capture and GNU parent-held descriptor transport. Ten final
      Darwin methods and the complete 245-method Linux target pass, with
      separate sanitizer and unprivileged checks recorded in
      `docs/SOURCE_SNAPSHOT_EVIDENCE.md`. I close this debug-provenance gate,
      not the broader input-inventory or standalone flag/include acceptance.
- [x] **5.0 assembler provenance measurement — native zero addresses.** I
      recognize `readelf`'s bare `0` address in explicit source-line rows,
      while rejecting wrong filenames and line numbers. I test both quote
      forms in Unicode source basenames and make strict characterization
      require the requested location as well as native object identity.
      Three focused Darwin methods pass; the complete Linux target passes
      238 methods with 29 platform skips. I record escaped native Unicode
      spellings separately, without normalizing debug data.
- [x] **5.0 assembler debug maps — equals-containing source paths.** I give
      retained inputs representable names when source or cache paths contain
      `=`, preserving native include lookup, physical-root debug provenance,
      native object identity, warm reuse and failure cleanup. Native Apple
      Clang 21, Debian Clang 14 and GNU assembler 2.40 all split debug maps at
      the first `=`; simply removing my guard produces the wrong directory.
      MAC `task_da2ca60a6acf481ab6d9e3f2fd276a31` retains this requirement.
      I first test selected Clang stdin transport with retained preprocessing
      and the original logical filename on both hosts. Darwin does not allow
      basename access below `/dev/fd/<directory-fd>`, so that alias is not a
      portable solution. Any stdin integration must retain the selected
      backend, debug selectors, include lookup, supervised deadline and
      failure recovery; GNU assembler remains a separate required path.
      GNU stdin controls retain `{standard input}`, not the native filename.
      I instead test a no-follow retained directory on child stdin, addressed
      as `/proc/self/fd/0/<original-basename>`, with an unambiguous directory
      map. Capture and replay must name the same primary input and validate
      the same sealed read manifest, without changing the compiler cwd.
      The descriptor-on-stdin trial passes direct compiler controls but
      prevents Python compiler wrappers from starting. I reject that transport
      and instead hold the descriptor in the supervising parent, naming it
      through `/proc/<parent-pid>/fd/<fd>/<basename>`. Wrapper subprocesses
      must work with ordinary stdin and may close inherited descriptors.
      Context v44 passes 72 native object/debug cases on Darwin and 108 on
      Linux, plus relative-read, restored-input, failure/recovery, warm-reuse,
      descriptor-boundary and unprivileged controls. Logical first-input
      identity remains checked separately from transient descriptor spelling.
- [x] **5.0 compiler identity — equals inside executable paths.** My simple
      compiler-token classifier rejects absolute wrapper paths containing
      `=`, silently declining snapshot admission in the new lifecycle tests.
      I admit `=` after a directory separator while continuing to reject
      assignment prefixes and shell syntax. I verify compiler identity,
      captured-input timing and cache recovery with wrappers in these paths.
      MAC `task_da2ca60a6acf481ab6d9e3f2fd276a31` records the reproducer.
      Admission then exposes `/usr/bin/env` treating the executable pathname
      as another assignment. I pass the capture marker in the spawned
      process environment and execute the literal compiler argv directly.
      The classifier, child-local environment, restored-input and post-capture
      regressions pass. I do not admit assignment prefixes or shell syntax.
- [x] **5.0 build evidence diagnostics.** I expose opt-in, stderr-only
      validation decisions without changing query order or running extra
      compiler observations. I distinguish missing reuse records from missing
      captured inputs, retain diagnostics on regression failures, and test
      controlled post-capture failures, deadline labels and trace-independent
      cache reuse. MAC `task_3f96ba3373db49a6b1c2a1987c1c0349`.
      Darwin and Linux GCC/Clang controls pass; the complete Linux gate passes
      251 methods with 29 platform skips. The recovery-stability item stays open.
- [x] **5.0 mixed-unit post-link deadline controls.** I inject a deadline
      into the selected standalone assembler query only after shared linking.
      I check both source suffixes, source groups and cache roots: captured
      output remains usable, reuse is withheld, the supervised child is reaped,
      private validation files are removed, and an unblocked invocation
      recovers. MAC `task_3f96ba3373db49a6b1c2a1987c1c0349`.
      All 16 Darwin and eight Linux integrated-Clang deadline cases pass;
      existing C-only controls also pass. I do not change production behavior.
- [x] **5.0 selected assembler hash bounds.** I reject non-regular selected
      assembler inputs without blocking, and check the shared capture deadline
      while hashing regular tool bytes. I test FIFO/device rejection, regular
      and symlinked tools, expired deadlines and unchanged successful hashes.
      Blocking `fopen` in tool hashing currently escapes child supervision.
      MAC `task_3f96ba3373db49a6b1c2a1987c1c0349` records this audit finding.
      Darwin and Linux boundary, admitted-FIFO recovery and existing deadline
      controls pass. Successful hashes retain their representation; regular
      file I/O still depends on the host filesystem returning. The historical
      recovery cause remains separate and unresolved.
- [x] **5.0 cache-record write diagnostics.** I report whether writing and
      renaming the reuse record succeeded, rather than discarding that result.
      I inject a record-rename failure and verify retained output, no reuse
      record, private-file cleanup and subsequent recovery without changing
      publication policy. MAC `task_3f96ba3373db49a6b1c2a1987c1c0349`.
      Darwin and Linux controls pass with tracing on/off and both cache roots.
      Record persistence is now distinguishable from prerequisite validation;
      I do not assign it as the cause of the historical recovery failures.
- [x] **5.0 tool-supervisor timing milestones.** I distinguish spawn return,
      first output, pipe closure and child reaping in opt-in timing evidence.
      I test delayed output and delayed exit without changing the shared
      deadline, output acceptance or cleanup policy.
      MAC `task_3f96ba3373db49a6b1c2a1987c1c0349`.
- [x] **5.0 tool-supervisor final deadline check.** I reject results observed
      after the shared deadline, including when EOF and exit arrive in the
      last polling iteration. I test that boundary deterministically with
      tracing enabled and disabled, retaining process-group cleanup.
      The full Darwin trace accepts a 47 ms query with 45 ms remaining.
      I now read the clock after completion and reject expired or unavailable
      time. All four injected completion-clock cases fail before the fix and
      pass afterward; five targeted supervisor/capture methods pass on Darwin.
      MAC `task_3f96ba3373db49a6b1c2a1987c1c0349`.
- [x] **5.0 early build failure evidence.** My full snapshot gate reports a
      warm build returning 1 with empty stderr despite enabled tracing. I
      distinguish metadata loading, invocation capture, cache setup, locking
      and staging failures before reuse/capture evidence begins. I test the
      diagnostics without changing rejection, publication or quiet-mode policy.
      Eight failure paths pass with both cache roots and tracing on/off,
      followed by successful build and warm reuse. The historical silent
      failure still has no assigned cause; these diagnostics identify a
      recurrence's phase, not its retrospective cause.
      MAC `task_3f96ba3373db49a6b1c2a1987c1c0349`.
- [x] **5.0 capture deadline policy.** I review the fixed five-second
      per-unit budget against measured discovery, tool hashing and native
      capture work. I define the supported host-latency policy and its bounds
      before changing defaults or adding configuration. Any implementation
      retains one shared deadline, fail-closed capture, descendant cleanup and
      actual reuse assertions; repeated retries do not establish stability.
      I select a 30-second default and a decimal 1..300000 ms host override
      for the existing Clang per-unit and Linux per-command capture scopes.
      I document their limits and acceptance requirements in
      `docs/SOURCE_SNAPSHOT_EVIDENCE.md`; this review changes no runtime default.
      MAC `task_3f96ba3373db49a6b1c2a1987c1c0349`.
- [x] **5.0 bounded capture deadline configuration.** I implement the reviewed
      `NANO_CAPTURE_TIMEOUT_MS` policy without resetting a scope's deadline.
      I test default and boundary values, malformed and overflowing input,
      unavailable clocks, controlled expiry and descendant cleanup. I verify
      capture lasting more than five seconds under the default, then actual
      warm generation reuse. Deliberate hang tests select short budgets;
      linker discovery and shadow deadlines remain unchanged. I retain all
      correctness and publication assertions and document tested host coverage.
      I wire the bounded parser and default into both production scopes.
      Darwin parser cases, a six-second supervised capture, short-budget
      expiry and final-completion clock controls pass. Configuration clock
      errors and absolute-deadline overflow reject before spawn; a pipe-holding
      descendant does not survive expiry in the targeted Darwin control.
      Six-second production queries pass with cold publication and two actual
      warm reuses, including a changed timeout without generation invalidation.
      On Debian bookworm aarch64, seven GCC-backed timeout, captured-read and
      recovery methods pass; Clang 14 slow production capture/reuse and final
      completion-clock methods pass separately. I close this bounded policy
      item, not the full Darwin recovery gate below.
      MAC `task_3f96ba3373db49a6b1c2a1987c1c0349`.
- [x] **5.0 external-query timeout fixture budget.** My full Darwin gate at
      `c9dccb0d` reports an error in the deliberate external-query timeout
      case. Its wrapper sleeps 60 seconds, but its parent allows only 20
      seconds against the new 30-second capture default. I select an explicit
      short capture budget for this fault matrix and audit other deliberate
      hangs for the same mismatch. I retain rejection, cleanup, publication
      preservation and actual recovery/reuse assertions, then rerun the matrix.
      I preserve the full run's source until it finishes: 124 methods in
      5634.060 seconds, with only these two errors and nineteen platform skips.
      Both fault invocations now select 5,000 ms; the complete six-failure,
      two-cache-root matrix passes in 119.437 seconds with all assertions intact.
      MAC `task_3f96ba3373db49a6b1c2a1987c1c0349`.
- [x] **5.0 recovery parent timeout evidence and allowance.** The corrected
      full gate at `80f0cc5e` times out at the Python parent's 20-second limit
      during shared `.S` recovery, before one supported 30-second capture scope
      can finish. I retain partial stdout/stderr on parent timeout and test that
      evidence path. I give the search-recovery build helper a finite allowance
      for its multiple capture passes without changing correctness, cleanup,
      publication or reuse assertions, then verify the exact failed case.
      Partial-evidence regression and the exact shared `.S` recovery case pass
      (0.024 and 24.244 seconds respectively). I select a 240-second parent
      guard for successful search-recovery builds, leaving production deadlines
      unchanged. A controlled shared `.S` recovery delays one real capture
      query by 21 seconds, verifies accepted query timing, publication and
      runtime result, then actual generation reuse. This control and timeout
      evidence regression pass together in 44.321 seconds. The earlier replay's
      total time alone was not evidence of one build exceeding twenty seconds.
      MAC `task_3f96ba3373db49a6b1c2a1987c1c0349`.
- [ ] **5.0 Darwin capture recovery stability — non-blocking historical incident.** During concurrent v45 checks,
      an external `.S` split/platform-flag recovery publishes without
      `source_hashes.json`, and integrated shared-unit report-failure recovery
      rebuilds instead of reusing its generation. I isolate both observations,
      establish their causes and verify stable capture/reuse without relaxing
      the assertions. MAC `task_3f96ba3373db49a6b1c2a1987c1c0349`.
      Following Jordan's explicit approval, I retain this incident for new
      evidence but remove it as an implementation prerequisite. Infrastructure
      can fail intermittently; missing historical diagnostics do not justify
      endless reruns or blocking unrelated work after corrected gates pass.
      This is a sequencing decision, not a finding that infrastructure caused
      every unexplained failure. Reproducible correctness defects still block.
      I use the evidence diagnostics above to interpret subsequent stress runs.
      The controlled deadline mechanism above does not assign a cause to the
      unobserved historical events.
      I require a reuse record immediately after report-failure recovery,
      before a warm invocation can overwrite its publication diagnostics.
      Search-order generation comparisons retain the relevant build trace.
      A full Darwin run at `57c81127` stops after 64 methods with failed warm
      reuse in integrated `.S` debug identity (shared cache, no macros).
      Native-object/debug comparisons pass. I retain cold/warm characterizer
      diagnostics and record presence to distinguish publication from reuse.
      The continuation batch at `ce7a7d96` stops after 36 methods: shared raw
      recovery (paired flags, common placement, external assembler, shared
      cache) reports `tool-deadline` during replacement capture. I record each
      traced tool's remaining budget and elapsed milliseconds to distinguish
      a slow child from budget already spent in earlier capture work.
      The next batch at `b002d077` stops after ten methods: standalone raw
      recovery (joined flags, package placement, integrated assembler, local
      cache) records a successful query taking 4,293 ms of its 4,599 ms entry
      budget, followed by `tool-hash-deadline`. I must distinguish query
      execution from supervisor waiting before selecting a bounded remedy.
      Four milestone-instrumented exact-case replays pass in 98.532 seconds.
      Their slower queries spend most time before output; they do not reproduce
      the four-second outlier or establish its cause.
      The full run at `67f635be` stops after thirteen methods on Apple native
      `.S` debug warm reuse (nested reads, local cache). A silent query reaches
      its five-second deadline after immediate spawn, without observed EOF or
      exit. I retain full case diagnostics in the remaining debug-reuse
      assertions before further investigation.
      Direct controls outside my supervisor reproduce a ten-second timeout in
      `/usr/bin/as -###`. The resolved Xcode `as` is a zsh launcher; a sampled
      slow invocation completes after 3.626 seconds. I distinguish this
      independently reproduced tool/host latency from my supervisor behavior
      before selecting a remedy; no host toolchain files are changed.
      A later slow resolved query exposes `zsh` with a `dirname` child and
      finishes in 1.901 seconds. This demonstrates launcher latency before
      compiler invocation; the underlying host wait is not yet identified.
      The non-fail-fast run at `e37737d0` completes all 116 methods in
      5752.886 seconds with three failures and eighteen skips: two replacement
      captures fail closed, while a warm validation timeout forces a successful
      recorded rebuild. Full traces distinguish these paths; stability is
      still unverified, not established by the other passing methods.
      After the final-clock fix at `90fa9927`, the three exact recovery cases
      pass in 73.773 seconds. A successful query still takes 1,655 ms before
      completion. Separate controls measure selected Clang hashing at a
      174.9 ms median and again observe delay in the Xcode assembler launcher;
      these replays do not replace the failed full gate.
      The full run at `510d10f9` completes 117 methods in 5865.459 seconds:
      four failures and eighteen skips. Two warm validations exceed their
      budgets and rebuild successfully with records; one cold capture times
      out. One warm build exits 1 with empty stderr and needs the early-failure
      evidence above. The final-clock regression and both deadline matrices
      pass; neither the full gate nor recovery stability is complete.
      With early diagnostics at `f0835380`, ten exact silent-failure replays
      pass in 102.371 seconds, followed by fifty in 553.547 seconds. The silent
      exit does not recur; its cause remains unknown. I stop this bounded
      experiment and review the recurring measured deadline failures above.
      With the reviewed timeout policy, the full gate at `c9dccb0d` completes
      124 methods without a recovery assertion failure; its two errors are the
      deliberate fixture-budget mismatch fixed above. That matrix then passes
      separately. This is not a green full run at the corrected test state,
      nor an explanation for the historical silent exit.
      The corrected-tree full gate at `80f0cc5e` reports an error in shared
      preprocessed search recovery: split `-Xassembler`, package flags,
      external assembler and local cache. The run continues; its final
      traceback is pending, so I do not assign this error to a deadline or
      publication path. The repaired deliberate-query matrix passes in this run.
      This run finishes 124 methods in 5832.298 seconds with one error and
      nineteen skips. The traceback identifies the 20-second Python parent
      timeout during missing-input recovery, not a failed generation comparison.
      Partial tool diagnostics were omitted by the default traceback; the
      evidence and parent-allowance item above addresses this gap.
      The corrected full gate at `cdb0d9ff` completes 125 methods in
      5659.990 seconds with nineteen platform-specific skips (exit 0).
      Source and probe remain unchanged throughout the run. All four
      ordinary/shared raw/preprocessed recovery matrices and the controlled
      21-second recovery pass. This establishes a green corrected-tree gate,
      not a cause for the historical silent exit or underlying host waits;
      the cause-establishment requirement remains open.
- [x] **5.0 assembler provenance fixture — instruction/path separation.** My
      full Linux gate finds a random temporary path containing `nop` rewritten
      to `INSTRUCTION` by fixture substitution. I construct instruction text
      separately from file paths and exercise a deterministic `nop` directory.
      MAC `task_3f96ba3373db49a6b1c2a1987c1c0349` records this test defect.
      The full Linux gate and both Darwin instruction/location methods pass
      with `nop` present in every characterizer directory name.
- [x] **5.0 mixed-unit integrated preprocessing search parity.** My standalone
      search matrix finds Darwin native 142 versus retained 42: preprocessing
      the C sibling drops integrated assembler include paths before retained
      compilation can use them. I preserve the native frontend search order
      during capture and verify `.s`/`.S`, both source groups and cache roots.
      MAC `task_3f96ba3373db49a6b1c2a1987c1c0349` tracks this defect.
      Context v45 preserves the native C frontend search order before retaining
      preprocessed input. Native-control matrices pass for all four standalone
      suffix/source-group combinations on Darwin and Linux, including both
      cache roots and common/platform/package flag placement. The independent
      recovery-stability concern remains open above.
- [x] **5.0 assembler translation-unit snapshots.** I first characterize
      mixed C/`.s` and C/`.S` modules under restored assembler-input edits.
      I retain raw and preprocessed assembler translation units without
      disabling capture for their C siblings, preserving native source/flag
      semantics. I verify cold/warm/fresh results, actual reuse, dependency
      changes, missing-input recovery and both cache roots on Darwin and Linux.
      MAC `task_3f96ba3373db49a6b1c2a1987c1c0349`.
      At `8724288c`, Darwin `.s` and `.S` and Linux `.S` reproduce 43/43/42
      with actual stale reuse under both cache roots. Linux raw `.s` instead
      fails publication: GCC emits an object but no required depfile. All
      native mixed-source controls return 42. I must supply source-appropriate
      dependency evidence, not merely remove the `.c` admission check. Baseline
      commands and measured limits are in `docs/SOURCE_SNAPSHOT_EVIDENCE.md`.
      GCC and explicit external-Clang paths now retain raw/preprocessed
      assembler units and their C siblings. Restored replacement/deletion,
      ordinary/shared units, macro recovery and absolute-source cache paths
      pass on Linux GCC 12/Clang 14 and Darwin Apple Clang 21. Context v43
      also admits source-kind-specific integrated Clang capture, including
      retained C-sibling assembler search flags. The standalone flag/include
      matrices now pass on the tested GCC 12 and Clang 14/21 toolchains.
      The corrected full Darwin gate at `cdb0d9ff` completes 125 methods
      with nineteen platform skips and no failures in 5659.990 seconds.
      Together with the recorded Linux GCC/Clang matrices, this satisfies
      translation-unit acceptance on the tested toolchains. The historical
      incident above remains open but is no longer a closure prerequisite.
      Before integrated-Clang admission I compare native objects with selected
      `-cc1as` text expansion and replay after deleting source/include/payload
      inputs. I check both `.s` and `.S`, ordinary/debug modes, runtime bytes
      and object differences; an expansion exit status alone is insufficient.
      I extended the inline-C search-order controls to standalone `.s`/`.S`: paired/joined
      `-Wa,-I`, split `-Xassembler`, common/platform/package flag placement,
      an earlier search candidate appearing, missing-input recovery, both
      cache roots and ordinary/shared units. Native controls distinguish
      C header lookup from assembler include lookup. These passing matrices
      do not establish a complete assembler input inventory or explain the
      separately recorded transient reuse failures.
- [x] **5.0 strict-aliasing flag snapshot coverage.** I characterize restored
      assembler-input edits with explicit `-fstrict-aliasing` and
      `-fno-strict-aliasing`, preserve these C optimization flags through
      retained compilation, and verify cold/warm/fresh answers and actual
      reuse under both cache roots. Unknown flags retain their documented
      compatibility path. MAC `task_5f1867205c26b3546d6a309793517c46`
      continues parent `task_443e8107d0ff4350999e0d5186a809f1`.
      Both flag settings fail the restored-input regression before admission.
      Context v46 admits these exact flags; focused capture/reuse and phase
      checks pass on Darwin Clang and Linux GCC 12/Clang 14. All 54 Darwin
      cache-publication methods pass in 146.166 seconds; six guides validate.
- [ ] **5.0 assembler-input snapshot capture.** I capture the bytes consumed
      by assembler file reads, including inline `.incbin`, and bind compilation
      and reuse to those captured inputs. I test restored edits and permanent
      replacements, nested includes, path spelling and compiler/assembler
      variants. Rehashing the C preprocessor output cannot satisfy this gate;
      merely withholding reuse is containment, not completed snapshot support.
      MAC `task_443e8107d0ff4350999e0d5186a809f1`.
      I first characterize Clang's external-assembler mode: whether `-S`
      resolves file reads, whether replay survives deletion, and whether the
      selected external assembler remains selected. I admit
      `-fno-integrated-as` only through the existing bounded capture paths;
      failed admitted external capture now stops the build. Unadmitted source
      and flag modes retain compatibility behavior, not a guarantee of
      cold-build snapshot consistency. Other compiler variants remain open.
- [x] **5.0 retained GCC PCH inputs.** I replace external PCH references in
      retained translation units with private copies and bind their bytes to
      cache identity. Before integration I verify that GCC accepts relocated
      PCH bytes with the original header and PCH removed, and that restored
      PCH replacement cannot affect compilation of that retained input.
      My include-trace parser also rejects GCC's selected-PCH `! ` record
      and following root-source line; I capture those dependencies to make
      retained PCH reuse possible.
      Production capture, restored edits, failure/recovery and reuse now pass
      for ordinary and shared-only C sources under local and shared caches.
      All 25 snapshots pass on GCC 12.2 and 13.3 Linux arm64 (two skips each),
      and again with the GCC 12 production builder/support under ASan/UBSan,
      leak detection disabled. Canonical unescaped PCH paths are retained;
      malformed, escaped and over-limit inputs keep the no-reuse fallback.
      Both bootstraps and the 126-method regression set pass (ten Linux and
      24 Darwin skips), with final rewrite rejection checks on both hosts.
      Other configured compiler inputs remain open (2026-09-13).
      MAC `task_443e8107d0ff4350999e0d5186a809f1`.
- [x] **5.0 native compile-flag collection capacity.** Before extending aggregate
      argument transport, I remove the fixed 1024-pointer compile-flag buffers
      in both source-free and compiled-module result paths. Metadata loops can
      write beyond their allocation. I preserve ordering and complete include
      paths, propagate allocation failure, and test lists above 1024 entries
      under sanitizers. Both returned paths now share a size-checked collector.
      Tests cover 1300 entries, long include paths, five allocation failures,
      count overflow and clean retry. The 136-method regression set passes on
      Darwin and Linux; Linux ASan/UBSan passes all 35 snapshots and focused
      leak-enabled failure checks (2026-09-13).
      MAC `task_443e8107d0ff4350999e0d5186a809f1`.
- [x] **5.0 native link-flag collection capacity.** I remove the remaining fixed
      pointer budgets in returned link flags and shared-link flag assembly.
      System-library and framework append loops still use unchecked increments,
      while bounded fragment appenders can silently drop flags. I measure the
      boundaries, preserve ordering, and verify allocation failure and retry
      before accepting aggregate native argument lists.
      I replace duplicate returned-link collectors with size-checked ownership,
      and append shared-link fragments directly through checked command writes.
      Shared framework deduplication currently separates names from their
      repeated `-framework` markers; I preserve complete pairs and explicit
      library repetitions, then test a real two-framework Darwin link.
      Both returned paths now use checked allocation, and shared linking has
      no temporary fixed-pointer lists. Tests cover 1300 entries, long library
      names, all collector allocation failures, overflow, retry, framework
      pairs, repeated libraries and rejected tail flags preserving old code.
      Both 140-method regression sets pass; all 39 Linux snapshot methods pass
      under ASan/UBSan, with focused leak-enabled checks (2026-09-13).
      MAC `task_641f3a82b3474cd28c895dd9ffcb290b`.
- [x] **5.0 Darwin driver-response regression gate.** My rebuilt tools at
      `d9bfd86c` pass the full 152-method compiler/shadow/cache/assembler/linker
      run in 341.417 seconds, with 24 expected skips and no failures or
      timeouts (2026-09-13). Ordinary probe deadlines remain unchanged; only
      the existing 40-package stress builds use their thirty-second bound.
      MAC `task_44c2d5851e1948e6a474036e263a0a73`.
- [x] **5.0 linker response end-of-input tokens.** I preserve the tested
      GNU/Apple behavior for unclosed quotes and trailing backslashes at end
      of a response. Both linkers accept the tested archive spellings, while
      my strict graph scanner previously declined them. I correct the scanner
      and cover nested references and exact decoded words. All twenty-two
      retained-graph native comparisons pass on Apple Clang 21 and GCC 12;
      Darwin passes thirty-six targeted methods and Linux passes sixteen
      graph/scanner methods with leak-enabled sanitizers (2026-09-13).
      MAC `task_443e8107d0ff4350999e0d5186a809f1`.
- [ ] **5.0 retained compiler response files.** I measure restored edits to
      `@file` arguments on Clang and GCC, including local and shared caches.
      I retain the selected arguments before compilation, preserving compiler
      response-file quoting and nested-path semantics rather than applying
      shell parsing to their contents. I verify restoration, permanent changes,
      missing/cyclic response files and phase-specific flags before accepting
      response-file builds as snapshots.
      The restored `-DANSWER=42` response file reproduces cold/warm/fresh
      43/43/42 on GCC 12 and 43/42/42 on Apple Clang 21. Both local and shared
      caches fail the cold-output requirement. My characterization CLI also
      checked only warm versus fresh output, incorrectly passing Clang's cold
      mismatch; I require both cold and warm results to match fresh output.
      I implement invocation-local metadata flag copies and fresh rebuild-check
      captures, expand literal GNU-style response words without shell evaluation,
      and keep package flag captures consistent with the same rule. Unsupported
      shell fragments remain on their existing path rather than being evaluated
      by the capture parser.
      The initial bounded literal path repaired the measured mismatch. Its
      limits were 16 nested files, 64 KiB cumulative file bytes, 4095-byte words
      and a 2048-byte serialized fragment. Transport below removes the last
      limit; I still need broader budgets and capture of shell-expanded or
      noncanonical response forms. Over-budget and noncanonical forms
      keep the previous path. Missing/cyclic/nonregular inputs fail capture.
      A Clang version banner alone does not establish GNU response syntax:
      I exclude named `clang-cl` drivers and explicit `--driver-mode` overrides
      from this capture boundary. Metadata and package captures roll back
      together if an unresolved response, shell fragment or driver-mode
      override remains; decoded escaped overrides follow the same rule.
      I next measure long literal argument lists through the production build,
      comparing cold, warm and fresh results in local and shared caches. The
      transport must survive phase filtering and the returned `ModuleBuildInfo`
      flags; increasing one parser buffer does not satisfy this lifetime.
      The 10,212-byte literal reproducer now confirms cold/warm/fresh
      43/43/42 with reuse on GCC 12 and 43/42/42 without reuse on Apple Clang
      21, in both local and shared caches. The explicit
      `--response-large --require-consistent` gate was red before transport.
      A passing characterization-instrument test did not satisfy that gate.
      I implement content-addressed GNU response sidecars under the module cache,
      keeping decoded arguments in identity and phase selection. Compilation,
      shared linking and returned native flags use the same transport; sidecars
      outlive build-info objects and are removed with the module cache. I verify
      byte equality on reuse and reject altered or nonregular sidecars. Tests
      cover long restored inputs, later consumers, phase filtering and recovery.
      The 10 KiB reproducer now returns 42/42/42 with reuse on both drivers.
      The remaining serialized-fragment budget is 64 KiB; many short fragments
      can still exceed aggregate command limits. Shell-expanded/noncanonical
      forms and other response dialects remain outside the retained boundary.
      I coalesce eligible common, active-platform and package compiler
      fragments in their invocation-owned arrays, preserving argument order
      and package slots. The combined strings use existing retained transport.
      I verify many short flags, many small response references, later consumers,
      allocation rollback and unchanged unsupported shell/driver-mode paths.
      The aggregate fixture passes for 1300 short flags, forty small response
      references and forty package fragments; all four variants fail against
      the previous revision. Groups beyond the 64 KiB budget remain outside
      this coalescing boundary.
      I quote include-directory arguments consistently for compilation
      and returned native flags, then coalesce them through retained sidecars.
      I keep original include paths in metadata and cache validation, verify
      ordered header search and later consumers, and test allocation rollback.
      The 200-directory fixture preserves quoted paths and header precedence;
      its compiled variant fails against the previous revision on GCC 12.
      I retain aggregate literal common and platform linker flags without
      changing their order, returned-flag lifetime or linker cache observation.
      I test long search lists, repeated libraries, later native consumers and
      failed replacements before extending transport to other linker groups.
      My new characterization reproduces shared-link rejection on Apple Clang
      21 and GCC 12 for common, platform, package and system-library groups.
      Direct links and later native consumers return 42 with those same flags;
      the builder preserves the old generation but cannot publish a new one.
      The explicit acceptance gate rejects failure until cold/warm answers, reuse,
      returned argument order and failed-replacement preservation all pass.
      I implement one ordered shared-link argument fragment through retained
      driver transport, including package libraries and system-library names.
      Darwin observation must recognize only its exact retained fragment;
      raw indirect response arguments remain on their previous path.
      The four long-link cases now pass on both drivers with cold/warm/later
      answers of 42, complete returned argument order and generation reuse.
      I reject changed retained linker files before publishing a replacement
      and recover after removal. The allocation-failure retry and indirect
      response visibility checks pass. Larger groups and indirect linker
      response capture remain open.
      I next measure response arguments supplied through linker metadata,
      restoring an archive-selection change during the shared link. I compare
      cold, warm and fresh results in local and shared caches before extending
      invocation-owned capture to these arguments.
      Apple Clang 21 reproduces cold/warm/fresh 43/42/42 in both caches.
      I extend atomic metadata/package captures to driver response arguments
      in common and platform linker flags and package libraries, retaining
      decoded arguments for later consumers. Forwarded linker response syntax
      remains distinct from driver response syntax.
      I also correct the characterization's fresh-link archive order: archive
      arguments follow the source/object that refers to them on GNU linkers.
      The corrected GCC baseline reproduces 43/42/42 for common/platform/package
      driver responses in both cache locations. My capture now returns 42/42/42
      with reuse. I test owned returned arguments after input removal, missing,
      cyclic and FIFO inputs, recovery, and metadata allocation rollback.
      I replace the old Darwin fallback expectation for captured driver
      `-Xlinker @file` arguments with explicit reuse-record and warm-reuse checks.
      The 40-package stress fixture exceeded its ten-second per-build timeout
      once; its isolated recheck passed. I give only those stress builds a
      thirty-second test bound, leaving ordinary probe deadlines unchanged.
      I next measure `-Wl,@file` separately from driver `@file` arguments,
      including common/platform flags and package libraries. I require actual
      linker evidence before sharing a response parser across those boundaries.
      The forwarded-response fixture reproduces cold/warm/fresh 43/42/42 in
      all six cases on Apple Clang 21 and GCC 12. The explicit acceptance gate
      remains red; driver-response capture does not repair this linker boundary.
      I compare native linker interpretation against my driver-response decoder
      for quoting, control whitespace, relative nesting and repeated response
      files. Published GNU and Apple implementations are not a shared parser
      contract. My twelve-case experiment finds three admitted substitutions
      that change Apple linker rejection into success (vertical tab, form feed,
      repeated response); all eleven admitted GNU cases agree. Unchanged driver
      decoder reuse is therefore not a valid cross-platform repair. Selected
      linker identity and preserved failure behavior remain required.
      I next compare byte-preserving retained responses, including distinct
      files with identical bytes and repeated references. Retention must not
      collapse path identity or leave nested inputs mutable; I test those
      constraints before selecting the capture representation.
      My sixteen-case prototype preserves native outcomes on both hosts after
      removing original response files when retention uses resolved-path
      identity. Apple distinguishes hardlink paths but rejects symlink aliases;
      content-only and spelling-only prototypes change native outcomes.
      Production graph capture and general nested-token rewriting remain open.
      I implement bounded graph capture with explicit GNU/Apple token grammar,
      resolved-path memoization, verbatim non-reference bytes and retained-file
      verification. I exercise that mechanism through the production probe
      before wiring invocation ownership and cache eligibility to it.
      The C mechanism matches all sixteen native-linker outcomes and retains
      original-file independence. Allocation recovery, bounds and tamper checks
      pass under sanitizers. I still need invocation-wide admission and returned
      transport, including retained cache paths containing commas that cannot
      be inserted unchanged into a `-Wl,` argument.
      I query the selected linker through the complete literal link command,
      with bounded output and a supervised deadline. Clang's
      `-print-prog-name=ld` still names Apple ld under `-fuse-ld=lld`, while
      its actual recipe selects ld64.lld; that lookup cannot authorize a grammar.
      Apple's version-details query can complete a link. Query integration
      therefore needs disposable outputs and must never target a published
      generation; a version flag is not a read-only guarantee.
      My bounded query mechanism now recognizes tested native tools through
      the complete command, including `-B` and response-file-hidden selectors,
      and rejects unsupported selected linkers. Query/graph sanitizer checks
      and the 168-method Darwin gate pass. Private probe construction and
      atomic invocation integration remain open.
      I construct disposable query directories and pin primary output at both
      driver and linker layers, preserving complete flag order. I test metadata
      output overrides and cleanup before using this in cache admission.
      Auxiliary outputs and unclassified option controls still need admission
      rules; a private primary output is not filesystem isolation.
      Twenty-four primary-output override combinations preserve bytes and
      timestamps on both hosts; private-directory cleanup and thirty-two
      allocation budgets pass. The query's indirect inputs still need retained
      validation before execution, not a check followed by a mutable reread.
      I capture multiple response roots in one transaction, sharing resolved
      identities, frozen source spellings and input budgets across the whole
      invocation. A failed root or allocation must return no partial path set;
      I test repeated/shared roots, intervening mutation and recovery.
      The batch API now returns a complete owned path set or nothing. Repeated
      source spellings retain their first binding across mutation, removal and
      symlink retargeting; a new transaction sees the new state. Native
      multi-root order, identity and lifetime checks pass on both hosts, along
      with allocation and shared-budget tests. Flag installation and admission
      remain separate open work.
      I next measure explicit `-Xlinker` argument materialization from retained
      graphs, including comma-containing cache paths, control whitespace,
      malformed quoting and repeated resolved identities. I require observed
      native outcomes before replacing response-file transport; decoder
      refusal alone is not evidence of equivalent linker rejection.
      My twenty-two-case comparison matches every GNU result after correcting
      end-of-input token semantics. Nineteen Apple cases execute equivalently;
      three repeated/aliased-response cases are deliberately refused, so the
      full Apple materialization gate remains red. Comma/space cache paths
      and quoted CRLF filenames are covered. Production transport is not yet
      installed, and the six restored-selection failures remain open.
      I implement C argument capture within the existing graph transaction:
      ordered owned fragments, no response-file reread, shared input limits,
      bounded aggregate expansion, and explicit Apple repeated-identity
      rejection. I test native links after input removal, exact quoting,
      comma/space paths, root order, allocation rollback and retry before
      installing this representation in the invocation.
      The C argument transaction now passes all twenty-two native outcome
      checks on both hosts: Apple rejects three repeated identities explicitly,
      while the other cases execute matching links after input removal. Exact
      output bounds, eighty allocation budgets and six read-boundary mutation
      cases pass. My Darwin bytecode/shadow/cache target passes 175 methods
      (fifteen skips); Linux passes seventeen graph/acceptance methods and the
      native comparison with leak-enabled sanitizers. I still need to admit
      and install the captured arguments across the actual invocation.
      I enforce literal option admission at the public linker-query boundary.
      I reject unresolved responses, indirect controls, auxiliary outputs,
      plugins and unknown switches before starting a tool; I test direct,
      `-Wl,` and `-Xlinker` spellings and operand boundaries. The raw
      primary-output mechanism remains separately testable. Option admission
      does not establish safety of native input contents or compiler wrappers.
      Public admission now rejects the tested controls before tool execution
      and preserves admitted native linker overrides. A real query succeeds
      using captured C arguments after response removal. Darwin passes all
      forty-three targeted methods; Linux passes twenty-nine normal graph/query
      methods and thirteen query methods with leak-enabled sanitizers. The
      native-input trust boundary and actual invocation installation remain
      open; I have not enabled reuse through this query API.
      I install complete owned metadata/package candidates in ordinary
      builds and rebuild checks. I prepare both grammar candidates before
      querying either, use a controlled empty C input with the shared-link
      flag recipe, and select only a self-confirmed grammar. The query keeps
      my existing trusted native-module/toolchain boundary; it is not a native
      input sandbox. I include linker flags in build identity and retain the
      selected package arguments during post-build validation. I require the
      six restored-selection cases, response edits, returned-flag lifetime,
      rejection, rollback and both-platform regressions before acceptance.
      My new compiler-group cases expose a Darwin reuse defect: linker-only
      arguments reach C capture, and their unused-argument diagnostics prevent
      complete dependency evidence. I remove confirmed linker-only pairs from
      source phases while preserving them in the actual shared link.
      The supported path now passes all six restored-selection cases, edited
      responses across all six argument groups, returned-flag lifetime,
      explicit Apple repeated-root rejection, complete fallback and 180
      allocation budgets with same-process retry. Both full shadow/cache gates
      pass 184 methods: Darwin has fifteen skips and Linux twelve. Unsupported
      forms, broader budgets and native input snapshots remain open; this
      checkpoint does not complete the whole retained-response item.
      Linux also passes thirty-two graph/query/invocation methods with
      leak-enabled ASan/UBSan, including allocation rollback and lifetime.
      MAC `task_443e8107d0ff4350999e0d5186a809f1`.
- [ ] **5.0 cache source snapshot acceptance.** I test source and header
      changes restored during compilation, preserving their original bytes
      and timestamps before final validation. I compare actual cold, warm and
      fresh library results. Matching preprocessing observations must not
      authorize a generation compiled from different input bytes. I record
      reproduced failures before implementing a supported snapshot boundary.
      `python3 -m tests.characterize_source_snapshot --require-consistent`
      failed against `34fab3e9` on Apple clang 21: all four cases publish/reuse 43 despite restored
      input bytes and timestamps; fresh compilation returns 42. Warm builds
      perform no additional C compilation. The ordinary-Clang repair above
      now passes those cases, as does ordinary GCC; configured modes still need
      snapshot acceptance (2026-09-12).
      MAC `task_443e8107d0ff4350999e0d5186a809f1`.
- [ ] **5.0 C-library cache transactions.** Content invalidation and private
      writes are addressed in adjacent items. Per-module generation publication
      and exact bytecode generation bindings are addressed below. I still need
      generation retention/collection policy, source snapshots and
      power-loss durability before I claim complete cache transactions.
      Driver and documented environment identity are addressed below. Automatic
      tracking of transitive compiler tools, linker/library bytes, SDK contents,
      pkg-config results and arbitrary wrapper inputs remains open. An explicit
      toolchain stamp supplies invalidation, not discovery or authentication.
      MAC `task_443e8107d0ff4350999e0d5186a809f1`.
- [x] **5.0 metadata identifier truncation.** I append exported metadata and
      constant identifiers directly to the dynamic output buffer. I test long
      3,000-character names and build with GCC 12 at `-O3 -Werror`, which
      rejected their passage through a 2048-byte formatting buffer. All 18
      metadata tests pass on Linux and macOS (2026-09-12).
      MAC `task_443e8107d0ff4350999e0d5186a809f1`.
- [x] **5.0 Linux compiler/VM cache acceptance.** I build the actual bytecode
      compiler and VM in isolated Linux storage and run shadows, the complete
      cache suite, wrappers, code generation and FFI. I distinguish probe-only
      evidence from CLI/runtime integration and record any portability defects
      before fixing them. Dependency installation stays inside the disposable
      environment; test execution has no network access.
      On Linux arm64, GCC 12.2/ld 2.40, 28 shadows, 45 cache tests (8
      platform-specific skips), 4 Linux linker tests, dependency rechecks,
      5 wrapper link tests, 7 wrapper boundaries, 63 codegen tests and 19 FFI
      tests pass. This is not a full release or sanitizer gate (2026-09-12).
      MAC `task_443e8107d0ff4350999e0d5186a809f1`.
- [x] **5.0 cache Linux linker evidence.** I run the existing archive/search,
      unusual-path, response-file and thin-archive experiment on an actual GNU
      linker before choosing its dependency-record integration. I record the
      compiler/linker versions and distinguish successful experiments from
      safe cache acceptance. I use read-only source input and disposable Linux
      build storage; the Darwin parser does not establish Linux behavior.
      GCC 12.2 / GNU ld 2.40 on Linux arm64 reproduces cached 42 versus fresh
      43/44 after archive/search changes. Dependency files omit the tested
      negative candidate, response file and external thin-archive member;
      unusual paths appear unescaped. Raw normalized JSON and reproduction
      steps are in `docs/LINKER_INPUT_EVIDENCE.md`. Six production-builder
      methods pass under GCC `-O3`, including the 28-case crash matrix and
      surviving-child cases. This is neither the Linux invalidation repair
      nor a full Linux compiler/VM or sanitizer gate (2026-09-12).
      MAC `task_443e8107d0ff4350999e0d5186a809f1`.
- [x] **5.0 module flag fragments — preserve quoted bytes.** My whitespace
      tokenizer changed a quoted newline in an archive path into a space.
      I preserve supplied flag fragments intact and in order instead of
      tokenizing them without a shell grammar. I test the unusual archive
      through cold and warm Linux links and keep package/foreign-build gates.
      This does not make configured shell fragments untrusted or sandboxed.
      Cold/warm archive replacement through a quoted space, quote, backslash
      and newline path passes on Darwin and Linux. The package and foreign
      build gates below pass. My v14 cache context invalidates the old
      flag-splitting semantics (2026-09-12).
      MAC `task_443e8107d0ff4350999e0d5186a809f1`.
- [x] **5.0 Linux warm-link validation.** Before reusing cached C artifacts,
      I run the original shared-link recipe against their objects into private
      staging and compare the library bytes. Equal output retains the current
      generation; changed output triggers rebuilding; a failed validation link
      fails without publishing or retrying past the failure. I test archive
      edits, earlier search candidates, thin members, response files, unchanged
      reuse, shared-only objects and failed-link recovery. Warm Linux builds
      pay for one link. This does not replace full input identity or snapshots.
      All four new Linux methods pass under GCC 12.2 `-O3` and ASan/UBSan;
      the sanitizer build instruments the production builder and linked
      support sources, not fixture libraries or system libraries. Eleven
      optimized Linux methods pass including prior barrier/crash/cleanup
      boundaries. Darwin passes 28 shadows, 53 cache tests, five wrapper link
      tests, seven wrapper boundary tests, 63 codegen tests, 19 FFI tests and
      package/dependency gates. The four Linux-specific methods skip on
      Darwin, rather than claiming that platform executes them (2026-09-12).
      MAC `task_443e8107d0ff4350999e0d5186a809f1`.
- [ ] **5.0 GNU linker cache invalidation.** I repair the reproduced stale
      archive and search-selection results, including thin members and
      indirect flag inputs. I establish a lossless supported-mode boundary
      rather than treating GNU ld's readable dependency file as complete.
      I require actual changed results, unchanged reuse, failed-build
      preservation and recovery on Linux before marking this complete.
      MAC `task_443e8107d0ff4350999e0d5186a809f1`.
- [x] **5.0 cache retention — ordinary build cleanup.** `make clean` previously
      removed `obj/module_cache`, invalidating existing bytecode bindings. I
      preserve the default and configured runtime caches while cleaning build
      trees, reject unsafe cleanup roots, and test actual clean invocations in
      isolated workspaces. Runtime-cache reset remains a separate destructive
      action requiring quiescence and acceptance of invalidated artifacts.
      I also replace the module guide's obsolete timestamp-only and
      `.build_info.json` descriptions with the current generation layout.
      Project/example recipes preserve six configured cache placements, local
      and default caches, earlier marked cache locations and private stages.
      Four cleanup methods cover those cases, direct generation/stage roots,
      unsafe roots and symlink ancestors. A real library loads in a fresh
      process after clean and the next build reuses its generation. The full
      normal gates pass: header invalidation, 28 shadows, 52 cache tests,
      five wrapper link tests, seven wrapper boundary tests, 63 codegen tests,
      19 FFI tests and package/dependency gates (Darwin, 2026-09-12). After
      extending conservative retention markers, I reran all cleanup methods
      and the real-library clean/reuse test successfully. Automatic collection
      and the broader transaction item remain open; no new sanitizer claim.
      MAC `task_443e8107d0ff4350999e0d5186a809f1`.
- [x] **5.0 cache retention — surviving compiler evidence.** I kill only the
      builder while a controlled compiler child is paused, then require a new
      acknowledgement from that child. I test whether another builder can
      publish while the child still uses its private stage, and verify that
      the child's late writes do not change published generations. I cover
      local and shared caches before choosing an abandoned-stage collection
      rule; a released module lock alone is not sufficient evidence.
      Both cases pass: a new builder publishes while the acknowledged child
      remains paused; the child then successfully writes its private object.
      Old and new generation bytes stay unchanged and warm reuse retains the
      replacement. All 51 cache tests pass on Darwin (2026-09-12). This
      increment changes tests and documentation only; it does not implement
      collection or claim a new sanitizer or power-loss result.
      MAC `task_443e8107d0ff4350999e0d5186a809f1`.
- [x] **5.0 cache cleanup — no symlink traversal.** My private-directory
      cleanup followed a substituted root symlink. I anchor cleanup
      to an opened directory descriptor, refuse symlink roots and never
      recurse into nested directories. I test external-file preservation,
      ordinary files, symlink entries, FIFOs and retained nested artifacts
      before adding abandoned-build collection. This does not complete the
      generation retention/collection policy above.
      The old implementation deletes an external fixture file; the repaired
      implementation passes all six cases, including directory-path replacement
      after opening. The same method passes with ASan/UBSan on the production
      builder probe; linked support objects remain uninstrumented. All 28
      shadows, 50 cache tests, five wrapper link tests, seven wrapper boundary
      tests, 63 codegen tests, 19 FFI tests and package/dependency gates pass
      on Darwin (2026-09-12).
      MAC `task_443e8107d0ff4350999e0d5186a809f1`.
- [x] **5.0 cache publication persistence barriers.** I flush generation files
      and the generation directory before publishing its name, then flush the
      cache directory before and after switching the current pointer. I test
      barrier ordering, interrupted calls, failure before/after the pointer
      switch, retained generations and retry. A post-switch failure must not
      delete the generation now referenced by current. Warm reuse must retry
      the cache-directory barrier. Ancestor barriers are addressed below;
      device-level power-loss behavior remains separate acceptance work.
      File, generation-directory, pre-pointer and post-pointer barrier failures,
      interrupted-call retry and warm barrier retry pass. I reject symlink,
      directory and FIFO entries without following them or blocking on a FIFO.
      All 28 shadows, 45 cache tests, five wrapper link tests, seven wrapper
      boundary tests, 63 codegen tests, 19 FFI tests and package/dependency gates
      pass on Darwin (2026-09-12). Both new persistence methods pass with
      ASan/UBSan on the production builder through the probe; linked support
      objects and fixture libraries are uninstrumented. These tests exercise
      syscall ordering and failures, not physical power loss.
      MAC `task_443e8107d0ff4350999e0d5186a809f1`.
- [x] **5.0 cache ancestor-directory barriers.** I flush the cache directory
      and its descriptor-resolved parents before switching current, and repeat
      the chain on warm reuse. I stop at the filesystem boundary or root;
      establishing the mount namespace is not a cache-build operation. I test
      newly created nested shared-cache directories, ancestor failure before
      first publication, retry, warm failure and barrier ordering. Device-level
      power-loss and full crash-recovery acceptance remain open.
      I verify the device/inode identities and child-to-parent order of newly
      created shared-cache ancestors. Injected ancestor failure before first
      publication and on warm reuse, retry and prior barrier ordering pass.
      All 28 shadows, 46 cache tests, five wrapper link tests, seven wrapper
      boundary tests, 63 codegen tests, 19 FFI tests and package/dependency gates
      pass on Darwin (2026-09-12). Three persistence methods also pass with
      ASan/UBSan on the production builder through the probe; linked support
      objects and fixture libraries remain uninstrumented. This establishes
      tested syscall ordering, not a physical power-loss result.
      MAC `task_443e8107d0ff4350999e0d5186a809f1`.
- [x] **5.0 Darwin cache device-flush requests.** I require `F_FULLFSYNC`
      after successful file and directory `fsync` barriers on Darwin, retry
      interruptions, and reject unsupported or failed device-flush requests
      without silently weakening the operation. I test actual host support,
      injected failures, publication preservation and warm retry. A successful
      request is not evidence of physical power-loss survival.
      Actual host calls, six injected barrier-failure/interruption cases,
      unsupported-operation rejection and retry pass. All 28 shadows,
      49 cache tests, five wrapper link tests, seven wrapper boundary tests,
      63 codegen tests, 19 FFI tests and package/dependency gates pass on
      Darwin (2026-09-12). Three barrier methods also pass with ASan/UBSan
      on the production builder through the probe; linked support objects
      and fixture libraries remain uninstrumented. My v13 cache context
      rebuilds older generations. Darwin filesystems that reject the stronger
      operation now fail the build; other hosts retain their `fsync` path.
      MAC `task_443e8107d0ff4350999e0d5186a809f1`.
- [x] **5.0 cache process-crash recovery.** I terminate the production builder
      at file/directory barriers and generation/pointer renames. I test first
      publication and replacement in local and shared caches, complete visible
      artifacts, retained old paths, lock release, recovery and warm reuse.
      Process termination is not a simulated filesystem or device power loss.
      All 28 combinations pass: seven boundaries, initial/replacement builds,
      local/shared caches. I verify actual SIGKILL and the last reached event,
      load libraries in fresh processes, and link/execute new and retained
      native objects. Recovery completes within the test timeout and subsequent
      warm reuse retains the recovered generation. No production repair was
      needed for these cases. All 28 shadows, 47 cache tests, five wrapper link
      tests, seven wrapper boundary tests, 63 codegen tests, 19 FFI tests and
      package/dependency gates pass on Darwin (2026-09-12). This increment does
      not add a sanitizer or physical power-loss result.
      MAC `task_443e8107d0ff4350999e0d5186a809f1`.
- [x] **5.0 binding string-pool and import-table allocation safety.** Adding
      binding paths exposed paired `realloc` growth that could leave dangling
      arrays after partial failure. I preserve existing entries on failed
      growth and return an unambiguous failure index. Six injected failures
      and retries pass, also under AddressSanitizer/UndefinedBehaviorSanitizer
      on the production pool implementation (2026-09-12). This is not a full
      audit of allocation propagation across every producer.
      MAC `task_0ba46839aee94135aaa99a9b7c207499`.
- [x] **5.0 bytecode foreign-generation bindings.** I retain the generation
      returned by each foreign build and bind both shadows and production
      imports to its library. I encode an explicit artifact import kind, preserve
      it through v2 conversion, reject lossy v1 output, and resolve bound symbols
      only through their library handle. I test rebuild stability, missing
      artifacts, transitive imports, isolation from unrelated loaded symbols,
      and wire-format validation. Relocatable packaging and artifact content
      authentication remain distinct requirements.
      My 28 shadow tests, 22 cache tests, 63 codegen tests, 19 FFI tests,
      v2 codec/conversion/end-to-end tests, verifier, nvm2c, package policy and
      C-seed dependency gates pass on Darwin (2026-09-12). I also test direct
      and transitive retained imports in co-process mode and a rebuild while
      a shadow is paused. Packaged-wrapper execution is checked below.
      MAC `task_443e8107d0ff4350999e0d5186a809f1`.
- [x] **5.0 standalone VM failure diagnostics.** Debug metadata does not
      establish that a stack trace was printed. I always report execution
      failure and its detail, including a missing bound foreign library.
      Direct and transitive missing-library regressions pass with debug
      metadata, including co-process failure exits (2026-09-12).
      MAC `task_4c6ff6e9986a49d6a01701a66b8842d6`.
- [x] **5.0 packaged-interpreter link acceptance.** I repaired missing UTF-8
      and module-build-directory objects and added crypto/thread/loader linkage.
      My wrapper compiler retains the build's optional OpenSSL library search
      directory, including with caller flag overrides in the checked dry run.
      Both positive standalone and daemon link tests now require success;
      the standalone positive gate failed before the repair and passes after it.
      An executed standalone wrapper retains its foreign generation after a
      rebuild and reports a missing artifact from a different working directory.
      Five wrapper tests, 28 shadows, 23 cache tests and 63 codegen tests pass
      on Darwin (2026-09-12). Daemon execution and cross-platform execution are
      not established by these link checks.
      MAC `task_4524b827bf464fe7bf4d0d43d6f88fd1`.
- [x] **5.0 wrapper command and publication boundaries.** Both wrapper modes
      share private sibling staging and atomic executable replacement. I quote
      filesystem arguments with my existing shell-path helpers and encode
      source import paths as C-string bytes. Compiler command fragments remain
      trusted configuration. I reject absent, empty, non-executable, symlinked
      and multiply-linked compiler outputs before publishing.
      Seven boundary tests cover literal paths (including option-like relative
      object directories), compiler failure, overlapping failure/success,
      interruption, destination symlinks, failed rename and import escaping.
      Five wrapper link tests, 28 shadows, 23 cache tests and 63 codegen tests
      also pass on Darwin (2026-09-12). Interrupted private stages are ignored,
      not automatically collected; unknown compiler side artifacts are retained
      with a diagnostic. I do not claim power-loss durability, a compiler
      sandbox or cross-platform execution from these tests.
      MAC `task_0750c33a06a14dd39baf4d3e77e37a0d`.
- [x] **5.0 foreign-module compiler path boundaries.** I quote filesystem
      arguments through ordinary compilation, relocatable-object combination,
      shared-only compilation and shared-library linking. Shared-only sources
      now receive the common include and platform compile settings. Five
      single/multi-source path cases cover quotes, dollar expressions,
      backslashes and newlines in source, include and cache locations. The
      original two cases fail before the repair; all five now execute and
      observe subsequent header edits. Compiler and flag command fragments
      remain trusted configuration, not a sandbox boundary.
      MAC `task_8c10a946dfa14d92b491fd80f4635187`.
- [x] **5.0 foreign command and dependency record bounds.** I reject truncated
      compiler commands before execution, decode supported Make escapes and
      require every expected dependency record. Malformed, missing, unreadable,
      NUL-containing or overlong records cannot establish reuse. My context
      version invalidates older evidence. The header-edit regression fails
      before this repair and passes afterward. Clang emits lossy dependency
      paths for literal backslashes: affected manifest paths (and newline
      paths) still compile but deliberately receive no reusable hash record.
      Parser boundary cases and oversized-command preservation pass alongside
      28 shadows, 26 cache tests, five wrapper link tests, seven wrapper boundary
      tests, 63 codegen tests, 19 FFI tests and package/dependency gates on Darwin
      (2026-09-12). These checks validate records, not compiler truthfulness.
      MAC `task_443e8107d0ff4350999e0d5186a809f1`.
- [x] **5.0 foreign system-header dependency coverage.** I record compiler
      dependencies for system headers as well as user headers. I test a
      transitive `-isystem` header edit without changing the manifest or source,
      and require new bytecode to observe its new value. `-MMD` omits these
      inputs; manually listed top-level headers do not cover their contents.
      I now request `-MD` for ordinary and shared-only compilations and bump
      the cache context version to invalidate older incomplete evidence.
      The regression fails before the repair and passes afterward, including
      unchanged-generation warm reuse and standalone VM execution. All 28
      shadows, 27 cache tests, five wrapper link tests, seven wrapper boundary
      tests, 63 codegen tests, 19 FFI tests and package/dependency gates pass
      on Darwin (2026-09-12). Lossless path reporting and full SDK/toolchain
      identity remain open; this does not establish either.
      MAC `task_443e8107d0ff4350999e0d5186a809f1`.
- [x] **5.0 transitive dependency alias rejection.** I capture include traces
      during the original compilation, without enabling saved-input mode.
      I decode supported escaped paths and hash the actual trace input as well
      as Make dependencies. Ambiguous trace paths must withhold cache evidence,
      not authorize reuse through a different readable dependency path.
      I preserve compiler diagnostics and test ordinary, multi-source and
      shared-only alias edits plus warm reuse for ordinary paths. This closes
      the reproduced stale-reuse case, not full lossless input capture.
      The three alias cases fail before the repair and pass afterward. The
      parser tests quoted, backslash, tab, newline and octal paths, ambiguous
      same-inode spellings, malformed records and missing files. GCC guard
      advice is accepted only for already recorded paths (format-fixture test,
      not a Linux end-to-end claim). Warning/error
      preservation and retry pass. All 28 shadows, 30 cache tests, five wrapper
      link tests, seven wrapper boundary tests, 63 codegen tests, 19 FFI tests
      and package/dependency gates pass on Darwin (2026-09-12).
      Both dependency-parser boundary methods also pass with ASan/UBSan on
      the production module builder; linked support objects are uninstrumented.
      MAC `task_443e8107d0ff4350999e0d5186a809f1`.
- [x] **5.0 fresh include-search validation.** I add fresh preprocessing as
      a supplemental cache veto, never as the original compilation's input.
      I compare observations before/after a cold build and before warm reuse,
      using the same configured flags for ordinary and shared-only sources.
      I test new earlier headers, unchanged reuse, failed observations and
      retained compilation mode. This is not an atomic source snapshot or
      complete PCH/module/toolchain identity.
      The twelve include-search cases fail before the repair and pass
      afterward. I also test failed/empty probes, a same-timestamp header
      mutation after compilation, retry and original Clang PCH behavior.
      My shared flag helper preserves successful empty pkg-config flags;
      the C-seed dependency gate caught and verified that regression repair.
      All 28 shadows, 34 cache tests, five wrapper link tests, seven wrapper
      boundary tests, 63 codegen tests, 19 FFI tests and package/dependency
      gates pass on Darwin (2026-09-12). Existing uncacheable configurations
      do not acquire extra probe invocations. Warm reuse now includes a
      preprocessing pass; original compilation and linking are still skipped.
      MAC `task_443e8107d0ff4350999e0d5186a809f1`.
- [x] **5.0 pkg-config query status.** I distinguish a successful empty flag
      response from failure, reject partial/invalid output, and propagate query
      failure through cold, warm and source-free module builds. I preserve
      existing artifacts on failure and test recovery without installing any
      packages. Query arguments and environment paths remain literal data.
      I support an explicit `PKG_CONFIG` executable override so builds and
      tests can select their tool without replacing a host installation.
      I also correct the compiler-input experiment's build command to use
      `make nano_virt nano_vm`: file targets alone did not rebuild stale tools.
      The partial-output/nonzero-exit regression fails before the repair and
      passes afterward. Cold/warm/source-free failure and recovery, signals,
      NUL/oversized output, the 64-KiB boundary and literal arguments pass.
      All 28 shadows, 36 cache tests, five wrapper link tests, seven wrapper
      boundary tests, 63 codegen tests, 19 FFI tests and package/dependency
      gates pass on Darwin (2026-09-12). The two new query methods also pass
      with ASan/UBSan on the production module builder through the probe;
      linked support objects and CLI binaries remain uninstrumented.
      MAC `task_443e8107d0ff4350999e0d5186a809f1`.
- [x] **5.0 single-build package flag snapshot.** I capture required package
      compiler and linker responses once, then use that set for preprocessing,
      compilation, shared linking and returned build information. I include
      both responses in reuse evidence and withhold new evidence when fresh
      post-build responses differ or fail. I test changing successful responses,
      link-only invalidation, source-free builds and recovery. This does not
      establish selected library byte identity or an atomic source snapshot.
      All 28 shadows, 38 cache tests, five wrapper link tests, seven wrapper
      boundary tests, 63 codegen tests, 19 FFI tests and package/dependency
      gates pass on Darwin (2026-09-12). The expanded snapshot test also checks
      actual shared-link arguments and failed post-build queries with recovery.
      Four package-query methods pass with ASan/UBSan on the production module
      builder through the probe; linked support objects and CLI binaries remain
      uninstrumented. I capture sequential responses, not an atomic package
      database transaction.
      MAC `task_443e8107d0ff4350999e0d5186a809f1`.
- [ ] **5.0 lossless transitive compiler dependency evidence.** My current
      cache excludes known lossy manifest paths and supplements Make records
      with include traces. I still need a verified compiler-mode boundary for
      PCH, modules and other external inputs before claiming every cached
      input is the input the compiler actually read. Newly earlier include
      files, source snapshots and complete toolchain identity remain open.
      Query status and per-build response consistency are addressed above;
      the identity of selected libraries remains open.
      MAC `task_443e8107d0ff4350999e0d5186a809f1`.
      - [x] I reproduce unchanged-flag library mutations and new earlier
        library-search candidates through the actual cache. I compare linker
        input traces and dependency records against selected archive bytes,
        member paths and unusual filenames before choosing a lossless boundary.
        I record tool/platform identity and distinguish characterization from
        acceptance; changing runtime dependencies remains a separate boundary.
        Apple clang 21 / ld-1267 retains cached 42 after a same-size,
        same-timestamp archive edit and an earlier search candidate; fresh
        links return 43 and 44. Binary records preserve unusual paths and
        absent candidates but omit the supplied response file. Line traces
        contain literal newlines. My host's successful `ar rcsT` produces a
        regular archive, so thin-archive coverage is not established.
        `python3 -m tests.characterize_linker_inputs` and all 38 existing cache
        tests complete on Darwin (2026-09-12). Those green tests do not cover
        these stale-library cases. `docs/LINKER_INPUT_EVIDENCE.md` records the
        evidence and implementation requirements; this is not the repair.
      - [ ] I capture and validate selected linker-input bytes, negative
        search state and indirect flag inputs. Same-timestamp archive edits
        and newly earlier libraries must change actual cached results; warm
        reuse, malformed evidence, unusual paths, failed builds and recovery
        must be tested. I handle each supported linker mode explicitly and
        do not infer completeness from a readable trace or dependency file.
        - [x] I integrate Darwin tagged linker records as an additional cache
          requirement, checking external input hashes and absent candidates.
          Unsupported capture preserves ordinary linking without reuse evidence.
          Response-file commands remain uncacheable until their indirect inputs
          are captured. I test actual archive/search changes, warm reuse,
          malformed records, literal paths and failed-capture recovery. Other
          linkers, indirect inputs and snapshots remain explicit unfinished work.
          The reproducer now returns 43 and 44 for the archive/search edits,
          with unchanged generation reuse. Fourteen parser cases plus actual
          unsupported/malformed capture, response-file bypass and recovery pass.
          All 28 shadows, 41 cache tests, five wrapper link tests, seven wrapper
          boundary tests, 63 codegen tests, 19 FFI tests and package/dependency
          gates pass on Darwin (2026-09-12). The three new linker methods also
          pass with ASan/UBSan on the production builder through the probe;
          linked support objects and fixture libraries are uninstrumented.
          I hash files after linking; this is not a linker-input snapshot.
        - [x] I reproduce library replacement during the link, then require
          matching observed linker inputs around the final link before storing
          reuse evidence. I retain that observation rather than replacing it
          with later hashes. I test mutation, recovery, unchanged reuse and
          failed final links. Matching observations are not an atomic snapshot;
          changes that occur and revert still require captured input bytes.
          Changes after the original link and during later preprocessing
          reproduce incorrect code/hash pairing before the repair. Tests now
          cover discovery/final/postprocessing mutations, retry and warm reuse,
          plus failed final links preserving the prior generation without an
          extra retry. All 28 shadows, 43 cache tests, five wrapper link tests,
          seven wrapper boundary tests, 63 codegen tests, 19 FFI tests and
          package/dependency gates pass on Darwin (2026-09-12). Three timing,
          final-failure and fallback methods also pass with ASan/UBSan on the
          production builder through the probe; linked support objects and
          fixture libraries remain uninstrumented.
      - [x] I reproduce a transitive backslash/slash alias against the actual
        compiler and cache, then measure preprocessed-input snapshots as a
        candidate boundary. I check unchanged replay, header edits, include
        search changes and external precompiled-header references before
        selecting an implementation. Characterization is not cache acceptance.
        `python3 -m tests.characterize_compiler_inputs` confirms stale code
        on Apple clang 21.0.0 and shows that `-save-temps=obj` changes the
        result of a stale-PCH build from 42 to 43. DOT keeps backslash bytes
        but exposes logical SDK paths. I reject an unconditional snapshot
        switch; compiler modes need explicit handling. The observations and
        implementation acceptance requirements are in
        `docs/COMPILER_INPUT_EVIDENCE.md` (2026-09-12).
- [x] **5.0 C-library cache — atomic generation publication.** I publish
      object, library, dependency files and reuse evidence as one generation
      through an atomic current-pointer replacement. Native link inputs and
      runtime library lookup resolve a retained generation path. I test failed
      publication, concurrent readers/builders, recovery and old-path stability.
      Bytecode-to-exact-generation binding, source snapshots, retention policy
      and power-loss durability remain separate acceptance requirements.
      I test stable native object and runtime library paths after replacement,
      a live reader during a blocked build, final-pointer rename failure with
      the old pointer intact, malformed pointers, corrupted cached artifacts,
      and recovery. All 28 shadow tests, 15 cache tests, 63 codegen tests, 18
      FFI tests, installation policy and C-seed dependency gates pass on Darwin
      (2026-09-12). I make no power-loss or full toolchain-identity claim.
      MAC `task_443e8107d0ff4350999e0d5186a809f1`.
- [x] **5.0 cache fixture path preservation.** My tests replaced every `42`
      in generated source, including random temporary import paths. I now
      change only the expected shadow value and exercise an import path that
      deliberately contains `42`. A missing-module failure was test corruption,
      not evidence of a compiler lookup defect.
      MAC `task_443e8107d0ff4350999e0d5186a809f1`.
- [x] **5.0 cache namespace identity.** I replace slash-to-underscore keys
      with `v2-<SHA-256 of realpath(module_dir)>`. I reject unresolvable
      directories and undersized destination buffers. I leave ambiguous legacy
      shared caches untouched and rebuild in the new namespace. Metadata include
      fallback also uses the physical module directory; a context-version bump
      invalidates old local records that could depend on alias spelling.
      Four initial namespace tests and the additional alias-header regression
      fail before their repairs and pass afterward. Tests cover distinct paths
      with different headers, canonical aliases, long paths, buffer bounds,
      migration and actual library/bytecode execution. All 28 shadow tests,
      20 cache tests, 63 codegen tests, 18 FFI tests, installation policy and
      C-seed dependency gates pass on Darwin (2026-09-12). Make builds the cache
      probe with project flags; required OpenSSL flags survive caller overrides
      in the checked dry run. This is namespace identity, not artifact trust.
      MAC `task_443e8107d0ff4350999e0d5186a809f1`.
- [x] **5.0 C-library cache — driver and build-environment identity.** I
      invalidate cached C artifacts when the selected compiler command,
      resolved driver bytes, working directory or documented build environment
      changes. I treat unresolved shell compiler expressions as uncacheable,
      preserve warm reuse for unchanged inputs, and test actual changed output
      as well as failure/recovery. I document the remaining transitive
      toolchain and wrapper-input boundary rather than claiming hermeticity.
      Six initial scenarios fail against the previous compiler and pass with
      the repair. Expanded tests cover PATH resolution, same-timestamp driver
      replacement, CPATH, working directory, explicit toolchain stamp, missing
      and shell-expression compilers, old/malformed context records, and driver
      mutation during compilation. All 28 shadow tests, 10 cache tests, 63
      codegen tests, 18 FFI tests, installation policy and C-seed dependency
      gates pass (2026-09-12). Warm tests no longer change the selected compiler
      as a no-build sentinel; damaged-record tests keep driver identity fixed.
      MAC `task_443e8107d0ff4350999e0d5186a809f1`.
- [x] **5.0 C-library cache — private writes and publication lock.** I
      compile into a private directory, validate the expected artifacts, and
      serialize cache validation/build/publication across compiler processes.
      I publish complete files by rename and invalidate hash evidence before
      publication so a partial publication forces rebuilding. This initial
      per-file publication is superseded by generation publication above.
      I test partial
      object/library failures, an interrupted compiler, overlapping builders,
      cleanup and recovery. Per-file rename is not an atomic multi-file
      transaction or an immutable artifact binding for later consumers.
      Five publication tests cover these boundaries, symlink rejection,
      multi-source artifacts and failed-publication hash invalidation. They
      pass with 28 shadow tests, 63 codegen tests, 18 FFI tests, installation
      policy and C-seed dependency/recovery gates (2026-09-12).
      MAC `task_443e8107d0ff4350999e0d5186a809f1`.
- [x] **5.0 foreign-cache test environment parity.** I pass the fixture's
      cache environment to standalone execution as well as compilation.
      Make's exported cache path previously caused six existing scenarios
      to read a different cache. The full Make gate now passes with its
      exported environment; the new publication tests follow the same rule.
      MAC `task_443e8107d0ff4350999e0d5186a809f1`.
- [x] **5.0 VM opaque signature representation.** With its foreign library
      built, `advanced/datetime_demo.nano` reaches execution but fails in `now`:
      an imported opaque `DateTime` return is declared as a struct in bytecode
      and arrives as an integer. I preserve the declared opaque runtime kind
      in local, imported and extern signatures, with round-trip and rejected
      wrong-kind tests rather than weakening runtime return checks.
      Local malloc/free and foreign-library direct/transitive round trips
      pass, as do wrong-kind rejection and ordinary record returns in the same
      program. The original datetime example executes root shadows and main;
      its standalone bytecode exits zero. All 25 shadow tests, 63 codegen
      tests, 18 FFI tests, 272215 VM checks and failure/recovery suites, and
      all 28 cross-backend contract rows pass (2026-09-12).
      MAC `task_4c6ff6e9986a49d6a01701a66b8842d6`.
- [ ] **5.0 opaque nominal identity and null boundaries.** My environment
      resolves qualified opaque names by global short-name fallback. My
      argument checker accepts any struct or integer for an opaque parameter,
      despite describing only the declared handle or zero as valid. I require
      module-aware nominal identity and exact null handling through checker
      and backend metadata, with colliding-name and wrong-handle rejections.
      MAC `task_4c6ff6e9986a49d6a01701a66b8842d6`.
- [ ] **5.0 imported function namespace isolation.** A helper that imports
      a module as `foreign` cannot define its own `make`, `echo` or `read`
      when that imported module has those names: short names leak into the
      helper's declaration scope. I preserve module ownership through lookup
      and lowering, with same-named wrapper tests instead of renaming APIs.
      The imported-shadow characterization also reproduced this for a pure
      `answer` wrapper on the C seed and bytecode CLI; Stage 2 accepted it.
      My bytecode extern table also deduplicates declarations by bare function
      name across modules. I must preserve distinct source declarations before
      exact library bindings can isolate same-named qualified foreign calls.
      Pure function binding is now addressed by the earlier backend gates.
      Private visibility, conflicting aliases in one importer, selective type
      aliases and nominal identities still need acceptance tests and consistent
      rejection. Sequential Stage2 binding state is not a reentrant context.
      MAC `task_4c6ff6e9986a49d6a01701a66b8842d6`.
- [ ] **5.0 shadow-enabled VM example acceptance.** My rebuilt quick gate
      rejects 98 of 229 eligible examples after shadow execution is enabled.
      Direct checks expose integer negation emitted for floats and missing
      shadow locals. I repair source/compiler/runtime failures without
      bypassing shadows or hiding examples in exclusions. Six previously
      excluded library inputs now compile as shadow-only sources; I reconcile
      their eligibility with the build contract. I rerun the full example gate.
      The isolated rerun reports 101 failures, including unresolved FFI
      functions, shadow-bytecode verification failures, type errors and
      assertions. These are execution-dependent results, not a stable
      compile-only acceptance count. I retain both observations.
      After root-shadow typing, array-annotation checks, numeric builtin and
      nested-signature repairs, the latest rerun reports 87 failures of 229
      and seven excluded shadow-only sources now accepted. No compiler
      crash appears in this rerun. Many failures are unresolved FFI functions;
      verifier, assertion and type failures also remain. The expanded shadow
      suite, 63 codegen tests, shared-checker unit tests, parser-recovery gate,
      272215 VM checks, bootstrap smoke/no-C-seed checks and all 28 contract
      rows pass. Native bootstrap binaries still differ; the full quick and
      release gates are not green (2026-09-12).
      MAC `task_4c6ff6e9986a49d6a01701a66b8842d6`.
- [x] **5.0 VM example evidence — actual failure status.** My coverage loop
      reads status after an `if` statement, reporting failed compiles as
      `exited 0`, and recognizes only legacy diagnostics. I retain the failed
      invocation's status and show modern diagnostics, test silent and
      structured failures, then rerun coverage without weakening acceptance.
      A regression exercises the production loop with silent status 7,
      modern-diagnostic status 42, legacy-diagnostic status 3 and success.
      It passes; the real coverage rerun remains nonzero with actual failure
      statuses and diagnostics. This test runs in `test-vm-examples`.
      MAC `task_4c6ff6e9986a49d6a01701a66b8842d6`.
- [ ] **5.0 execution contract — daemon exit status parity.** I verify entry
      values and traps end-to-end through `nano_vm --daemon` and the vmd
      protocol without confusing RPC success with the program's result.
      MAC `task_6ac194c20d284641ad2798bf780177bd`.
- [x] **5.0 contract evidence — versioned executable matrix.** I version
      the existing seven-program corpus and check exact stdout and successful
      execution with the C seed, Stage 2, VM, and actual NanoISA-to-C AOT.
      I run VM and AOT from the same bytecode artifact, expose unsupported
      AOT cases as failures rather than conformance passes, and
      test runner failure handling, timeouts, and manifest validation. This
      initial corpus does not complete specification reconciliation or cover
      the whole language; I retain the umbrella contract item above.
      Ten runner tests pass. The real matrix reports 25 passing rows and
      three AOT failures and exits nonzero (2026-09-11); this checks existing
      rebuilt tools, not a new bootstrap or full release gate.
- [x] **5.0 AOT prerequisite — classifier control flow.** I preserve operand
      stack state across branches, stop interpreting terminated paths, and
      reject incompatible joins before C emission. I test both outcomes of a
      branch carrying a live operand, unreachable instructions, malformed
      joins and classifier local-count bounds. This does not establish
      flow-sensitive local or interprocedural aggregate type inference.
      My AOT suite passes 335 checks, also with the translator and harness
      instrumented by ASan/UBSan (leak detection disabled; linked support
      objects and emitted programs are not instrumented) (2026-09-11).
- [x] **5.0 AOT prerequisite — executable edge transfers.** I emit join
      assignments before unconditional jumps and inside conditional taken
      edges. I copy incoming values simultaneously so loop-carried stack
      permutations cannot overwrite one another. Native regressions must
      exercise a live accumulator and a swapped pair across backedges.
      All four accumulator/permutation and unconditional/conditional
      combinations compile under C11 warnings-as-errors and execute with
      expected status. This is tested behavior, not a correspondence proof.
- [x] **5.0 contract defect — AOT source corpus aggregates.** I make the
      record-return and two variant corpus programs compile and execute via
      `nvm2c`. All three now execute with exact expected stdout. The required
      seven-program matrix passes all 28 rows on C seed, Stage 2, VM and AOT
      (2026-09-12). This checks the rebuilt translator and existing compiler
      tools, not a new bootstrap or complete language conformance.
      MAC `task_9a7214c876e24c3491f8d1d49c1384a1`.
      MAC rejects direct closure from `open`; verified evidence awaits its
      normal claim/review workflow.
- [x] **5.0 AOT record calls — converged type facts.** I propagate parameter
      kinds and flat aggregate fields across direct calls, converge return
      field facts before emission, and support record/variant value returns.
      I test caller/callee ordering, multi-hop string fields, tail returns,
      conflicting layouts and unresolved fields. I retain runtime field
      guards and explicit rejection outside this representable subset.
      My structured-C suite passes 371 checks, including recursive returns
      and branch joins with initially unknown field kinds. The translator
      and harness also pass with ASan/UBSan, leak detection disabled; linked
      support objects and generated programs are not instrumented. A known
      string-as-integer return now fails translation before its runtime guard.
- [ ] **5.0 AOT aggregate breadth — declared layouts and richer fields.** I
      extend the flat direct-call subset to nested aggregates, field kinds
      varying by variant or control flow, arrays across aggregate signatures,
      separately linked parameter metadata and path-sensitive local facts.
      I use declared NanoISA type/layout evidence where available and add
      corresponding VM/AOT semantic fixtures without weakening rejection.
      MAC `task_a4fde0d59ad24fe18c285a76ad58c176`.
- [x] **5.0 AOT aggregate boundaries — unreachable labels and field kinds.**
      I omit labels reached only by jumps in dead code after returning match
      arms. I retain runtime field kinds so incomplete static classification
      traps instead of reading the wrong parallel field storage. I test
      variant tags, payloads, empty variants, and non-variant tag rejection;
      richer interprocedural field classification remains required above.
      My structured-C suite passes 356 checks, including full-width variant
      tags, integer/string payloads, dead labels at function end and two
      runtime guard failures. Generated programs compile with C11
      warnings-as-errors and execute; source variant cases pass the matrix
      (2026-09-11).
- [x] **5.0 contract evidence — cross-backend runner failures.** I reject
      execution failures even when stdout matches, reject unknown or empty
      backend selections, and use private scratch paths. I retain explicit
      XFAIL/XPASS and validation-only reporting. Eight tests exercise the real
      runner with controlled compiler/executor fixtures; all four executor
      failure cases falsely succeed with the previous runner and fail correctly
      after repair. `make test-cross-backend-runner` passes and is a prerequisite
      of the full runner. The real C seed passes all seven C corpus programs;
      fixture success alone is not compiler conformance (2026-09-11).
- [x] **5.0 contract defect — self-hosted CLI output modes.** I parse
      options before compilation and reject unknown options, missing operands,
      unsupported targets and extra input files. `--target c` writes C source
      without invoking the native toolchain; `--target native` retains native
      compilation. I derive a sibling `.c` filename when `-o` is absent and
      accept `--` before a hyphen-prefixed input path. Nine CLI tests pass on
      rebuilt Stage 1 and Stage 2, covering output preservation on bad options,
      source compilation/execution with native compilation disabled in the
      driver, default paths, explicit native execution and write errors.
      `make test-selfhost-cli` passes; my full self-hosted suite reports
      16 passed, 0 failed. Both the C seed and Stage 2 compile and execute all
      seven C corpus programs using repository runtime headers. Eight runner
      fixture tests pass. The serial bootstrap passes its smoke/no-C-seed gates;
      native stage binaries still differ. README distinguishes self-hosted
      modes from C-seed-only target, documentation and profiling switches.
      MAC rejects direct closure from `open`; verified evidence awaits its
      normal claim/review workflow (2026-09-11).
      MAC `task_09aa81ed3aa442cba41a53d2c4b646e6`.
- [x] **5.0 bootstrap — evidence-bounded reporting.** I report native
      byte equality only for the artifacts compared in this build, label
      possible difference causes as hypotheses, and distinguish smoke checks
      from compiler correctness and canonical NanoISA equality. Three
      isolated reporting tests exercise identical and differing artifacts
      and preserve deterministic-mode rejection. Four source-dependency
      tests also pass; fake reporting fixtures do not test compiler semantics.
- [ ] **5.0 audit — formal foundations.** I assess Rocq, Lean,
      Isabelle/HOL, and HOL4 against my existing NanoCore development,
      then record a reproducible proof build, theorem assumptions, and
      implementation correspondence limits. I evaluate Sail on real NanoISA
      decoding and execution, with differential tests against the VM and
      explicit ownership of the ISA source of truth. I retain useful existing
      proofs and adopt additional frameworks only with demonstrated benefit.
- [x] **5.0 audit defect — quick gate after bootstrap.** On 2026-09-11,
      my three-stage build passes but `make test-quick` reports six language
      passes and eight failures through the selected `bin/nanoc`: five parser
      failures, a module-relative import failure, and two audio-example C
      compilation failures with undeclared string/filesystem helpers. I
      reproduce these against explicit compiler stages, repair their causes,
      and rerun the full quick gate without skipping tests. Bootstrap success
      is not evidence that the selected compiler passes my language corpus.
      Explicit `nanoc_c` compilation passes for control flow, enums, and
      module metadata; `bin/nanoc` selects the older `nanoc_stage2` binary.
      A fresh C-seed-built NanoLang compiler with the parenthesized-expression
      repair below now compiles and runs 11 of 14 language tests. Module
      metadata and the two audio examples still fail. I can select a compiler
      explicitly with `NANOLANG_COMPILER` in `tests/run_all_tests.sh`; the
      default remains `bin/nanoc` and the runner prints the selected path.
      After the relative-import repair, the candidate passes 12 of 15 tests
      including the new import regression. Metadata now resolves its import
      but fails on missing introspection declarations; both audio examples
      still fail on missing generated-C declarations (2026-09-11).
      The declaration/string-runtime increment now passes 16/16 language tests
      on the C seed and 15/16 on the fresh NanoLang candidate. Both audio
      examples pass; metadata fails at link time on absent `___module_*`
      definitions. The installed stage-two compiler is still unchanged.
      MAC `task_b11b688543574e82a96c0fb4f782d234` tracks this repair.
      Final verification on 2026-09-11: the dependency-aware `make test-quick`
      rebuilds both compiler stages and all three components, then exits 0.
      Selected `bin/nanoc` passes all 17 language programs, explicit native
      parser/extern/introspection regressions, VM example coverage, frontend
      checks, Jackson evidence and Forth PTY liveness. Gforth comparison skips
      because Gforth is absent; GLUT interactive launch is not requested;
      IDE graphical initialization skips without xvfb-run/timeout. This gate
      is not the full release suite or proof of compiler correctness.
      MAC rejected direct closure from `open`; I retain the verified evidence
      and leave ledger closure to its normal claim/review workflow.
- [x] **5.0 parser parity — grouped identifiers and tuple projections.**
      I no longer commit to a prefix call merely because `(` is followed
      by an identifier. Operator, comma, and tuple-projection continuations
      use the existing expression/tuple parser. Eight AST-shape cases,
      seven call-classification cases, equal-precedence left association,
      and malformed grouping now execute in `make test-parser-parenthesized`.
      I corrected the earlier evidence claim: the imported empty driver did
      not establish shadow execution. The explicit test also exposed outdated
      lexer/parser argument counts in those shadows, which I corrected.
      A fresh compiler candidate compiles and runs the five previously
      parse-failing language tests (2026-09-11). This does not establish
      complete compiler parity or install a new stage-two compiler.
- [x] **5.0 audit defect — imported parser shadow evidence.** I run the
      parenthesized-parser assertions explicitly from a compiled test entry
      point. A successful importer build or empty driver run does not establish
      that imported shadows ran; the C seed skips extern-dependent shadows.
      `make test-parser-parenthesized` passes and is included in `test-quick`.
- [ ] **5.0 audit — component test execution.** I audit remaining component
      shadows and bootstrap validation claims, execute meaningful assertions
      explicitly, and distinguish skipped from executed checks. MAC
      `task_56a065134a6e4394ae5c307c05e9597d`.
      I reviewed PR #266's complete change: 39 lexer-call signature repairs,
      already present in my integration tree with source, filename and
      diagnostics. That repair does not execute component assertions. My
      stage-three drivers still print library-load messages; the broader
      component-execution audit remains open. I retain the PR's ancestry
      without replacing newer transpiler code. Evidence:
      `docs/evidence/pr-266-reconciliation.md`.
- [x] **5.0 import parity — bare relative paths.** I try a bare import path
      beside its importing file before falling back to the repository root.
      `tests/nl_functions_relative_import.nano` compiles and runs through both
      the C seed and a fresh NanoLang compiler candidate. Module metadata
      emission remains a separate failing case in the language gate.
- [x] **5.0 generated-C parity — declarations and string helpers.** I
      stop treating an interior runtime-family substring as a supplied C
      declaration. I share trim and suffix-test runtime implementations across
      the C seed and self-hosted emitter, test boundary behavior through both
      compilers, and rerun the audio regressions. Metadata definitions remain
      a separate module-emission requirement, not satisfied by a prototype.
      Shared `runtime/string_edges.h` preserves the seed's trim/suffix behavior.
      The compiled string-boundary program passes on both compilers, both
      candidate audio regressions pass, and `make test-transpiler-externs`
      executes and passes declaration-selection and prototype assertions.
      That native test is now part of `test-quick` (2026-09-11).
- [ ] **5.0 audit defect — string-call equality.** The new compiled string
      boundary test exposes pointer comparison for two string-returning calls
      in the self-hosted C emitter. I use semantic operand types for string
      equality, retain a runtime regression, and audit similar text-based
      type guesses in arithmetic and comparison lowering.
      Direct string-call equality now uses inferred types and declared function
      return types. Compiled trim/character and user-function equality and
      inequality cases pass on both compilers. The broader inference audit
      remains open under MAC `task_e7a7395191d44cf793d684e21ea16a79`.
- [x] **5.0 audit defect — string-call concatenation.** Stage-two bootstrap
      emitted C pointer addition for `(mi_inventory ...) + (mi_inventory ...)`.
      I use semantic operand types for concatenation, retain a native
      string-returning-call regression, and rerun the actual bootstrap.
      The old stage one rejects the regression; the C seed passes it. With
      generated-text guesses removed, rebuilt stage one compiles and executes
      all 17 language programs, including nested concatenation and an
      integer-returning function named `str_concat_count`. Actual stage two
      now compiles successfully and passes its executable smoke test
      (2026-09-11). This does not establish canonical artifact equality.
- [ ] **5.0 audit defect — interpreted versus native codegen assertions.**
      Calling the extern-prototype regression's `main` from a C-seed shadow
      failed its three generated-text assertions, while the compiled entry
      point passes them. I investigate the evaluator/AST-list correspondence
      and retain distinct native and interpreted evidence; running only the
      compiled assertions does not establish interpreter parity.
      MAC `task_e7a7395191d44cf793d684e21ea16a79`.
- [x] **5.0 audit defect — bootstrap source invalidation.** My stage-two
      component and bootstrap-stage-one sentinels depend on prior-stage
      sentinels, not the NanoLang sources they compile. I add source dependency
      tracking and test that parser/import changes rebuild the affected
      compiler stages. A stale selected binary must not survive a successful
      source rebuild unnoticed. MAC `task_a98f6940228e47b3a503472d0ba5d2a4`.
      I also check missing stage binaries through transitive dependencies:
      the current wrapper only checks the stage named directly by the user.
      Source dependencies and selected-compiler bootstrap ordering are now
      implemented. Four isolated tests query the real make rules for clean
      state, changed sources/build rules, unrelated examples, and missing
      artifacts. Stage one rebuilds and passes its executable smoke test.
      Stage-two recompilation now succeeds and its smoke test passes. The
      full selected-compiler `make test-quick` now exits 0 (2026-09-11),
      with the optional-tool/display limits recorded above. I do not count
      dependency queries as compiler validation.
- [x] **5.0 module introspection parity.** I derive module name/path,
      public function and struct inventories, unsafe status, and FFI status
      from each original module before import flattening. I emit the complete
      eight-operation introspection surface, test private/nested declarations,
      comments, names, counts and index bounds, and reject ambiguous legacy
      module identities. This does not replace the separate typed-module ABI
      and import-authority work. The source-fact native test passes; both
      compiler candidates execute all eight operations and flag regressions.
      The self-hosted candidate rejects duplicate identities (2026-09-11).
- [x] **5.0 audit defect — boolean C ABI.** Module introspection exposed
      self-hosted `bool` declarations emitted as C `int`, conflicting with the
      seed's C `bool` definitions. I reconcile boolean declaration/field
      mappings and verify generated prototypes and executable behavior.
      `make test-transpiler-externs` passes explicit C `bool` assertions.
      Boolean filter callbacks agree with their generated function signatures.
- [x] **5.0 self-hosted build isolation — in-memory merged source.** I remove
      shared merger files and the unconditional debug dump, reuse my existing
      string accumulator, and lex the result directly. I test overlapping
      source/native compilations with distinct imports, preservation of legacy
      scratch paths, failure cleanup and source-only compilation without a
      writable temporary directory. I rebuild the compiler and rerun native
      shadow, CLI and language-contract gates before claiming isolation.
      I update the import-path regression that still reads the removed fixed
      C filename, and verify the reported private `--keep-c` artifact instead.
      Four isolation tests pass, including confirmed live overlapping source
      and native compiles with distinct executable outputs. The legacy-symlink
      regression overwrites its protected target on the old compiler and
      preserves it after repair. Bootstrap smoke/no-C-seed checks, five import
      tests, nine native-shadow tests, nine CLI tests, five language-claim
      tests and all 28 contract rows pass. The broader quick gate passes
      17 native language programs but fails VM example coverage as recorded
      above; I do not claim a green quick or release gate (2026-09-12).
      Generic-list generation remains separate under the broader item below.
      MAC `task_443e8107d0ff4350999e0d5186a809f1`.
- [ ] **5.0 self-hosted diagnostics — original source provenance.** My
      flattened import stream uses merged line numbers, now labelled with
      the root input path rather than a temporary filename. I retain original
      paths and positions through merging/tokenization and translate lexer,
      parser and type diagnostics, including machine-readable output. I test
      root and nested imports, removed declarations and escaped path bytes.
      MAC `task_3c235c2533a5499093804b26da53801b`.
- [ ] **5.0 audit defect — compiler build isolation.** Concurrent C-seed
      compilations shared `obj/nano_modules/transpiler.o.c`; one compilation
      removed it before the other invoked clang. I isolate intermediate module
      files and publish cache entries atomically, then test concurrent builds.
      The self-hosted merger also uses predictable shared temporary paths;
      I replace those and remove its unconditional merged-source debug dump.
      MAC `task_443e8107d0ff4350999e0d5186a809f1`.
      The C seed also invokes each module C compilation twice: once via
      `popen` for diagnostics and again via `system` for status. I execute
      it once, drain diagnostics, and use that same invocation's exit status.
- [x] **5.0 module build — one invocation, one status.** I execute the C
      compiler once per module, drain all diagnostic output while retaining
      a bounded prefix, and use that invocation's exit status. Three isolated
      integration tests cover success, a first failure that a second call
      would hide, and 32768 diagnostic lines. The old binary fails the
      single-invocation checks; the rebuilt seed passes all three tests.
      Shared intermediate paths and atomic publication remain open above.
- [x] **5.0 module build — private intermediates and atomic objects.** I
      create a private build directory beside each destination object, compile
      into it, and rename only a successful object into the cache. I test
      overlapping module compilations and preserve an existing cached object
      after a compiler writes partial output and fails. This does not complete
      generic-list generation or self-hosted temporary-file isolation.
      Six integration tests pass, including deterministic overlap, partial
      output failure preserving cached bytes, and rejected success without an
      output object. Private directories have mode 0700; successful normal
      builds remove their intermediates. Failure retains only private C for
      diagnosis. The rebuilt C seed also passes module introspection, extern
      declaration selection and parenthesized-parser regressions (2026-09-11).
- [ ] **5.0 module build — command argument boundaries.** My C-seed module
      compiler still concatenates source/include paths and flags into a shell
      command. I must preserve paths containing spaces or shell metacharacters
      as data and reject truncated arguments, with execution tests. Private
      staging and atomic object publication do not repair this boundary.
      I now quote module object/source/include paths and top-level module
      include paths, with executable import tests below. Top-level artifact
      and CLI library paths are repaired below. Runtime-source paths,
      generator invocations, manifest package names used in shell probes, and raw flag
      fragments still require their own boundary audit; this item stays open.
- [x] **5.0 module imports — literal shell paths.** I share one literal
      path-quoting helper between module compilation and top-level include
      construction, reject oversized module commands and dropped flags, and
      keep NANO_CC command-fragment compatibility for configured wrappers.
      Nine module-build tests pass, including executable imports from a
      directory with spaces/apostrophes, literal command-substitution syntax
      that must not create a marker, and a rejected oversized command before
      compiler invocation (2026-09-11). The first run exposed the still-unquoted
      top-level include path after module compilation; both callers are now
      covered. This does not make all compiler commands shell-free.
      Rebuilt C-seed module introspection, extern selection and compiled
      parenthesized-parser regressions also pass.
- [x] **5.0 C output — literal artifact and CLI library paths.** I quote
      the executable and temporary C paths in the final compile command and
      append CLI `-L`/`-l` values as literal words with overflow rejection.
      Fourteen module/build tests pass, including output names and TMPDIR with
      spaces, apostrophes and shell-substitution syntax, plus library names
      and directories with the same characters (2026-09-11). The native
      library test calls a real function from its archive and verifies its
      result; its compile-time shadow checks only the pure module assertion
      because the evaluator cannot execute that linked archive. Marker files
      must remain absent. This does not cover generated-list command paths,
      runtime-source paths or all raw compiler/linker flag fragments.
- [x] **5.0 C link paths — runtime sources and module objects.** Runtime
      sources and generated-wrapper paths now enter the command as quoted
      words, and NanoLang module-object lists quote and deduplicate complete
      arguments while rejecting overflow. Copied compiler installations under
      spaces/apostrophes and shell-substitution paths compile and execute.
      The first expanded suite passed 16 of 18 tests; two filename tests
      failed in metadata C generation below, after checking no shell marker
      was created. With the metadata repair, all 20 tests now pass without
      an expected-failure exemption (2026-09-11).
- [x] **5.0 module metadata — filenames versus C identifiers.** The C
      transpiler previously used raw module filenames in `___module_*` symbols.
      Filenames containing spaces, apostrophes or command-substitution syntax
      produced invalid C even when shell arguments were quoted correctly.
      I now share an injective reserved-prefix/hex encoding across helper
      definitions, C function mapping, extern declarations and serialized
      metadata suffixes. Ordinary suffixes retain their spelling; reserved
      prefix names are encoded too. The 20 build tests pass, including the
      two previous failures, collision cases and an encoded helper call that
      reports the original module name. C-seed module introspection, extern
      selection and parenthesized-parser gates pass; 17 metadata unit tests
      linked against rebuilt objects also pass (2026-09-11). This is tested
      C-backend behavior, not complete self-hosted/VM metadata parity.
- [x] **5.0 metadata — module name/path C text boundaries.** I encode module
      names and paths with fixed-width octal escapes and omit their raw text
      from generated comments. All 255 nonzero byte values survive a compiled
      C round trip under UBSan; a module path containing `*/` compiles and runs.
      All 22 module build regressions pass (2026-09-11). This covers my C-seed
      module name/path metadata, not every serialized field or import syntax.
      Module introspection, extern selection and parenthesized-parser gates
      also pass; I did not run a full bootstrap for this change.
- [ ] **5.0 imports — escaped path parity.** My C-seed lookup previously used
      raw escape spellings; decoding exposed raw filenames in generated `#line`
      directives. Both C-seed boundaries are repaired below. I must carry the
      same path contract through self-hosted parsing and source merging, with
      executable parity tests; C-seed success does not establish that parity.
      My line-based merger also chose the last quote (including comment text)
      and could silently remove malformed module imports. I must scan escaped
      closing quotes and fail dependency collection on invalid quoted paths.
      The first Stage 2 run rejected plain paths because it dropped `break`;
      the repair below passes five driver/helper tests on both Stage 1 and
      Stage 2 after a fresh bootstrap (2026-09-11). The self-hosted runner now
      includes these tests. Full syntax-aware discovery remains open.
- [x] **5.0 self-hosted loop control — frontend nodes.** My lexer previously
      treated `break` as an identifier and emitted no loop-control node, despite
      existing C emitters. I connect break/continue keywords and parser nodes.
      Compiled parser assertions and self-hosted execution of continue, break
      and nested-loop break pass. The production path-helper regression now
      executes correctly after Stage 1 compilation and the rebuilt Stage 2
      passes all five import tests (2026-09-11).
- [ ] **5.0 self-hosted loop control — context validation.** I must reject
      break/continue outside loops in the language frontend and test nested
      function boundaries. C compiler rejection is not a frontend diagnostic.
- [x] **5.0 native match blocks — scalar acceptance.** I follow the approved
      rule: `return` exits the enclosing function; an expression arm yields
      its expression, and a block arm yields its final expression after its
      preceding statements. I replaced the C emitter's return scan and the
      self-hosted emitter's placeholder with ordered statement/value emission.
      I preserve the enclosing return context while checking block statements,
      retain nonlocal return flags in my shadow evaluator, and type C match
      temporaries independently of the enclosing function's return type.
      I also resolve named union-variant fields in my self-hosted checker.
      Five executable cases cover locals, effects, conditional returns,
      string values in integer functions, and return-only arms. Three
      negative cases reject wrong function returns, incompatible arm values,
      and missing final values. `tests/test_match_block_semantics.py` passes
      on the C seed with rebuilt Stage 1 and again with rebuilt Stage 2.
      The combined fixture and migrated match-bindings fixture execute from
      `main`. My full self-hosted runner reports 15 passed, 0 failed, including
      five import-path checks. A serial `make bootstrap3` passes its configured
      gates, including operation without the C seed; native binaries still
      differ. Type-inference, effects, NanoCore and parser-recovery gates pass.
      These are tests, not a compiler correctness proof (2026-09-11).
- [ ] **5.0 match control flow — full backend acceptance.** I still need
      acceptance cases for nonlocal returns nested in operands and call
      arguments, matches whose every arm exits, guarded/exhaustive matches,
      aggregate and resource values, and NanoISA lowering. I must verify
      arm-local scope and escaped ownership beyond directly yielded locals.
      My bootstrap checker saves/restores function return context in sequential
      global state; I must pass explicit context before enabling parallel or
      reentrant checking. The scalar native cases do not complete this work.
- [x] **5.0 self-hosted unary expressions.** I parse bare `not` and unary
      minus at primary precedence, reusing my existing call and binary nodes.
      Compiled parser assertions verify the grouping of `not false and true`.
      The infix fixture now executes its arithmetic, comparison, logic, unary,
      else-if and mixed-syntax assertions from `main`. It passes when compiled
      by rebuilt Stage 1 and Stage 2. Full bootstrap passes; the self-hosted
      suite improves to 13 passing entries and one failure, match bindings,
      with all five import/helper regressions passing (2026-09-11).
- [x] **5.0 parser recovery — failed prefix arguments.** During import-path work
      I used `byte` (a type keyword) as a local name. My C seed diagnosed the
      syntax error but then exited with a bus error while parsing the compiler
      driver. I now reject the minimized malformed declarations without a signal.
      The minimized failure is a prefix argument that cannot parse: the loop
      appends NULL without advancing and doubles storage until signed overflow
      (UBSan, `parse_prefix_op`). Both operator and function-call loops have
      this pattern. My shared argument parser stops on failure/no progress,
      checks allocation growth and frees prior arguments before returning failure.
      `make test-parser-recovery` passes 13 CLI rejection cases and 10 direct
      parser cases with lexer/parser ASan/UBSan instrumentation, including valid
      list growth. Leak detection is disabled; linked runtime objects are not
      instrumented. Module introspection, extern selection and parenthesized
      parser gates also pass (2026-09-11). No full bootstrap was run here.
- [x] **5.0 parser fuzzing — production API alignment.** My legacy
      `tests/fuzzing/fuzz_parser.c` declares a TokenList-based `tokenize` and
      `parse` API, while production uses `Token *tokenize(..., int *)` and
      `parse_program`. I rebuilt the target against current headers and ran a
      bounded corpus; an obsolete harness was not fuzz coverage.
      The harness now includes production headers and shares one input path
      between libFuzzer and its retained AFL++ entry point. `make fuzz-parser-check`
      builds the harness/lexer/parser with coverage, ASan and UBSan and replays
      four seeds. Installed LLVM clang passes replay and a 1000-run campaign
      (`-seed=211 -max_len=4096 -timeout=5 -detect_leaks=0`, 2026-09-11).
      Apple's clang lacks libFuzzer here; `FUZZ_CC` selects the installed LLVM.
      Support/runtime objects are not instrumented; leak freedom, AFL++ build
      validation and unbounded input coverage remain unestablished.
- [ ] **5.0 C lowering — string length result type.** A direct comparison
      between an `int` index and `(str_length line)` lowers to signed `int64_t`
      versus unsigned `strlen`, failing the driver's `-Werror` build. An
      explicitly typed local restores this call site. I must make builtin
      lowering honor the language result type consistently and test it.
- [x] **5.0 C-seed imports — decoded paths and safe line directives.** I decode
      quoted import paths once, preserve unknown escapes as the evaluator does,
      and reject NUL escapes before lookup. I encode filenames in generated
      `#line` directives. All 24 module build regressions pass, including an
      executable import whose filename contains quotes, backslashes, newline,
      tab and carriage return, and NUL rejection before C compilation
      (2026-09-11). Self-hosted and merger parity remain open above.
      Module introspection, extern selection and parenthesized-parser gates
      pass against the rebuilt C seed; no full bootstrap was run here.
- [x] **5.0 generic-list generator — publication boundaries.** I stage both
      generated files privately before replacing either published file, reject
      type names that can escape the output directory, and test overlapping
      generation and executable output. Individual file replacement is not a
      transaction across a header/source pair; compiler-private generation
      directories remain required for distinct type definitions.
      Five tests pass: executable generated C in a quoted/space-containing
      directory, rejected path-like type names, literal substitution of sed
      metacharacters, failure preserving both previous files, and deterministic
      overlapping generators that never expose templates. The checks run in
      `test-impl`; shell syntax and six module-compiler tests pass (2026-09-11).
- [x] **5.0 generic-list runtime — capacity boundaries.** My generator now
      rejects negative/unrepresentable capacities, checks push/insert length
      before addition, and clamps growth before signed or byte-size overflow.
      Zero capacity starts with null data and grows on first insertion. I
      preserve the existing fail-fast API rather than silently lose values.
      Six generator tests pass, including compiled UBSan cases for negative
      capacity, maximum-length push/insert, zero-capacity growth through 33
      values, and a checked large allocation request intercepted without
      allocating gigabytes (2026-09-11). This checks newly generated code,
      not every checked-in runtime list implementation.
- [x] **5.0 runtime lists — checked-in capacity correspondence.** I use
      one internal checked-capacity helper across all 39 retained list
      implementations, preserve their APIs, and support zero-capacity growth.
      Each implementation passes five compiled UBSan scenarios: normal growth,
      negative capacity, maximum-length push/insert, and intercepted large
      growth. All six generator tests also pass. `make test-runtime-lists`
      rebuilds the bootstrap and passes 33 AST-list and six non-AST-list
      API/value tests (2026-09-11). The direct checks run in `test-impl`.
      I removed the orphaned `list_ASTMatchClause` source/header: its struct
      was absent from the schema, it was absent from RUNTIME_SOURCES, and
      the old tests explicitly excluded it. Git preserves that unused pair.
      These checks establish tested capacity behavior, not full heap safety.
- [x] **5.0 runtime lists — string-copy allocation boundaries.** I check
      string duplication before publishing an element or shifting insertion
      slots, and copy replacements before freeing the old value. `set` now
      accepts a value borrowed from itself or a substring without dangling
      access. The API remains fail-fast, not recoverable allocation handling.
      An ASan/UBSan harness intercepts that exit boundary and checks failed
      push/insert/set, unchanged element pointers/content, successful aliased
      updates, ownership transfer on removal/pop and zero remaining tracked
      copies. All 39 list capacity cases also pass. The C seed rebuild and
      normal 33 AST/six non-AST value tests against rebuilt runtime objects
      pass (2026-09-11); I did not rerun the full bootstrap for this change.
- [x] **5.0 audit defect — typed filter dispatch.** My executable callback
      tests exposed float and string arrays routed to the integer filter
      helper. I select the helper from the array element type and retain
      native integer, float, and string callback regressions. The `filter`
      spelling now shares `array_filter` dispatch. Both the C seed and a fresh
      NanoLang compiler candidate compile and execute all 17 language programs,
      including these callbacks (2026-09-11). General collection-expression
      inference remains part of the wider typing audit. My installed stage-two
      binary is unchanged; this is not a passing full `test-quick` gate.
- [ ] **Formal audit — Sail adoption trial.** I record a sourced framework
      decision, run a pinned Sail toolchain on a NanoISA stack/constant slice,
      then extend it to schema-checked decoding and differential VM execution.
      I test truncated encodings, unsupported instructions, stack underflow,
      integer boundaries, branches, and traps before promoting any model to
      an ISA authority. A typechecked smoke model alone does not complete this item.
      `docs/FORMAL_TOOLING_DECISION.md` assesses all five tools and retains Rocq.
      The pinned Sail 0.20.2 stack slice typechecks, generates C, compiles, and
      passes seven smoke assertions via `bash scripts/check_sail_container.sh`
      (2026-09-11). The same runner now checks the schema and matches production
      `isa_decode` on 1,524 deterministic byte sequences; nine schema-drift tests
      pass. The runner also matches NanoVM on 1,140 integer stack cases,
      including 34 underflows and frames with locals (2026-09-11). Broader
      execution semantics and prover exports remain unverified; this trial
      is not a complete ISA model.
- [x] **Formal audit — Sail/Rocq export boundary.** I exercise the pinned
      Sail Rocq backend on my actual stack model, inspect its definitions and
      required support-library imports, then pin that library and compile the
      generated development with Rocq. Generation alone does not complete
      this item; I also need checked model lemmas and assumption reports.
      The first real export fails inside Sail 0.20.2 rewriting a literal-byte
      list pattern (`Cannot infer type of: p0# :: rest`). I retain a
      reproducible export-only command and test a type-explicit representation
      before changing semantics or upgrading the pinned toolchain.
      An explicit decoder input annotation reproduces the same error; I
      reverted that ineffective experiment. MAC
      `task_c692a020a81a4b11bea6d7c99f98689e` tracks export and proof checking.
      A whole-pattern annotation also fails. Separating the opcode byte
      from its payload before matching it now permits actual Rocq export.
      The emitted decoder and executor import SailStdpp.
      `--rocq-check` now provisions exact support-package versions in the
      disposable proof container and attempts compilation, eight stack laws,
      assumption reports, and independent `coqchk`.
      I also make the runner reject missing or axiom-bearing assumption
      reports, reusing the existing NanoCore checker with an explicit Sail
      theorem inventory; printing assumptions alone is not a proof gate.
      The first run compiled both generated files and all eight lemmas,
      printed eight closed reports, and passed independent `coqchk`. A fresh
      report also passed the shared checker. The outer shell failed after a
      live edit of its runner; I require an unchanged-script rerun before
      marking the end-to-end check complete. Twelve proof-gate tests pass.
      The frozen rerun exited zero on 2026-09-11: generated definitions and
      eight lemmas compiled, all eight named reports were closed and accepted
      by the shared checker, and independent `coqchk` succeeded. Decoder
      correctness and VM refinement remain separate, unproved obligations.
- [x] **Formal audit defect — stack-slice underflow.** My unverified VM
      silently accepted insufficient operands for `DUP`, `POP`, and `SWAP`.
      These operations now trap without consuming locals or caller values.
      All 736 NanoVM assertions pass, including underflow regressions with
      empty and singleton operand stacks, locals, and a caller stack prefix.
      Executable Sail agrees on 1,140 bounded integer stack cases; three
      corpus tests check determinism, boundaries, and non-vacuous coverage.
- [x] **Formal audit — Sail frame-extension law.** I prove that whenever
      an instruction succeeds on its operand stack, appending arbitrary
      caller-frame values leaves those values untouched in the result.
      I require compilation against the generated executor, a closed named
      assumption report and independent checking. This is a model theorem,
      not production-VM refinement or a claim about failing instructions.
      The first compilation rejects overloaded `++` as string concatenation
      under Sail's imports. I use explicit `List.app` and require a fresh
      proof-gate run; the theorem remains unchecked until that run succeeds.
      The next run accepts the statement but rejects the case script's use
      of `rest`. I explicitly destruct unit constructor payloads and stack
      shapes in one case split; the revised proof still needs checking.
      Its next run stops before proof compilation because the stdpp archive
      server returns HTTP 429. I retain the revised proof as unverified and
      defer another fetch rather than treating rate limiting as proof evidence.
      The next unchanged-runner attempt exited zero (2026-09-11): generated
      definitions and all nine lemmas compiled, all nine named assumption
      reports were closed and accepted, and independent `coqchk` succeeded.
      Twelve proof-gate, nine schema-drift and three VM-corpus tests also pass.
      This establishes the stated success-only law of the five-instruction
      model, not underflow preservation when a suffix is added or C VM
      refinement. Broader Sail adoption remains open above.
- [x] **Formal audit defect — indexed and rotating stack boundaries.** I
      make `ROT3`, `PICK`, and `ROLL` check frame-relative operands before
      mutation. My tests cover insufficient operands with locals and caller
      prefixes, including indexed depths zero and 65,535, and check that
      every surviving operand remains unchanged. `make test-nanovm` passes
      1,360 assertions (2026-09-11). This does not close the broader audit.
- [x] **Formal audit defect — fixed-effect operand preflight.** I check
      fixed input requirements using existing ISA metadata before dispatch
      of unverified instructions, before handlers can consume locals or caller
      values. I test every operand-free primary instruction with a positive
      fixed requirement on insufficient stacks, including preservation of
      remaining operands. `make test-nanovm` passes 6,801 assertions
      (2026-09-11). These tests rely on the declared metadata; they do not
      prove that every handler agrees with it.
- [x] **Formal audit defect — aggregate input bounds and capture width.**
      I preflight frame-relative inputs for array, struct, union, tuple,
      aggregate and closure construction before allocation or popping.
      Insufficient-stack tests preserve locals, caller prefixes and remaining
      operands. My closure loop now uses a full-width index; tests inspect
      every capture at counts 0, 32,768, 32,769 and 65,535. `make test-nanovm`
      passes 269,537 assertions (mostly capture tags and values), 2026-09-11.
- [x] **Formal audit defect — partial aggregate allocation.** Struct and
      union helpers release their headers and return null when a nonempty
      field allocation fails, without changing allocation statistics. An
      isolated test compiles the actual heap implementation with a failing
      allocator and checks cleanup, legal zero-field allocations and recovery.
      `make test-nanovm` passes, including this failure test (2026-09-11).
- [x] **Formal audit — VM boundary integration regression run.** After my
      stack and call-boundary changes, `make test-nanovirt
      test-frontend-matrix test-scheme test-ml test-actor test-dataflow
      test-object test-shell test-logic` exits zero (2026-09-11). These are
      current frontend laboratory checks, not production runtime guarantees.
- [x] **5.0 audit — post-hardening quick-gate checkpoint.** A fresh
      `make test-quick` exits zero on 2026-09-11 with native sources through
      `a2296993` unchanged during execution. Both bootstrap stages pass smoke
      checks; the installed compiler runs without the C seed. All three
      self-hosted component driver checks and 17 selected language tests pass,
      followed by compiled parser, extern selection, introspection, dependency,
      release-workflow and Forth evidence checks, 280 Forth example cases and
      PTY liveness. The IDE binary compiles; graphical initialization is skipped
      without `xvfb-run`/`timeout`, and interactive GLUT launch is not requested.
      Native stage binaries differ. Canonical NanoISA equality, complete
      release checks and the remaining audit findings are not completed by
      this checkpoint.
- [x] **Formal audit defect — aggregate allocation error propagation.**
      Array, struct, union, tuple and closure constructors reject null before
      consuming operands. My isolated failing-heap test executes these five
      instructions and aggregate pack through the actual VM, checking memory
      errors, preserved caller/locals/integer inputs and unchanged heap size
      and object counts. `make test-nanovm` passes (2026-09-11). This does not
      establish recovery from stack-growth failure or every allocation site.
- [x] **Formal audit defect — stack reserve and embedding entry atomicity.**
      I share an overflow-safe reserve helper between pushes and embedding
      entry points. Entry calls reserve all local slots before mutation and
      reject invalid parameter storage. An isolated actual-VM test checks
      zero capacity, unrepresentable sizes, allocation failure without buffer
      replacement, unchanged direct/invoke entry state and successful retry.
      `make test-nanovm` passes with both allocation-failure suites (2026-09-11).
- [x] **Formal audit defect — internal call-frame reservation.** Direct,
      tail, indirect and linked calls reserve local slots before argument
      moves, callable consumption, frame teardown or local initialization.
      Injected growth failures preserve the stack buffer, caller frame,
      caller values, locals, arguments and indirect callable. My actual-VM
      failure suite and `make test-nanovm` pass (2026-09-11).
- [x] **Formal audit defect — instruction output reservation.** I reserve
      positive net stack growth before dispatch for declared fixed effects,
      aggregate constructors and indexed stack operations. Failure-injection
      tests cover push, duplication, indexed duplication and empty tuple/closure
      construction without operand mutation or heap allocation. Addition on
      a full stack succeeds without reallocation. `make test-nanovm` passes
      (2026-09-11). The check relies on the stated instruction effects and
      does not prove every handler's intermediate stack usage agrees.
- [x] **Formal audit defect — foreign result capacity preflight.** I reserve
      the result slot before producing a foreign-call trap. Four actual-VM
      trap-boundary cases cover zero/one argument and void/non-void results:
      failed required growth traps before dispatch, and reusable argument slots
      need no allocation. The final result push also releases the result on
      failure; that defensive path is not directly fault-injected here.
      `make test-nanovm` passes (2026-09-11). The tests invoke no host function.
- [x] **Formal audit defect — movable embedding arguments.** `vm_invoke`
      snapshots borrowed arguments before stack growth and accepts complete
      live stack slices. I reject stack-backed result destinations and stack
      aliases in the ownership-transferring low-level call; `vm.h` documents
      the distinction. An actual-VM test forces relocation with a string
      argument, checks original/result reference counts, and checks failed
      growth and invalid slices without ownership changes. `make test-nanovm`
      passes (2026-09-11).
- [x] **Formal audit defect — explicit array append failure.** Array append
      returns a boolean and retains its input only on success. Production VM,
      co-process, FFI and hashmap-array builders now check failure and discard
      partial results. A real-VM injected growth failure preserves a boxed
      string array and its references; retry succeeds and cleanup returns to
      the baseline live-object count. NanoVM, 32 protocol, eight protocol-fuzz
      and 18 FFI tests pass (2026-09-11). Not every builder failure branch has
      a dedicated injection test. Typed arithmetic storage remains separate.
- [x] **Formal audit defect — typed vector arithmetic.** I replaced eight
      duplicated array/broadcast paths with one borrowed-input, owned-result
      implementation. I validate participating element pairs before allocation,
      preserve packed int/float results and boxed string/mixed results, release
      temporary strings after insertion, and keep scalar operand order and the
      shorter-array rule. Vector integer operations wrap without signed C
      overflow; division is total at zero and `INT64_MIN / -1`.
      Tests cover both opcode families, three operand shapes, five numeric
      representations, strings and ownership, unsupported element pairs,
      integer boundaries, empty results and injected allocation failure with
      successful retry. `make test-nanovm` passes 272,125 assertions plus the
      allocation-failure suites; all 18 FFI tests pass (2026-09-11).
      This is tested implementation behavior, not a VM refinement proof.
- [ ] **Formal audit — allocation failure boundaries.** I repaired explicit
      array append failure and migrated production callers. Nested decode
      ownership passes 32 protocol tests, including truncated nested messages,
      single-owner decoded references and collection back to baseline.
      Typed vector storage and temporary string ownership are repaired above.
      Other constructors and handlers still need allocation-result checks,
      preflight or ownership-safe failure propagation. Embedding arguments
      follow the relocation/aliasing contract above. I require deterministic
      failure tests before claiming recoverable memory exhaustion. I track
      this with runtime boundary task
      `task_0ba46839aee94135aaa99a9b7c207499`.
- [x] **Formal audit defect — scalar arithmetic and string boundaries.**
      Dynamic and typed integer ADD/SUB/MUL now use unsigned intermediates
      for modulo-2^64 arithmetic. My prior assumption that the typed handlers
      already avoided signed C overflow was wrong. Thirty scalar boundary
      cases check both opcode families, including total division/remainder.
      String creation/concatenation rejects unrepresentable allocation sizes;
      concatenation checks length before payload access and avoids malloc(0).
      Both scalar concatenation opcodes propagate allocation failure. Failed
      initial intern-table allocation/insertion remains unpublished and can
      recover on retry. Tests cover synthetic overflowing lengths without
      payloads, empty strings, each temporary/result allocation, partial
      vector string results, input references and baseline object recovery.
      NanoVM passes 272,215 assertions plus allocation-failure suites; all
      18 FFI and 32 protocol tests pass (2026-09-11). The VM suite also passes
      with production vm.c/heap.c and its test driver compiled using
      `-fsanitize=undefined -fno-sanitize-recover=undefined` at O3; other linked
      objects were not instrumented. These tests do not establish complete
      runtime safety or correspondence with the formal arithmetic model.
- [x] **Formal audit defect — call argument boundaries.** Direct, tail,
      linked-module, indirect and foreign calls check frame-relative operands
      before consuming arguments or changing frames. Indirect calls retain
      the callable on the operand stack until validation succeeds. Forty
      malformed-call cases check insufficient arguments with locals and caller
      prefixes, unchanged frame counts and preserved arguments/callables.
      `make test-nanovm` passes 269,805 assertions (2026-09-11); most assertions
      remain capture-value checks, not distinct call scenarios.
- [x] **Formal audit defect — linked call signature correspondence.** I use
      the existing linked verifier, not separate module verification, to
      establish the unchecked-path flag. Linked calls also check declared
      arity and result count against the target before frame mutation.
      Regressions use individually valid modules with mismatched linked
      signatures and check preserved operands and frames. I corrected an
      existing string-call fixture that declared zero results instead of one.
      `make test-nanovm` and `make test-nanoisa` pass (2026-09-11).
- [ ] **Formal audit — remaining unchecked boundaries.** Dynamic-effect call
      handlers still require their own count and shape checks. I audit the
      remaining checked stack handlers, which still
      return void or skip operations on some insufficient-operand paths.
      I establish a consistent failure contract, preserve frame boundaries,
      and test malformed modules through the embedding API before claiming
      complete runtime underflow protection. MAC
      `task_0ba46839aee94135aaa99a9b7c207499`.
- [x] **Formal audit defect — missing evaluator theorem.** My formal README
      advertised `eval_fn_sound` while `EvalFn.v` contained only selected case
      lemmas. I implemented all aggregate cases and strong fuel induction,
      connecting the earlier control-flow, binding, and operator proofs. The
      general theorem now covers every expression constructor with no recursive
      soundness premise. Twenty-two regression examples and all eleven libraries
      pass `bash scripts/check_proofs_container.sh` on pinned Rocq 9.0.1
      (2026-09-11), with 38 closed assumption reports and independent library
      checking. Production correspondence, completeness, and extraction
      correctness are not consequences of this theorem. MAC
      `task_2b291a75ca2840519d47e08bf991c021`.
- [x] **Formal audit defect — eager reference logical operators.**
      `EvalFn.v` evaluated both operands of `OpAnd` and `OpOr`, unlike
      `Semantics.v`. I restored short-circuit evaluation and added seven
      regression examples in `EvalFnTests.v`, covering truth tables, skipped
      effects and stuck operands, required effects, left-side effects, and
      operand types. `eval_fn_and_short` / `eval_fn_or_short` prove skipped-RHS
      behavior for arbitrary expressions. `eval_fn_sound_logic` proves operator
      soundness conditional on recursive soundness. The full pinned proof gate
      passes on 2026-09-11, including 22 closed assumption reports and independent
      checking of all eleven libraries. General evaluator soundness was completed
      separately in the item above.
- [x] **Formal audit defect — clean proof build fails.** A fresh Rocq 9.0.1
      build found an unhandled empty-list case in `tuple_nth_type`, now
      repaired without changing its statement; `Soundness.v` compiles.
      `Progress.v` now compiles after correcting the tuple-tail statement,
      proving tuple-step target shape, and repairing tuple-index induction.
      `Determinism.v` also compiles. I removed a stale empty-tuple tactic in
      `Equivalence.v`. I restored omitted tuple cases in value transfer,
      substitution compatibility, step simulation, symmetry, and transitivity;
      I extended strong expression induction and substitution identities to
      tuples, repaired tuple closure proofs, and used related elements for
      tuple-index simulation. I repaired value destructuring in `EvalFn.v`
      (conditional and string-index cases omitted the tuple constructor) and
      simplification/rewrite failures in `Exhaustiveness.v`'s wildcard and
      complete-or-pattern lemmas, previously excluded from the build.
      On 2026-09-11, `bash scripts/check_proofs_container.sh` passes: all nine
      proof modules plus `Assumptions.v` compile, all 19 named reports are
      closed under the global context, and `rocqchk` independently checks the
      compiled libraries and dependencies. General evaluator soundness was later
      completed separately; production implementation correspondence remains open.
- [x] **Formal audit tooling — checker command.** The pinned Rocq 9.0.1
      image rejects `rocq chk`; its `rocq check` launcher fails to execute
      the installed checker. I invoke `rocqchk` directly in the container
      wrapper and documented command. The full gate passes on 2026-09-11.
- [x] **Formal audit — enforce the proof gate.** I add an unfiltered PR/push
      CI job using the pinned proof image, enforce the advertised theorem
      types, and reject non-closed or missing assumption reports. I test the
      gate's failure paths as well as the complete proof build. Eleven negative/
      positive gate tests pass; the pinned build and independent checker pass
      all twelve libraries and 43 closed reports on 2026-09-11. GitHub run
      `34591153105` also passed in 2m22s. It identified checkout v4's deprecated
      Node 20 runtime; I pin checkout v6 (Node 24) for the new job. Repository
      branch-protection requirements remain a separate release-control check.
- [ ] **5.1 audit — native runtime services.** I run a useful native
      NanoLang service in a separate worker through typed NSI calls, with
      scoped capabilities, restart supervision, tracing, and module packaging.
      I expose mailboxes, monitoring, links, supervision, and upgrade behavior
      through native NanoLang runtime APIs; laboratory frontends remain tests.
      Acceptance exercises real operations, worker death, recovery, denied
      authority, bounded queues, and version compatibility end to end.
- [ ] **5.1 audit — FFI authority.** I resolve declared typed imports by
      module identity, remove ambient symbol fallback from the secure path,
      validate manifests, and isolate untrusted native code. I test symbol
      collisions, undeclared imports, unload lifetime, and crash containment.
- [ ] **5.1 audit — measured ergonomics.** I benchmark human authoring and
      LLM generation/repair with representative programs and recorded compiler
      diagnostics. I use correctness, repair rounds, and author feedback to
      improve imports, operator grouping, shadow policy, and standard-library
      consistency; I publish migrations for changed accepted syntax.
- [x] **5.0 / self-hosted lexer physical string positions.** I account for
      newlines consumed by plain and interpolated strings before assigning
      later tokens to merged modules. A Stage 2 regression compiles and runs
      multiline literals across an import boundary.
      `tests/test_selfhost_module_bindings.py`.
- [ ] **5.1 / Phase 20.** NanoISA-only compilation. Verified `.nvm` is
      the only compiler product. Native AOT does not embed `nano_vm`.
      Public GitHub Release `v5.1.0` after this phase closes.
      `docs/NANOISA_ONLY.md`.

- [ ] **5.1 audit — enforceable release evidence.** I require successful
      CI and review before merging, make lint and proof gates blocking, pin
      build dependencies and Actions, and publish reproducible artifacts with
      signed checksums, SBOMs, and provenance. I remove destructive release
      synchronization and predictable temporary files, reconcile task evidence,
      and verify clean installation and rollback before the public release.
      My [3508bb20 diagnostic snapshot](evidence/release-integration-3508bb20.json)
      records three failing unit targets, one authoring-test skip and twelve
      open PRs requiring reconciliation. It is not the clean/full release gate
      or a completed branch review. MAC `task_cffdafd16e641ac417ccfddb962534b9`.

- [x] I made the 3.5 benchmark workloads execute successfully on NanoVM and
  recorded 20 repeatable profiles for NanoLang execution, allocation, direct and
  indirect calls, FFI, and the current Forth interpreter. Compiled Forth, its
  compiler, and Forth exceptions remain Phase 13 work.
- [x] I added optional NanoVM opcode instrumentation with one-time process
  configuration, a single hot-path boolean guard, value and FFI diagnostics,
  regression tests, and an LLM troubleshooting skill.
- [x] I unified generated-C profiling behind an optional runtime hook: I read
  `NANO_PROFILE` once at process startup, cache its boolean, keep disabled hooks
  cheap, preserve timing and flamegraph evidence, and test enabled and disabled
  generated executables without rebuilding.
- [x] I ship a windowed SDL editor (`examples/emacs/nano_emacs.nano`) that uses
  `modules/ui_widgets` and the line buffer in `examples/lib/source_editor.nano`,
  with Emacs-shaped panes, a minibuffer, and C-x / M-x keys.
- [x] I evaluate NanoLang in that editor through a persistent tree-walker
  session (`modules/nano_eval`) with host primitives `ed_message`, `ed_insert`,
  `ed_buffer_string`, `ed_point`, `ed_goto_char`, `ed_find_file`,
  `ed_save_buffer`, `ed_split_window`, and `ed_other_window`.
- [x] I test keymap prefix dispatch, buffer point/insert, `eval_string` of
  `(+ 1 2)`, surviving `defn`, queued `ed_message`, and I compile the editor
  with a timeout. I do not claim GNU Emacs compatibility.
- [x] **4.1 / Phase 13 (primary).** I implement typed Forth import declarations
  that lower to `NvmImportEntry` and `OP_CALL_EXTERN`.
- [x] **4.1 / Phase 13 (primary).** I reject FFI signatures the active ABI
  cannot call correctly instead of guessing, and I restart an isolated FFI
  co-process after dynamic import-table mutation.
- [x] **4.1 / Phase 13 (primary).** I make `SEE` disassemble the actual compiled
  NanoISA function and describe imported words.
- [x] **4.1 / Phase 13 (primary).** I close Forth kernel defects found during
  Core (Phase 13 subsection of the same name). I do not mark Core complete while
  those remain open.
- [x] **4.1 / Phase 13 (primary).** I implement and test Forth 2012 Core on the
  NanoISA session. Jackson Core evidence files pass under `make test-forth-core`
  through C `REFILL`. Remaining word sets, `pi.fs`, and a Standard System
  label stay in Phase 13. Locals-on-`THROW` waits on recursive Locals. Passing
  a suite is evidence. I still do not claim Core as a banner.
- [x] Core-suite fan-out 1/4 (parallel): I vendor Gerry Jackson
  `forth2012-test-suite` **v0.15.0** (`9773f84dd12390f342d37195da8848b04e1f4a23`)
  into `tests/forth/vendor/gerryjackson/`, keep every Johns Hopkins and Jackson
  notice, set `vendor: true` in `tests/forth/pins.json` for that suite only, and
  do not copy optional-word-set files as Core evidence.
- [x] Core-suite fan-out 2/4 (parallel): I write a per-file license inventory of
  the pinned forth200x snapshot (`91f1ed9c756aac27f57e939c270b5f2c84262427`) in
  a new `docs/FORTH_200X_INVENTORY.md`. I do not vendor that tree.
- [x] Core-suite fan-out 3/4 (parallel): I add `make test-forth-jackson` that
  runs `bin/nano_forth` / `bin/forth` against Jackson Core via `INCLUDED` (or
  records a precise INCLUDE/file-access gap). I update `docs/FORTH_2012.md` so
  it no longer says there is no NanoISA Forth. I do not run optional word sets
  in this target. I do not claim Core if any Core case fails.
- [x] Core-suite fan-out 4/4 (parallel): I publish a Core word coverage matrix
  (`docs/FORTH_CORE_COVERAGE.md`) of Forth 2012 Core names against the NanoISA
  session, with tested / missing / ambiguous for each word. This is evidence,
  not a Core pass.
- [x] **4.1 / Phase 13 (primary).** I load Jackson Core (`prelimtest.fth`,
  `tester.fr`, `core.fr`, `coreplustest.fth`) through C file-source `REFILL`,
  not Forth `INCLUDED`. `#ERRS` and `#ERRORS` are 0. I do not claim File
  Access. Passing those files is evidence. I still do not claim Core as a
  banner. (`task_d9719aedab784a01928eec699191fa82`)
- [x] Parallel to 4.1 (does not wait on the Core gate): I close the interpreter
  example failures in `docs/KNOWN_LIMITATIONS.md`. `bin/nano` loads a shipped or
  built-in word list for `nl_random_sentence.nano`, uses a smaller workload for
  the sieve and Game of Life when `NANO_INTERPRETER` is set, and skips the
  libdispatch examples with a printed reason when the host has no GCD or when
  the tree-walker cannot run GCD callbacks. I test those files under `bin/nano`
  with a timeout.
- [x] Parallel to 4.1 (does not wait on the Core gate): I wire `--bench` to the
  tree-walker, add `examples/bench_sample.nano` with two zero-parameter
  workloads, emit non-zero ns/op, and enable the CI `bench` job without
  `if: false` or `|| true`. The job fails on a benchmark error or a zero
  measurement; it does not claim a 2× baseline comparison until I store one.
- [x] Side-quest after 4.1: I isolate the SDL editor walker in
  `bin/nano_emacs_worker` with a length-prefixed pipe protocol (create/destroy,
  bind buffer, eval string, drain `ed_*` commands, crash detection and restart).
  I do not overload `COP_MSG_FFI_REQ` as eval. The frame does not `dlopen` the
  interpreter. (`make nano_emacs_worker`, `docs/NANO_EMACS.md`,
  `make test-nano-emacs-worker`)
- [x] Side-quest after 4.1: I keep the editor's `nano_eval_*` C API and make
  the bridge an RPC client of that worker. A walker crash echoes an error,
  restarts the worker, and keeps buffers. The child never re-enters SDL.
  (`modules/nano_eval/nano_eval_bridge.c`, `make test-nano-emacs-worker`)
- [x] Side-quest after 4.1: I add freeze-defun (`C-x C-z` / `M-x freeze-defun`):
  extract the current top-level `fn`, timeout-compile it to `.nvm`, and run
  `nano_vm` as a grandchild. `C-x C-e` stays walker eval. Frozen v1 is pure
  (result or error echo only; no `ed_*` inside the `.nvm`). A freeze-child
  crash does not kill the frame. (`modules/nano_eval/nano_eval_freeze.c`,
  `examples/emacs/emacs_keys.nano`, `make test-nano-emacs-worker`)
- [x] Side-quest after 4.1: I test killing the worker mid-eval (parent survives
  and can restart), freeze of a pure function, and timeout compile of
  `nano_emacs`. In-process `make test-nano-eval` remains. I document a live
  editor plus an isolated worker. I do not claim GNU Emacs compatibility.
  Capability-supervised isolation of the same children is 4.4 work, not this
  cut. (`make test-nano-emacs-worker`, `make test-nano-eval`,
  `make test-nano-emacs`, `docs/NANO_EMACS.md`)
  (`task_d36c571fef9a4dcca2d2a1a845040b27`)
- [x] freeze-frame isolation test reads the eval result before destroy so the
  string is not a dangling pointer (`tests/test_nano_emacs_worker.c`).
- [x] I recorded the 5.1 One IR rewrite (`docs/NANOISA_ONLY.md`, Phase 20;
      `task_87bcff8dad43407884c4dc9e06837f98`). Frontends twice, one verified
      `.nvm`, translators as host tools, bootstrap that compares `.nvm`,
      native AOT that does not embed `nano_vm`. I do not execute that rewrite
      in 4.x.
- [x] **4.2 / Phase 15 (primary).** I resolve process locale:
      `--locale` > `NANO_LOCALE` > `LC_ALL` > `LANG` > `en`. Invalid CLI
      and `NANO_LOCALE` fail closed. Garbage POSIX env is skipped.
      `--print-locale` prints tag, language, script, region, variant,
      direction, encoding, collation, source, and fallback. Diagnostics
      still render English. I do not call the system internationalized.
      Catalogs and translated guides are later items in this phase.
      (`task_63a472a3ff564d438532e89692267b66`)
- [x] **4.2 / Phase 15.** I reject invalid UTF-8 in `.nano` source (`CSRC01`)
      and keep pipeline compiler diagnostics as stable IDs (`src/diag_id.c`,
      `src_nano/compiler/diagnostics.nano`). English is a lookup. Typechecker
      `E003` reuse, logs, catalogs, and other UTF-8 boundaries are not this
      item. I do not call the system internationalized.
      (`task_ec1b3368f4684ffcb25ecfe489b67639`)
- [x] **4.2 / Phase 15.** UTF-8 at JSON/TOON emit (invalid becomes
      `<invalid UTF-8>`), `module.json` fail closed, docgen source/module
      name fail closed. Identifiers are ASCII `[A-Za-z_][A-Za-z0-9_]*`;
      non-ASCII fails closed (`L0003`). Typechecker titles that went
      through `emit_context_error` have unique `E001`–`E034`
      (`--json-errors` emits `E024` for an undefined variable). Unicode
      FFI: grapheme, NFC/NFD, case conversion, casefold, display width
      (`make test-unicode-ffi`). Logs, catalogs, translated guides, and
      remaining locale-ctype surfaces are not this item. I do not call
      the system internationalized.
- [x] **4.2 / Phase 15.** UTF-8 message catalogs for six languages, log
      event ids, locale-aware user-guide editions, translated nav titles,
      and LTR-isolated code fences on `dir=rtl` pages. JSON/TOON stay
      English. I do not call the system internationalized.
      (`task_232233a85c9f445ebf7ce93eddec48cc`)
- [x] **4.3 / Phase 16.** I define NSI v0 with stable interface, method, type,
      error, and capability identifiers (`schema/nsi/`, `src/nsi.c`,
      `docs/NSI.md`, `make test-nsi`). Direction, ownership, generation,
      and module migration are later items.
      (`task_3581c13d862e43f0a76a52e86185b92e`)
- [x] **4.3 / Phase 16.** I describe parameter direction, ownership, borrowing,
      transfer, lifetime, mutability, optionality, and streaming in NSI v0
      (`make test-nsi`, `docs/NSI.md`). Records and generation are later.
      (`task_ab516c43dbdf4915b76d178409b2e56b`)
- [x] **4.3 / Phase 16.** I support records, variants, arrays, strings, binary,
      resources, callbacks, asynchronous results, and versioned errors
      (`schema/nsi/examples/types.nsi.json`, `make test-nsi`). Compatibility
      and generation are later.
      (`task_d9e85990a21944fe836852591361f2ff`)
- [x] **4.3 / Phase 16.** I define NSI document compatibility
      (`nl_nsi_compat`). Adding a method is compatible; removing one is
      breaking. Wire frames are not in v0.
      (`task_094c94be861641f1a601a1fb152f3fb7`)
- [x] **4.3 / Phase 16.** I reject omitted `params`, omitted type `kind`,
      `opaque`, unknown keys such as `c_type`, and unresolved types
      (`make test-nsi`). I generate NanoLang, Forth, Python, Rust, and C++
      bindings plus dispatch, frames, validation, docs, mocks, compatibility
      comments, and NanoISA imports (`make test-nsi-gen`).
      (`task_d758f4e8215042889e48821b329aa870`)
- [x] **4.3 / Phase 16.** I extend module manifests with an `nsi` block,
      keep `module.json` as build metadata, and inventory every current
      module (`schema/nsi/inventory.json`, `make test-nsi-manifest`).
      (`task_d758f4e8215042889e48821b329aa870`)
- [x] **4.3 / Phase 16.** I invoke by method id over in-process, mock, and
      local-process adapters with frames, hello, backpressure, idempotence,
      auth, and typed handles (`make test-nsi-runtime`, `docs/NSI_TCB.md`).
      I do not claim a service fabric.
      (`task_d758f4e8215042889e48821b329aa870`)
- [x] **4.4 / Phase 17.** Unforgeable capabilities, shared-memory data plane,
      and per-service budgets (`src/nsi_cap.c`, `src/nsi_shm.c`,
      `make test-nsi-cap test-nsi-shm`).
      (`task_7afc6b4fc32546459fd9c16f83b3d4d8`)
- [x] **4.4 / Phase 18.** POSIX fabric, supervisor, scoped service migration,
      remote cap denial, and the editor as a fabric client of walker/freeze
      (`src/nsi_fabric.c`, `docs/NSI_FABRIC.md`, `make test-nsi-fabric`).
      I do not claim a kernel, GNU Emacs compatibility, or a CUDA/CPython wrap.
      The 4.1 astronaut dedicated-pipe worker is `bin/nano_emacs_worker`
      (`docs/NANO_EMACS.md`); 4.4 keeps fabric stand-ins for walker/freeze.
      (`task_7afc6b4fc32546459fd9c16f83b3d4d8`)

## Release Map

I use releases as integration boundaries, not date promises. A release closes
only when its checked deliverables, quality gates, and documentation agree.
Patch releases may ship completed fixes without changing this dependency order.

| Release | Theme | Required outcome |
| --- | --- | --- |
| **3.5** | NanoISA measurement and cleanup | I have repeatable profiles, generated ISA metadata, typed scalar and aggregate operations, side-table debug data, and no active direct LLVM/Wasm backend matrix. |
| **4.0** | NanoISA v2 and NanoVM v2 | I have a versioned, verified, compositional IR; predecoded execution; regular calls, memory, traps, layouts, ownership, modules, and measured performance. |
| **4.1** | Nano Forth | I compile Forth words to NanoISA, implement Forth 2012 Core and optional word sets, run pinned conformance suites, and use the same typed service/import boundary as NanoLang. |
| **4.2** | International Nano platform | I provide language-neutral UTF-8 diagnostics and logs plus English, Mandarin Chinese, Hindi, Spanish, Modern Standard Arabic, and French guide editions. |
| **4.3** | Service interfaces and module migration | I turn module manifests into versioned service contracts and generate clients, servers, wire schemas, policy declarations, compatibility tests, and documentation. |
| **4.4** | Capability service fabric | I run modules as supervised least-privilege services with typed capabilities, asynchronous IPC, shared-memory bulk transfer, quotas, cancellation, and restart-safe handles. The SDL editor's walker and freeze-ISA children become first clients of that fabric (option C); the dedicated-pipe astronaut in the active queue is the earlier isolation cut, not the fabric. |
| **4.5** | Effects, policy, and replay | I derive deployment policy from effects, record nondeterministic traps, replay executions deterministically, inject failures, and audit service interactions. |
| **4.6** | Multi-language laboratory | I validate NanoISA with bounded Scheme, ML, actor, dataflow, object, shell, and logic frontends, each chosen to test a distinct semantic pressure. |
| **5.1** | One IR: NanoISA-only compilation | I emit verified `.nvm` as the only compiler product. C11 AOT, LLVM, Wasm, and GPU targets are translators of that module. Self-host proof compares `.nvm`. Native binaries do not embed `nano_vm`. Contract: `docs/NANOISA_ONLY.md`. |
| **6.0** | Nano operating environment | I package signed services, startup graphs, upgrades, rollback, health monitoring, and kernel adapters into a complete operating environment. Linux, 5BSD, seL4, and other kernels remain interchangeable substrates below the service ABI. |

My 3.5 release presentation is [NanoLang 3.5](RELEASE_3.5.md). It records my
shipped foundation, verification evidence, and the boundary where my 4.0 work
begins.

Release dependencies:

```text
3.5 measurement and semantic cleanup
 |
 v
4.0 NanoISA/NanoVM v2
 |-------------------------|
 v                         v
4.1 Nano Forth        4.2 internationalization
 |                         |
 +------------+------------+
              v
4.3 service interfaces and module migration
              |
              v
4.4 capability service fabric
              |
              v
4.5 effects, policy, record/replay
              |
              v
4.6 multi-language laboratory
              |
              v
5.1 One IR (NanoISA-only compilation; docs/NANOISA_ONLY.md)
              |
              v
6.0 operating environment and kernel adapters
```

### Phase 12 - NanoISA v2 (3.5 foundation complete, 4.0 completion)

Goal: I will make NanoISA a regular, compositional, verified instruction set
for NanoLang, Forth, and future frontends. My portable bytecode will remain
readable. Verification and instantiation may translate it into a faster private
form, including measured superinstructions.

I completed the 3.5 measurement and cleanup foundation. The unchecked items in
this phase are now explicitly scoped to 4.0 NanoISA/NanoVM v2 and remain open
until their implementation and evidence exist.

The following unchecked groups are 4.0 completion work: portable ISA
regularization, verified and optimized dispatch IR, runtime representation,
verifier and safety, module format and tools, FFI and traps, and their related
documentation and acceptance evidence. I will not count them against the 3.5
measurement-and-cleanup release.

Workflow and evidence:
- [x] I wrote the initial Forth-on-NanoISA architecture contract in `docs/superpowers/specs/2026-08-30-ans-forth-nanoisa-design.md`.
- [x] I added persistent `vm_invoke`, latest-function lookup, and incremental function verification on PR #115.
- [x] I measured static NanoVirt opcode and instruction-sequence frequencies across the repository.
- [x] I added a reproducible NanoISA benchmark and profiling harness before changing execution architecture.
- [x] I record opcode, pair, and triple frequencies, retired instructions, branches, call kinds, stack depths, traps, retain/release traffic, allocations, and FFI byte/latency counters.
- [x] I benchmarked NanoLang execution, allocation, direct and indirect calls, FFI, and the current Forth interpreter with 20 samples per workload; compiled Forth, its compiler, and Forth exceptions remain blocked on the Phase 13 runtime.
- [x] I publish benchmark summaries with hardware, OS, compiler, commit, distributions, retired instructions, and normalized costs.
- [x] I require semantic equivalence and full quality gates for every accepted optimization in `docs/NANOISA_OPTIMIZATION_POLICY.md`.

Portable ISA design:
- [x] I defined `spec/nanoisa.yaml` as the source of truth and generate the active opcode metadata plus v2 stack and ownership metadata from it.
- [x] I use a regular local/stack hybrid: indexed locals for named state and an operand stack for expression evaluation.
- [x] I removed fictional architectural registers from the active NanoISA documentation and v2 schema.
- [x] I give each portable instruction one comprehensible meaning and keep operand forms symmetric.
- [x] NanoVirt emits explicit signed integer, floating, comparison, boolean, string-concatenation, and array-arithmetic operations; legacy polymorphic scalar operations remain assembler compatibility instructions but no frontend emits them.
- [x] I added signed and unsigned division, remainder, comparison, shifts, carry, borrow, and wide multiplication primitives required by Forth double cells.
- [x] I added coherent indexed `PICK` and `ROLL` operations alongside the basic stack operations.
- [x] I added byte-addressed little-endian memory loads and stores at 8, 16, 32, and 64 bits; unaligned access is explicitly supported.
- [x] I replaced NanoVirt's language-specific struct, tuple, and union lowering with regular layout-driven `AGG_PACK`, `AGG_GET`, `AGG_SET`, and `AGG_TAG` operations.
- [x] I separated direct function references and heap closures into unambiguous value tags and constructors.
- [x] I regularize direct, indirect, tail, imported, and linked calls around verified signatures: the NVM verifier checks direct/tail call targets, tail-call result signatures, imported-call signatures (return and parameter type tags), and keeps linked module calls in the same verified taxonomy.
- [x] I resolve separate-module calls to callable handles during linking rather than carry module/function pairs through dispatch.
- [x] I replaced special print, assert, and host operations with typed traps in the NanoISA v2 `trap` family so every side effect is one composable instruction with explicit stack effects and ownership.
- [x] I moved trimming, case conversion, splitting, replacement, formatting, parsing, and collection algorithms out of the portable ISA into runtime libraries; the v2 `instruction_families` expose only primitive string and aggregate operations, and `spec/nanoisa.yaml`'s `runtime_library_algorithms` records the moved functions with a schema-test drift guard (`tests/test_nanoisa_schema.py`).
- [x] I retain only primitive string and aggregate operations justified by representation or measured cost, and classified each string and aggregate opcode in `docs/superpowers/specs/2026-09-01-nanoisa-primitive-string-aggregate-ops.md`.
- [x] I added compact constants, short local forms, and compact general operands to the v2 schema as encoding-only aliases of canonical instructions, so assembly stays regular.
- [x] I defined a clean extended-opcode space in `spec/nanoisa.yaml`: the primary plane holds one-byte identifiers `0x00..0xfe`, `0xff` is a reserved extension prefix that escapes into a separate 256-entry extended plane, and `NANOISA_PRIMARY_OPCODE_LIMIT` is an exclusive range bound rather than an opcode count.

Execution architecture:
- [x] I separated compact serialized bytecode, verified instruction IR, and optimized dispatch IR.
- [x] I decode each function once and dispatch predecoded instructions rather than call the generic decoder for every retired instruction.
- [x] I build instruction-boundary maps and resolve branches plus direct and tail calls during instantiation; layouts, constants, globals, imports, and linked callable handles remain.
- [x] Computed-goto dispatch ships where the compiler supports labels as values, with the portable `switch` retained as the fallback and selectable via `-DNANO_NO_COMPUTED_GOTO`. Measured at 20 samples of 2000 iterations: `nl_forth_interpreter` improves **4.2%** against a 1.0-1.6% noise band, no workload regresses, and both workload groups are preserved or improved -- so it clears the optimization policy rather than being adopted on principle. Forth is where it shows because a Forth interpreter written in NanoISA is a dispatch loop inside a dispatch loop, which is the shape the technique is for. The 161 handlers are written once and reached through `VM_CASE`/`VM_NEXT` macros, so the two strategies cannot drift in what an instruction does; the threaded build has no loop around the handlers, which turns a `break` left behind by the conversion into a compile error rather than a silent exit (it caught two). `test-dispatch-equivalence` runs all 152 programs through both builds and compares output, and a schema test compares the label table against the handler labels. [NANOISA_MEASUREMENTS.md](NANOISA_MEASUREMENTS.md)
- [x] I moved generated source locations entirely to side tables and removed executable `DEBUG_LINE` instructions from NanoVirt output.
- [x] I made `--strip-debug` remove all generated runtime debug cost by stripping the side table from code that contains no debug opcodes.
- [x] I removed generated `PUSH_VOID; POP`, unreachable `RET; JMP`, and statements after terminating control flow from NanoVirt lowering.
- [x] I added direct tail-call lowering and frame-replacement execution with verifier and runtime signature checks.
- [x] I add profile-selected private superinstructions in the optimized dispatch IR (`src/nanovm/vm_dispatch.c` `VmDispatchProfile` and `VmSuperOp`): a fusion runs only when a profile opts it in, the fused step lives entirely in representation 3 with no portable opcode, no serialized-bytecode or verified-IR footprint, and no frontend bookkeeping, and it preserves the byte-addressed `ip` contract; `docs/NANOISA_OPTIMIZATION_POLICY.md` records the acceptance policy and `tests/nanovm/test_vm.c` proves fused and unfused projections return identical results and never rewrite a branch target.
- [x] I initially evaluated local-field load, local increment, compare-branch, union-tag branch, and tail-call fusions in `docs/superpowers/specs/2026-09-01-nanoisa-fusion-initial-evaluation.md`: each candidate is located in the predecoded dispatch IR (`src/nanovm/vm_dispatch.c`) and the lowering that generates it (`src/nanovirt/codegen.c`), ranked by expected win against correctness cost, and gated on the frequency and measurement floors of `docs/NANOISA_OPTIMIZATION_POLICY.md`; no fusion is accepted yet.
- [x] The mechanism ships and nothing is enabled. `VmDispatchProfile` carries one opt-in flag per candidate fusion, every flag defaults off, and an unconfigured VM runs the plain verified stream; the bar is 1% of retired baseline instructions in a maintained workload. No candidate has been measured over it, so none is on by default -- which is the policy working rather than work left undone, since the mechanism exists precisely so a fusion can be evaluated without shipping it.

Runtime representation:
- [x] Measured and declined for 4.0. The prize is bounded by what tag handling costs in the hot stack, and the workloads that could show it (`nl_hashmap_word_count`, `nl_array_complete`) are the ones whose run-to-run noise is 5-10% because they finish in under 20 microseconds -- at that band a change under 10% is invisible. The cost side is not small: every stack accessor, the verifier's depth model, the co-process serializer and the FFI marshaller. Declined on the same grounds as computed goto: the evidence to accept it does not exist, and making it measurable means longer-running workloads rather than a different opinion. [NANOISA_MEASUREMENTS.md](NANOISA_MEASUREMENTS.md)
- [x] I dynamically size globals from serialized declarations instead of embedding 4,096 values in every VM.
- [x] I preinstantiate each module's string constants once, then retain the indexed value without allocating or searching during execution.
- [x] I replaced the linear intern-table scan with a chained hash-bucket table so string interning lookup, insertion, and removal are O(1) amortized instead of O(n); `test-intern` covers dedup, unlink-on-free, bucket growth, and embedded-NUL content.
- [x] I consistently use stored string lengths and preserve embedded zero bytes: NanoVM string constants load with their serialized byte length via `nvm_get_string_len` instead of `strlen`, and `STR_CONTAINS`, `STR_SPLIT`, and `STR_REPLACE` search by stored length through `vmstring_find`/`vm_mem_find`; `tests/nanovm/test_vm.c` covers embedded `\0` bytes across find, substr, char_at, split, replace, and contains.
- [x] I store homogeneous `array<int>`, `array<float>`, `array<bool>`, and `array<byte>` values unboxed in a compact packed buffer (`VmArray.packed`) rather than a boxed `NanoValue[]`, selected by `VmArray.unboxed` and routed through the `vm_array_*` accessors; `test-nanovm` covers the packed round-trip, mutation, slice, and remove paths.
- [x] `ARR_POP` leaves only the element it removed. It used to push the element *and* the array, making it the only instruction in the ISA with two results and forcing every caller to discard one -- codegen emitted an `OP_POP` after every `array_pop`. The array is a heap reference the caller already holds, so threading it back through the stack bought nothing. `ARR_PUSH`, `ARR_SET` and `ARR_REMOVE` keep returning the array because `array_push` and friends are expressions that evaluate to it; that thread-through is language-visible, unlike ARR_POP's second result.
- [x] I replaced NanoVM's separately allocated chained hash-map entries with a contiguous open-addressed table. `test_hashmap_contiguous_collisions` measures one entry-array allocation per table rather than one allocation per inserted entry and tests collisions, growth, deletion tombstones, and tombstone reuse.
- [x] All four ownership sites leaked, and each had a matching correct case nearby to compare against. `vm_array_remove` shifted the tail left without releasing the element it dropped, while `vm_array_set` released the value it overwrote -- so removing from an array of strings leaked one object permanently; the function now takes the heap, as `vm_array_push` already did. `CALL_INDIRECT` popped the callable, taking the stack's reference, and stored a borrowed `VmClosure*` in the frame that nothing released, so every closure call kept its closure alive forever; the frame now owns the callable and releases it at return, at tail-call frame reuse, and on trap unwind. The `OP_CALL_EXTERN` trap popped its arguments and never released them, so an FFI call with a string or array argument leaked it -- released now on every path including the failing one, and a void import's unused result too. `marshal_result` created each string element of a returned array and pushed it without releasing, but `vm_array_push` retains, so each element kept a reference nothing would drop. `test-nanovm` counts live heap objects against the pre-execution baseline rather than zero (`vm_init` preinstantiates the string pool, so zero would measure the constant pool instead of ownership); both new tests were confirmed to fail when the fix is reverted.
- [x] Tracing collection, because the restriction is not enforceable without giving up ordinary data structures. A cycle is constructible from plain NanoLang -- `struct Node { children: array<Node> }`, then push the node into its own array; `array_push` mutates in place, so the field and the local are one object that now contains its owner. Forbidding that would forbid recursive types reached through an array, which rules out trees and graphs. `src/nanovm/heap_cycles.c` implements Bacon-Rajan synchronous collection (PLDI 2001), the same algorithm `src/runtime/refcount_gc.h` already used for generated C, so the two backends no longer disagree about whether a program leaks. Objects whose count drops without reaching zero are the only possible roots and are buffered; a pass applies trial deletion, restores anything still reachable from outside the candidate set, and frees the rest. It runs on a buffer threshold and at heap teardown, guarded against re-entry because collection itself releases. Five tests in `test-nanovm` cover a self-reference, a two-object cycle, a payload leaf freed with its cycle, teardown collection, and -- the failure worse than the leak -- a *reachable* cycle that must survive and still be readable afterwards.

Verifier and safety:
- [x] I reject wrapped function code ranges and require every tested branch target to be an instruction boundary or the function-end sentinel; `test-verifier` verifies the malformed cases.
- [x] The verifier walks the control-flow graph rather than the instruction sequence: a worklist over each instruction's successors, with branch targets resolved through the decoder's instruction-boundary table (`vm_decoded_function_has_boundary`) so a branch into the middle of an instruction is rejected at decode. Two bugs in that walk are fixed here. Branch successors were resolved by looking an *absolute* CODE offset up in a *per-function* index table, so every branch in a function at a nonzero code offset read out of range -- and because the old code incremented the successor count whether or not the lookup succeeded, it then walked an uninitialized index; function 0 starts at offset 0, which is why only it ever worked. A failed lookup is now an error rather than a silent bad successor. `make test-verify-all-programs` verifies every program in `tests/` so a program that compiles but does not verify cannot sit unnoticed.
- [x] Heights are proven through every reachable instruction with merge states required to agree, and `src/nanoisa/verifier_types.c` adds abstract type interpretation over the same walk. The lattice is a known tag or unknown, and a merge of two different known tags widens to unknown rather than failing -- generated code joins a real value with a `PUSH_VOID` placeholder whenever a `match` arm is void, so demanding equality there would reject working programs. What is rejected is a definite contradiction, and every rule restates a check the VM already performs at run time (`I64_ADD` traps with "requires two integers", `F64_ADD` with "requires two floats"), so proving them statically converts a trap into a rejection. A rule that is not true of the VM is a language restriction smuggled in through the verifier: I gave `JMP_TRUE` a boolean condition and had to take it back out, because the VM branches on `val_truthy` and any value is legal. Unknown never fails, so imprecision costs only missed diagnostics. The pass immediately found a real bug -- a transitively imported function whose float arithmetic lowered to integer opcodes (issue #223) -- which is on the `test-verify-all-programs` allowlist with removal as its acceptance test. `test-verifier` covers a string operand, float operands, the matching float op, an unknown operand, and a widening merge.
- [x] I verify call arity and result shape through signature-aware stack effects, dynamic aggregate and closure counts, local/global/flattened-upvalue bounds, encoded type tags, and complete import signatures; `test-verifier` covers each rejected malformed form.
- [x] All five, and they are not all the same kind of claim. **Return shape**: a return must leave exactly the declared result count, checked at `OP_RET` and at the implicit return that reaching the end of a function's code performs -- both were run-time traps before. **Maximum operand depth**: `verify_stack_heights` returns the deepest reachable height; a v2 producer declares it and the loader confirms it, so a producer/verifier disagreement fails at load instead of overflowing later. **Frame depth**: a call reserves the function's locals plus that proven depth, and only their sum can exceed what a frame addresses -- a frame that wrapped would overlap its caller's. **Ownership effects**: `GC_RETAIN`/`GC_RELEASE` move a reference count without touching the operand stack, so height says nothing about whether they pair; the walk tracks the balance alongside the height, rejects a release with nothing held, a return still holding one, and a balance that differs between two paths reaching the same instruction -- the last of which stack height cannot see. **Explicit termination**: reaching the end of a function is defined behaviour, not a fall-through into the next function's code, so the guarantee worth having is that every exit leaves what the function declares, which the return-shape check provides on both the explicit and implicit paths. `test-verifier` covers each.
- [x] `nvm_verify_linked` resolves every `OP_CALL_MODULE` operand pair against the linked-module table (module index, live module, callee function index), and per-function verification now covers every opcode family—type-tag operands are range-checked and any opcode reaching the default arm with an unhandled table operand is treated as a verifier bug; `test-verifier` exercises the linked-call and type-tag cases.
- [x] I reject wrapped and overlapping function code ranges and validate section ranges as an overflow-safe, non-overlapping partition; focused verifier and module-format tests cover containment, adjacency, and directory order.
- [x] The VM re-runs `nvm_verify` over the root and every linked module (`vm_recompute_verified`) and records the result in `VmState.verified`; where that proof holds, the operand-stack accessors dispatch to unchecked private handlers (`stack_pop_unchecked`/`stack_peek_unchecked`) instead of re-checking depth the verifier already proved, falling back to the guarded path for any unverified module. `tests/nanovm/test_vm.c` (Verified Fast Path) pins the flag to the verifier verdict and checks both paths.
- [x] I fuzz the decoder, loader, verifier, assembler, and disassembler with random, truncated, and bit-flipped bytecode in `tests/nanoisa/test_fuzz_malformed.c` (`make test-fuzz-malformed`), and fuzz the co-process wire protocol's value decode and message framing in `tests/nanovm/test_cop_fuzz.c` (`make test-cop-fuzz`); both run under `test-units`.

Module format and tools:
- [x] The NanoISA v2 header (`src/nanoisa/nvm_format_v2.[ch]`) carries a format version, an ISA version, feature bits, a total size, and a section directory whose every entry is bounded against that total by subtraction, so an offset near `SIZE_MAX` cannot wrap into range. Unknown feature bits, unknown section types, and nonzero reserved fields are all refused rather than ignored; `make test-nvm-format-v2` covers the header and directory.
- [x] All ten sections encode and decode: CONSTANTS, SIGNATURES, LAYOUTS, FUNCTIONS, GLOBALS, IMPORTS, LINKS, METADATA, DEBUG and the opaque CODE range the directory locates (`test-nvm-v2-constants`, `-signatures`, `-layouts`, `-functions`, `-imports`, over a shared bounds-checked cursor in `test-nvm-v2-cursor`). `nvm_v2_module_serialize`/`_deserialize` assemble them and enforce what no single codec can see -- every index into another section, every function's code range, and the feature bits the sections require (`test-nvm-v2-module`, 43 assertions). `nvm_v2_from_nvm_module`/`_to_nvm_module` bridge the v1 in-memory module with exact signature deduplication (`test-nvm-v2-convert`, 44). `--emit-nvm` writes v2 by default and `nanoisa_load_bytes` dispatches on the container byte, refusing v1 with a rebuild instruction (`test-nvm-v2-endtoend`, 20).
- [x] I reject duplicate singleton sections, overlaps, partial fixed-width records, trailing data, interior gaps, and arithmetic overflow in `nvm_deserialize` via a structural section-directory validation pass, covered by `tests/nanoisa/test_nanoisa.c` module-format tests.
- [x] I accept typed symbolic functions, imports, fields, types, constants, and labels as assembler operands while retaining numeric operands, covered by focused NanoISA assembler tests.
- [x] Canonical disassembly (`DISASM_STYLE_CANONICAL`) reassembles to byte-identical bytecode and identical tables. The instruction stream already round-tripped; the tables did not, because the import table, the linked-module references and the type-definition counts had no textual form at all -- a module with a `CALL_EXTERN` disassembled fine and then failed to reassemble, because the text never declared the import table it indexed. `.import`, `.module_ref` and `.types` close that, with one shared length-aware quoting routine so the string pool and the directives that name entries in it cannot disagree about escaping. `make test-disasm-roundtrip` compares code bytes, every table, the flags and the entry point across eleven modules including binary string constants, an import with parameter tags, a void import, linked references and an indirect call's encoded shape.
- [x] I reject trailing instruction and directive operands, and I run the bytecode verifier before returning any assembled module; `test-nanoisa` covers both rejection paths.
- [x] I correct disassembler import annotations (`CALL_EXTERN` now resolves the imported module and function), branch operand roles (only genuine branch operands become labels; non-branch `i32` immediates print numerically), label construction, and binary-string handling (length-aware `\xHH`/`\r` escaping that round-trips embedded NUL and non-printable bytes), covered by disassembler tests in `tests/nanoisa/test_nanoisa.c`.
- [x] I tie every legacy opcode to its schema and enum value, VM dispatch behavior, an explicit or decode-backed verifier route, and tested encode, decode, assembly, and disassembly paths; `schema-check` and `test-nanoisa` fail when these layers drift.
- [x] I removed the no-op GC-scope opcodes (`GC_SCOPE_ENTER`, `GC_SCOPE_EXIT`) and the duplicate `CLOSURE_CALL` (subsumed by `CALL_INDIRECT`), which no frontend emitted; the remaining unemitted polymorphic scalar/string opcodes are retained as justified assembler-compatibility instructions (see the row above).
- [x] I use separately linked `.nvm` modules: serialized `MODULE_REFS` define the checked dependency names and `CALL_MODULE` indices, `vm_link_named_module` enforces their order, and `test-nanovm` serializes, loads, links, and executes a two-file module graph end to end.

FFI and traps:
- [x] I will resolve imports once into typed call descriptors.
- [x] I dispatch mixed integer and floating FFI signatures through generated typed stubs (`scripts/gen_ffi_dispatch.py` → `src/nanovm/ffi_dispatch_generated.h`) so int/pointer args use general-purpose registers and float args use FP registers per the platform ABI, covered by mixed-signature tests in `tests/nanovm/test_vm_ffi.c`.
- [x] I made argument limits consistent across imports, traps, direct FFI, and co-process calls by sharing a single `NANO_MAX_FFI_ARGS` limit (in `src/nanoisa/nvm_format.h`): the verifier rejects imports past it, the `OP_CALL_EXTERN` trap buffer is sized to it and traps instead of silently truncating, and the direct in-process, co-process, and interpreter FFI dispatch paths all size their argument arrays and bounds to it, covered by `tests/nanoisa/test_verifier.c` import-limit tests.
- [x] Measurement does not support it. The same FFI workload costs 15.8 microseconds in-process and 764.2 through the co-process, so the boundary crossing is roughly 48x the entire in-process call including the copy -- removing the copy cannot move the isolated case, and the in-process case is already at the noise floor. The copy also buys a property the range does not: the trap owns its arguments, so the operand stack may move underneath it. The item was explicitly conditioned on measurement, and this is the measurement. [NANOISA_MEASUREMENTS.md](NANOISA_MEASUREMENTS.md)
- [x] Co-process serialization is little-endian by construction, not by host accident: every wire integer goes through `cop_put_u*`/`cop_get_u*` byte shifts, and `CopMsgHeader` exists only in memory -- `cop_send` builds the eight header bytes explicitly rather than writing the struct. `wire_header_byte_layout_is_fixed` pins those bytes, because a host-order struct write would pass every same-machine test and fail across architectures. The large-payload path is now tested rather than assumed: the mailbox has 4 KB slots so anything bigger takes the pipe path, and a 1 MB payload plus a 512 KB string value (with an embedded zero, so only the length field saves it) now round-trip through a forked reader -- a pipe holds tens of kilobytes, so this needs a concurrent drain and cannot be faked in one process. Previously nothing exercised more than 256 bytes; the fuzzer only checked that a length *above* `COP_MAX_PAYLOAD` is refused, which says nothing about a legal large one arriving intact. `make test-cop-protocol` covers all four.
- [x] All of them, after fixing a harness that measured none of them. `make benchmark-nanoisa` used to time one process per sample, so every workload took about 17 ms whether it retired 78 instructions or 32,082: startup was the whole measurement and execution under 1% of it, meaning the suite could not have detected an interpreter change of any size. Each workload is now measured twice per sample -- one iteration and many, behind one startup -- and the per-iteration cost is the difference, so the startup terms cancel and drift between the two is cancelled by pairing them. Cold startup then becomes a dimension rather than a confound (17.4 ms, 60-1800x a single execution); warm and scalar calls, strings and arrays come from the existing workloads now that they are visible; and the co-process boundary is measured against the same FFI workload in-process, giving the crossing cost that justifies batching (48x) and the launch cost that bounds crash recovery (57.9 ms).
- [x] I batch high-frequency host work through a coalesced co-process path: `vm_ffi_call_cop_batch` packs many extern calls (`COP_MSG_FFI_BATCH`) into each shared-memory mailbox crossing so a signal/ack pair covers the whole batch instead of every element, with `tests/nanovm/test_cop_protocol.c` covering the forked round-trip, empty batches, and mid-batch error reporting.

Documentation and acceptance:
- [x] I replaced stale NanoISA opcode counts and architecture claims in the active documentation; historical changelog entries remain historical.
- [x] I document the portable ISA separately from verified and optimized runtime representations in [docs/NANOISA_PORTABLE_ISA.md](NANOISA_PORTABLE_ISA.md): the portable contract lives in `spec/nanoisa.yaml` and the `.nvm` format, while the verified instruction IR and optimized dispatch IR are documented as internal, non-portable runtime representations.
- [x] `examples/nanoisa/nanolang_shapes.nasm` and `examples/nanoisa/forth_words.nasm` show the same ISA serving two calling conventions: NanoLang keeps values in locals and uses the operand stack for temporaries, while Forth makes the stack the convention itself and leans on `DUP`, `SWAP` and `PICK` -- primitives that must be instructions because a library cannot reach into the operand stack. Every operand is symbolic: named constants, named functions and named labels, resolved by the assembler and reconstructed by the disassembler. `nanoisa asm` makes them runnable rather than decorative, and `make test-asm-examples` assembles each one (which verifies it), runs it, checks its documented output, and confirms it survives canonical disassembly and reassembly byte-for-byte -- so the hand-written symbolic form and the emitted resolved form are the same language rather than two dialects.
- [x] I record why every public instruction belongs in the ISA rather than a runtime library: every `instruction_families` entry in `spec/nanoisa.yaml` carries a `justification` (representation, core-semantics, execution-substrate, control-flow, host-boundary, or encoding), `scripts/gen_nanoisa_schema.py` refuses to emit the schema without one, and the classification and rule live in `docs/superpowers/specs/2026-09-01-nanoisa-public-instruction-rationale.md`; covered by `tests/test_nanoisa_schema.py`.
- [x] `make benchmark-nanoisa` prints median, IQR and IQR-as-a-share-of-median per dimension, and the summary JSON carries mean, min, max, p25 and p75 alongside the environment and commit. The share matters more than the spread: two workloads have a 1.6% band and two have a 10% band, so "5% faster" means something different in each, and the reporting convention in [NANOISA_MEASUREMENTS.md](NANOISA_MEASUREMENTS.md) requires a change to be stated against this baseline on the same machine, compiler, flags, workloads and sample count.
- [x] I made generated-C tracing and profiling dynamically selectable at
  process startup, with one shared hook mechanism and no per-event environment
  lookups or expensive work when each hook is disabled.

### Phase 13 - Forth 2012 on NanoISA (4.1)

Goal: I will implement a standards-oriented Forth system whose colon words are
verified NanoISA functions and whose typed library words use the same import and
co-process machinery as NanoLang. This phase is the primary remaining 4.1
goal. The SDL editor astronaut is a side-quest after this phase
(`docs/NANO_EMACS.md`, `bin/nano_emacs_worker`).

Foundation:
- [x] I selected Forth 2012 Core and every optional word set as the target.
- [x] I established the persistent NanoVM invocation boundary required by an interactive compiler.
- [x] I added and verified `examples/language/forth/pi.fs` under Gforth for 0, 1, 10, and 50 places.
- [x] I pinned Forth 2012, Jackson test-suite v0.15.0,
      forth200x `91f1ed9c756aac27f57e939c270b5f2c84262427`, and Gforth 0.7.3
      in `docs/FORTH_2012.md` and `tests/forth/pins.json`.
- [x] I confirmed Jackson-suite notices permit vendoring with copyright
      retained, inventoried forth200x per file instead of copying it, and
      refused to vendor GPL Gforth. Jackson v0.15.0 is vendored under
      `tests/forth/vendor/gerryjackson/`. forth200x and Gforth are not.
- [x] I added `make test-forth-gforth-diff` against pinned Gforth 0.7.3 for
      `pi.fs` at 0, 1, 10, and 50 places.
- [x] I documented cells, characters, addresses, division, floats, files,
      terminals, blocks, limits, and ambiguous-condition behavior in
      `docs/FORTH_2012.md` as the assumed NanoISA Forth contract.

Compiler and runtime:
- [x] I will create one mutable `NvmModule` and persistent `VmState` per Forth session.
- [x] I will add VM-owned data, return, floating-point, and control-flow stacks.
- [x] I will add a byte-addressable virtual Forth address space with validated allocation and file handles.
- [x] I will implement dictionary headers, execution tokens, name tokens, early binding, immediacy, and word lists.
- [x] I will implement nested terminal, evaluated-string, included-file, and block input sources with `SOURCE` and `>IN` restoration.
- [x] I will compile each colon definition privately to NanoISA, verify it, then publish it atomically.
- [x] I will compile calls to earlier definitions as stable `OP_CALL` references and `RECURSE` to the reserved current definition.
- [x] I will compile structured control flow with a checked compile-control stack and branch patching.
- [x] I restore Forth stacks, input sources, and NanoVM invocation state from C
      (`forth_catch` / `forth_colon_throw`). That is not dictionary words
      `CATCH` and `THROW`.
- [x] I expose `CATCH` and `THROW` as Forth words. Nested `vm_invoke` keeps a
      saved frame count (`forth_invoke_nested`). `THROW` HALTs the outer NanoISA
      function, not only the inner host call. I test `CATCH` inside a colon
      definition and `THROW` from a called word.
- [x] I restore Forth locals on `THROW`. Recursive `{: :}` frames are
      NanoISA locals; `CATCH` of a throwing recursive locals word leaves
      later `{: :}` words with fresh slots (`make test-forth-session`).
- [x] I will implement typed Forth import declarations that lower to `NvmImportEntry` and `OP_CALL_EXTERN`.
- [x] I will reject FFI signatures the active ABI cannot call correctly instead of guessing.
- [x] I will restart an isolated FFI co-process after dynamic import-table mutation.
- [x] I will make `SEE` disassemble the actual compiled NanoISA function and describe imported words.

Kernel defects found during Core (dependency order; Core stays open until
these that belong to Core are verified). Ownership:
`task_a13c30e77f703469932d36d7d1065b41`. Dictionary `CATCH`/`THROW` lives with
compiler and runtime above, not only this list.

Closed this session:
- [x] I stop sharing `FORTH_CTRL_ORIG` between `LEAVE` and `IF`/`THEN`. Leave
      jumps are a chain on `ForthCtrlItem.aux`. `IF LEAVE THEN LOOP` compiles.
- [x] `S"` and `."` skip leading blanks after the word so `TYPE` does not print
      a leading space.
- [x] `CREATE` captures `HERE` after the name is allotted, so `,` and `@` hit
      the body, not the name bytes.
- [x] `bin/forth` is a copy of `bin/nano_forth`, not `nl_forth_interpreter`.
      `make sdl` and `make forth-ide` require it. The PTY test execs it.
- [x] `forth_take_word` consumes the trailing blank. `S"` with extra blanks
      still types the payload, not a leading space.
- [x] `VARIABLE` allots the data cell after the name, matching `CREATE`.
      `ALIGN VARIABLE VX HERE VX -` is one cell.
- [x] Interpret-time `S"` uses one `WORD` buffer; a second `S"` clobbers the
      first. The contract is in `docs/FORTH_2012.md` and
      `test_kernel_defects`.
- [x] A runtime host that returns `0` at interpret `STATE` fails closed. It
      does not `vm_invoke` its own trampoline.
- [x] I removed unused `FORTH_HOST_UNLOOP` and `FORTH_HOST_BRACKET`.
- [x] `g_forth` is the session on the C invoke/interpret stack. Nested invoke
      saves and restores it. Two sessions do not share a dictionary.
- [x] `BYE` is a Forth word. The REPL does not special-case the line `bye`.
      `: FOOX BYE ; FOOX` sets `forth_exit_requested`.
- [x] `EXECUTE` of a compile-only stub (`IF`, `:`, …) is rejected. The
      ambiguous-condition policy is in `docs/FORTH_2012.md`.
- [x] Remaining Core words: `>BODY`, `>NUMBER`, `POSTPONE` (via `COMPILE,`),
      `ABORT"`, `KEY`, `ACCEPT`, and `QUIT`. `KEY`/`ACCEPT` read remaining
      `SOURCE`. `QUIT` empties the return stack, sets interpretation `STATE`,
      and stops the current line. Core stays open until pinned suites pass.
- [x] `+LOOP` uses the Forth 2012 crossing rule (unsigned interval
      `(index, index+n]` contains `limit`). Increment `0` does not terminate.
      Jackson `GD7`/`GD8` need this; signed `<` after add hangs on
      `MAX-UINT`/`USTEP`, and signed XOR of `index-limit` false-stops at
      `2^(bits-1)`.
- [x] `IMMEDIATE` is a runtime host trampoline, so `CREATE , IMMEDIATE
      DOES>` compiled into a defining word marks the child, not a no-op
      `RET` stub.
- [x] Nested `CREATE` / `;` from a running colon defers NanoISA rebuild
      until `vm_exec_depth` is 0 so the defining word keeps executing.
- [x] Nested colon publish from `EVALUATE` while `vm_exec_depth != 0`
      appends decode/dispatch (`vm_sync_new_functions`) instead of skipping
      rebuild, so Jackson `SSQ7`/`SSQ9` can define then execute the new xt.
      I do not free the caller's decoded instructions.
- [x] `:NONAME` and number prefixes `#` `$` `%` `'c'` so Jackson
      `coreplustest.fth` can load. `:NONAME` is Core Ext.
- [x] Core Ext kernel: `VALUE`/`TO`, `MARKER`, `CASE`/`OF`/`ENDOF`/`ENDCASE`,
      `PARSE-NAME`, `BUFFER:`, `DEFER`/`IS`/`ACTION-OF`, `HOLDS`, `S\"`, `C"`,
      `.R`/`U.R`, `UNUSED`, `SAVE-INPUT`/`RESTORE-INPUT`, `REFILL`/`SOURCE-ID`
      as Forth words. Jackson `coreexttest.fth` is the gate, loaded through
      C `REFILL` after Core evidence (`make test-forth-coreext`). Passing
      that file is evidence. I still do not claim Core Ext as a banner.

Standard word sets, in dependency order:
- [x] I will implement and test Core. Fan-out 1–4 is verified (Jackson vendored,
      forth200x inventoried, `make test-forth-jackson` records the INCLUDE gap,
      coverage matrix published). Jackson Core evidence files pass under
      `make test-forth-core`. Passing a suite is evidence. I still do not
      claim Core as a banner.
- [x] I will implement and test Core Extensions. Jackson `coreexttest.fth`
      passes under `make test-forth-coreext`. Passing a suite is evidence.
      I still do not claim Core Ext as a banner.
- [x] I will implement and test Exception and Exception Extensions.
      Jackson `exceptiontest.fth` passes under `make test-forth-exception`.
      `ABORT` is `THROW -1`. An undefined word throws `-13`. Passing that
      file is evidence. I still do not claim Exception as a banner.
- [x] I will implement genuine double-cell arithmetic and test Double Number and its extensions.
      Jackson `doubletest.fth` is the gate (`make test-forth-double`), loaded
      through C `REFILL` after Core evidence, `errorreport.fth`, and
      `utilities.fth`. Passing that file is evidence. I still do not claim
      Double as a banner.
- [x] I will implement and test String and String Extensions.
      Jackson `stringtest.fth` is the gate (`make test-forth-string`), loaded
      through C `REFILL` after Core evidence, `errorreport.fth`, and
      `utilities.fth`. Passing that file is evidence. I still do not claim
      String as a banner.
- [x] I will implement and test Search Order and Search Order Extensions.
      Jackson `searchordertest.fth` is the gate (`make test-forth-searchorder`),
      loaded through C `REFILL` after Core evidence, `errorreport.fth`, and
      `utilities.fth`. Passing that file is evidence. I still do not claim
      Search Order as a banner.
- [x] I will implement and test File Access and File Access Extensions.
      Jackson `filetest.fth` is the gate (`make test-forth-file`), loaded
      through C `REFILL` after Core evidence, `errorreport.fth`,
      `utilities.fth`, and `coreexttest.fth` (for `S$` / `SI_INC`). Passing
      that file is evidence. I still do not claim File Access as a banner.
- [x] I will implement and test Memory Allocation.
      Jackson `memorytest.fth` is the gate (`make test-forth-memory`), loaded
      through C `REFILL` after Core evidence, `errorreport.fth`, and
      `utilities.fth`. Passing that file is evidence. I still do not claim
      Memory-Allocation as a banner.
- [x] I will implement recursive, reentrant Locals and Locals Extensions.
      Jackson `localstest.fth` is the gate (`make test-forth-locals`), loaded
      through C `REFILL` after Core evidence, `errorreport.fth`, and
      `utilities.fth`. Locals are NanoISA frame slots, so `RECURSE` and
      nested calls do not share values. Passing that file is evidence. I
      still do not claim Locals as a banner.
- [x] I will implement and test Facility and Facility Extensions.
      Jackson `facilitytest.fth` is the gate (`make test-forth-facility`), loaded
      through C `REFILL` after Core evidence, `errorreport.fth`, and
      `utilities.fth`. Passing that file is evidence. I still do not claim
      Facility as a banner.
- [x] I will implement and test Programming Tools and Programming Tools Extensions.
      Jackson `toolstest.fth` is the gate (`make test-forth-tools`), loaded
      through C `REFILL` after Core evidence, `errorreport.fth`, and
      `utilities.fth`. Passing that file is evidence. I still do not claim
      Programming Tools as a banner.
- [x] I will implement an IEEE binary64 floating stack and test Floating Point and its extensions.
      Jackson `ak-fp-test.fth` is the gate (`make test-forth-float`), loaded
      through C `REFILL` after Core evidence, `errorreport.fth`, and
      `utilities.fth`. Passing that file is evidence. I still do not claim
      Floating-Point as a banner.
- [x] I will implement UTF-8 Extended Character and Extended Character Extensions.
      Jackson has no xchar file. I do not vendor forth200x `tests/xchar.fs`.
      Session tests in `tests/forth/test_forth_session.c` (`CHAR`, `XC@+`,
      `X-SIZE`, `+X/STRING`, `XC-SIZE`) are the gate
      (`make test-forth-session`). Passing those is evidence. I still do not
      claim Extended Character as a banner.
- [x] I will implement Block and Block Extensions against an explicitly disposable image.
      Jackson `blocktest.fth` is the gate (`make test-forth-block`), loaded
      through C `REFILL` after Core evidence, `errorreport.fth`, and
      `utilities.fth`. The session RAM image holds 32 blocks; Jackson
      overwrites 20–29. Passing that file is evidence. I still do not claim
      Block as a banner.

Tests, examples, and SDL IDE:
- [x] I made the noninteractive example runner pass its declared output
      checks for `nl_affine_resource_demo.nano`, `nl_array_infer.nano`, and
      the finite `nl_boids.nano` simulation. `make -C examples test` passes
      70 language and 9 verified examples.
- [x] I record a successful `SKIP:` result from a platform-gated example
      without disguising it as an output mismatch. The Linux libdispatch
      examples remain skipped rather than claimed runnable.
      `make -C examples test` exercises that boundary.
- [x] I select a Python interpreter with the `yaml` dependency required by
      `scripts/gen_nanoisa_schema.py`, so `make test` reaches the test suite on
      a supported Linux host.
- [x] I made `sudo make install-deps` select Arch package names from
      module metadata, including Bullet, GLEW, and GLFW, then document the
      full Arch dependency set.
- [x] I made the Linux Forth IDE PTY liveness test wait for the NanoISA session
      dictionary to initialize before it requires the banner and prompt.
      `make test-forth-pty` passes.
- [x] I will retain the existing 280 cases as regression tests while replacing their nonstandard harness assumptions.
      `make test-forth-examples` loads Jackson `tester.fr` and the nine
      `examples/language/forth/test_*.fs` files through C `REFILL`.
- [x] I will run pinned committee Core and optional-word-set tests.
      `make test-forth-core` through `make test-forth-block` plus
      `make test-forth-jackson` and `make test-forth-gforth-diff`.
- [x] I will run licensed Forth-2012 tests and record unsupported or manual cases separately.
      Jackson v0.15.0 is the licensed suite I execute. forth200x is inventoried
      and not vendored. Skips are `tests/forth/forth2012_skips.txt`.
- [x] I will add malformed definitions, multiline definitions, early binding, immediate words, execution tokens, overflow, unsigned output, loop boundaries, exceptions, source nesting, and UTF-8 tests.
      `make test-forth-session` (`test_malformed_and_utf8` and the named
      session tests).
- [x] I will make `pi.fs` pass under my Memory-Allocation and Exception implementations with the exact 50-place output.
      `make test-forth-session` (`test_pi_fs`) matches `tests/forth/pins.json`.
- [x] I will update every file in `examples/language/forth/` to standard behavior.
- [x] I will update `sdl_forth_ide` to launch the NanoISA-backed Forth executable.
- [x] I will keep the SDL IDE as a PTY client rather than create a second Forth implementation.
- [x] I will add build, PTY, file-loading, interpreter-liveness, and graphical smoke coverage.
      `make test-forth-pty`, `make test-forth-ide-smoke`, C `REFILL` file
      loads, and `bin/forth` as the IDE child.
- [x] I will publish the precise standard-system label only after tests and required documentation support it.
      The label is `docs/FORTH_STANDARD_SYSTEM.md`: I am not a Forth 2012
      Standard System. Jackson evidence is not that banner.
      `make test-forth-jackson`.
      MAC `task_6087b948f1c9a7672420b4e1ea72bd35`.

### Phase 14 - NanoISA-Centered Backends (4.0 spike; rewrite is 5.1)

Goal: 4.0 records that NanoISA is the common IR and that a closed integer
subset can become structured C11 without embedding the VM. The ambitious
rewrite — I emit only `.nvm`; C/LLVM/Wasm are translators; bootstrap
compares `.nvm`; `transpiler.nano` leaves the compiler — is **5.1**.
Contract: `docs/NANOISA_ONLY.md`. I do not delete the AST→C path in 4.x.

Architecture:
- [x] I selected NanoISA as the common IR for NanoLang, Nano Forth, and future frontends.
- [x] I wrote `docs/NANOISA_HL_ROUNDTRIP.md`: reconstruction means named
      functions, types, structured control, and a host ABI. A C file that
      embeds `nano_vm` plus a bytecode array is not reconstruction.
      Canonical disassembly is a different test.
      (`task_4bd034f6029b7458201db74e2c3aeb32`)
- [x] I inventoried `.nvm` v2 (`FUNCTIONS`, `SIGNATURES`, `LAYOUTS`,
      `IMPORTS`, `LINKS`, `CONSTANTS`, `DEBUG`, `METADATA`) against that
      bar. Local names are still slot numbers.
- [x] I emit structured C11 from a closed NanoISA subset (`nvm2c`: i64
      arithmetic, locals, `CALL`, `RET`/`HALT`). `make test-nvm2c` compiles
      that C with `cc` and checks exit status `42` for `add(40, 2)`. The
      C contains integer `+` and does not contain `nano_vm` or a bytecode
      blob. I do not delete `transpiler.nano`. I do not ship a `nvm2c` CLI.
      I refuse `CALL_EXTERN` (VM FFI / co-process), not a host C ABI.
      (`task_863f10a181aba8d2dfdc9127e7113938`)

Direct backend retirement:
- [x] I removed the immature direct NanoLang-to-LLVM backend and its CLI, build, test, and CI surface.
- [x] I removed the immature direct NanoLang-to-WebAssembly backend and its CLI, build, test, signing, publication, and CI surface.
- [x] I removed LLVM and Wasm from the direct cross-backend CI matrix.
- [x] I retain the retired direct backends in Git history rather than carry dormant implementation files in the active tree.

I moved these 4.0-or-later rewrite items to Phase 20 (5.1): frontend facts as
metadata, compute profiles, C11 as the canonical AOT backend, LLVM/Wasm/GPU
translators, semantic equivalence across targets, a second HLL surface from
the same `.nvm`, and the published reconstruction finding. `nano_virt`'s
default native output still embeds the VM. That is recorded, not solved.

### Phase 15 - Internationalization and UTF-8 Neutrality (4.2)

Goal: I remain language-neutral in source, runtime text, diagnostics, and logs,
and I publish a useful translated guide as evidence. English remains the
canonical source until the translation workflow proves otherwise.

Language scope:
- [x] I selected the six largest languages by total-speaker metrics for the initial proof: English, Mandarin Chinese, Hindi, Spanish, Modern Standard Arabic, and French.
- [x] I recorded Ethnologue 2026 total-speaker estimates as the dated selection method and treat the ranking as revisable.
- [x] I parse BCP 47 tags into language, script, region, and variant, with a
      fallback chain that ends at `en` when the tag is not English. Encoding,
      collation, and text direction are separate fields, not one setting
      (`src/bcp47.c`, `make test-bcp47`). Human stderr uses catalogs;
      JSON/TOON stay English.
- [x] I will use BCP 47 language tags and explicit fallback chains.
      `nanoc` resolves a process locale (`src/locale.c`) and prints the
      chain from `--print-locale`. Catalogs are a later item in this phase.
- [x] I will distinguish language, script, region, locale, encoding, collation, and text direction rather than treating them as one setting.
      `--print-locale` prints those axes as separate fields.

UTF-8 language and runtime contract:
- [x] I reject invalid UTF-8 in `.nano` source at compile (`CSRC01`,
      `src/utf8.c`, `make test-utf8` / `make test-src-utf8`). `nl_string`
      and `bstr_validate_utf8` use the same walker. Explicitly binary
      payloads remain allowed. I do not claim diagnostic, log, metadata,
      or documentation boundaries.
- [x] I link `src/utf8.c` into programs that use `nl_string` (`nanoc`
      runtime list, `nanoc_v06`, `driver.nano`) so `nl_utf8_validate`
      resolves at user-program link.
- [x] I require valid UTF-8 at JSON/TOON diagnostic emit (invalid fields
      become the ASCII marker `<invalid UTF-8>`), `module.json` (fail
      closed), and Markdown docgen source/module name (fail closed).
      `make test-diagnostics`, `make test-module-metadata`, `make test-docgen`.
- [x] I require valid UTF-8 at remaining text boundaries unless the value
      is explicitly binary: log payloads (`make test-log-utf8`), NanoVM
      stack-trace/trap text (`nl_utf8_cstr_or_marker`), SDL TTF
      (`TTF_RenderUTF8_*`, fail closed), displayed paths, and terminals
      via `nl_utf8_sanitize_log`. Binary payloads stay binary.
- [x] Identifiers are ASCII `[A-Za-z_][A-Za-z0-9_]*`. Non-ASCII is not an
      identifier and fails closed (`L0003`). The C lexer matches
      `src_nano/compiler/lexer.nano` (empty token list on failure).
      Confusable/homoglyph rejection is this ASCII policy. NFC of
      identifiers is unused while names are ASCII.
      (`make test-parser`, `make test-src-utf8`)
- [x] The self-hosted lexer records `L0003` on unexpected bytes and
      returns no tokens, matching the C compiler fail-closed policy.
- [x] Where the public Unicode FFI claims them, I implement grapheme
      iteration, NFC/NFD/NFKC/NFKD, Unicode case conversion, case folding,
      and display width via utf8proc (`modules/unicode/unicode_ffi.c`,
      `make test-unicode-ffi`). `nl_string` stays byte-length plus
      code-point ops; it is not the grapheme API.
- [x] Byte-oriented APIs stay explicit and separate from code-point and
      grapheme APIs (`nl_string_length` vs `nl_string_utf8_length`;
      `nl_str_byte_length` vs `nl_str_grapheme_length`).
- [x] I preserve embedded zero bytes only in binary strings and
      length-aware protocols (`nl_string_new_binary`, `make test-nl-string`).
- [x] I test malformed, overlong, truncated, combining, supplementary-plane,
      emoji, and bidirectional UTF-8 at the walker (`make test-utf8`) and
      combining grapheme/NFC at the Unicode FFI (`make test-unicode-ffi`).
- [x] Locale-dependence audit: `nl_ascii_is*` in `lexer.c`, NanoVM
      builtins, eval, nvm2c identifiers, docgen, FFI bindgen, BCP 47
      subtag case, and `main.c` wrapper type names. Source-map and SDL
      text paths use UTF-8 fail-closed helpers. JSON/TOON emit and
      `module.json` are fail-closed.

Language-neutral diagnostics and logging:
- [x] I keep pipeline compiler diagnostics as stable IDs (`src/diag_id.c`,
      `diag_en` in `src_nano/compiler/diagnostics.nano`). English is a
      lookup. Typechecker titles that share `E003`, lexer/parser events,
      LSP fallback `E000`, and log events are not this table.
- [x] Typechecker titles that go through `emit_context_error` carry unique
      `E001`–`E034` (`E024` undefined variable via `--json-errors`).
      Inference `TYPE MISMATCH` is `E001`. I do not claim lexer/parser/LSP
      or log events.
- [x] Remaining diagnostic and log events have stable IDs independent of
      English: lexer `L0003`–`L0008`, parser `P0001`/`P0002`, unlabeled
      typechecker JSON `E035`, log `LOG01`–`LOG04`. LSP uses those ids
      instead of `E000`.
- [x] Structured JSON/TOON fields (severity, phase, location, code,
      English message) stay stable. Localized prose is catalog lookup on
      human stderr only (`nl_catalog_text`).
- [x] Locale selection through `--locale`, `NANO_LOCALE`, POSIX
      `LC_ALL`/`LANG` (`en_US.UTF-8` → `en-US`, `C`/`POSIX` → `en`).
      `--print-locale` is a query and does not compile. Compile output
      does not grow a locale banner. JSON/TOON diagnostics stay English.
      Flags may precede the input file.
- [x] UTF-8 message catalogs with English fallback and missing-key
      counting (`src/catalog.c`, `catalogs/messages/{en,zh,hi,es,ar,fr}.json`,
      `make test-catalog`).
- [x] Plural rules, number formatting, dates, lists, quoting, and
      parameter reordering in `nl_catalog_format` (`make test-catalog`).
- [x] LLM JSON and TOON diagnostics stay language-neutral (English
      lookup via `nl_diag_en`). Localized rendering is human stderr.
- [x] Logs drop bidi overrides and ANSI CSI/OSC (`nl_utf8_sanitize_log`,
      `make test-log-utf8`).
- [x] Catalog completeness, placeholder compatibility, fallback, invalid
      UTF-8, and six-language keys (`make test-catalog`). Human stderr
      for all six languages vs English JSON (`make test-locale-catalog`).

Translated documentation and user guide:
- [x] User-guide builder is locale-aware: per-language navigation, canonical
      URLs, `lang`, `dir`, `hreflang`, fallback metadata for generated
      pages (`scripts/build_userguide.py`).
- [x] Translation source format: YAML front matter plus Markdown; code
      fences, links, snippet markers, and identifiers stay English
      (`userguide/i18n/`).
- [x] Translation memory (`userguide/i18n/memory.json`) hashes English
      sources; stale drafts get a banner rather than silent publication.
- [x] Machine-translated Simplified Chinese, Hindi, Spanish, Modern
      Standard Arabic, and French drafts of the published nav pages, marked
      `machine_generated: true`.
- [x] English code examples and identifiers preserved in those drafts
      (fence byte-identity checked while writing; `tests/test_build_userguide.py`).
- [x] Language switching keeps the current page (`hreflang` + `.langs`).
- [x] Generated links and fragments validated for all six editions
      (`make userguide-html`). Code blocks preserved. Font fallback and
      Arabic `dir=rtl` plus mobile breakpoints are in CSS and unit tests.
      The published guide has no in-page search; `nano-docs` searches
      Markdown separately.
- [x] Contributor translation workflow: `userguide/i18n/README.md` and
      `docs/USERGUIDE_BUILD.md`.
- [x] Human reviewers vs machine drafts: `reviewed: false` until a named
      reviewer accepts a page. No page is claimed as human-reviewed yet.

Acceptance:
- [x] Six-script program through C and NanoVM
      (`examples/language/i18n_six_scripts.nano`, `make test-i18n-scripts`).
- [x] Localized diagnostics for all six languages on human stderr;
      JSON stays English (`make test-locale-catalog`). Log events carry
      stable ids (`make test-log-utf8`).
- [x] Six guide editions build and link-check (`make userguide-html`;
      CI `userguide-html` job).
- [x] Visual properties for Hans, Devanagari, Latin, and Arabic: desktop
      screenshots of en/zh/hi/es, mobile zh/hi, Arabic print raster
      (Chrome `--screenshot` of `dir=rtl` stayed black). Glyphs rendered;
      switcher keeps the page; Arabic reverses switcher order. Code
      fences on RTL pages are LTR-isolated. Not a WCAG audit.
- [x] I do not call the system internationalized: JSON/TOON and catalog
      fallback still use English; machine drafts are not human translations.

I chose the initial publication languages from total-speaker estimates reported
by Ethnologue 2026: English, Mandarin Chinese, Hindi, Spanish, Modern Standard
Arabic, and French. Counts and even language boundaries are estimates, so I
record this as a dated coverage decision rather than a permanent ranking.

### Phase 16 - Service Interface Description and Module Migration (4.3)

Goal: I will turn a module boundary into a language-neutral service contract.
The same source-level import can bind to an in-process implementation, a local
service, a protected process, another NanoVM, or a remote endpoint according to
deployment policy.

Interface definition:
- [x] I define a versioned Nano Service Interface schema with stable interface, method, type, error, and capability identifiers (`schema/nsi/examples/log.nsi.json`, `src/nsi.c`, `docs/NSI.md`, `make test-nsi`). Names without ids fail closed.
- [x] I describe parameter direction, ownership, borrowing, transfer, lifetime, mutability, optionality, and streaming in the schema (`docs/NSI.md`, `make test-nsi`). Unknown enumerations fail closed.
- [x] I support records, variants, arrays, strings, binary data, resources, callbacks, asynchronous results, and versioned errors (`schema/nsi/examples/types.nsi.json`, `make test-nsi`). Unknown kinds fail closed.
- [x] I define backward- and forward-compatibility rules for NSI documents (`nl_nsi_compat`, `make test-nsi`). Adding a method is compatible; removing one is breaking. Session hello applies the same rule before calls.
- [x] I reject ambiguous ABI inference; every foreign boundary has an explicit typed contract (`make test-nsi`).
- [x] I generate NanoLang and Nano Forth bindings from the same interface description (`make test-nsi-gen`).
- [x] I generate client stubs, server dispatch, serialization, validation, documentation, mocks, and compatibility tests (`src/nsi_gen.c`, `make test-nsi-gen`).
- [x] I generate NanoISA imports and typed trap descriptors from service contracts (`nl_nsi_gen_nanoisa_imports`).
- [x] I preserve implementation language neutrality: C, C++, Python, Rust, NanoLang, NanoVM, and remote bind the same method ids (`nl_nsi_gen_language_index`). Remote transport is not implemented.

Module refactoring:
- [x] I extend module manifests with interface version, required capabilities, isolation policy, resource budgets, restart policy, and implementation adapter (`nsi` in `module.manifest.json`).
- [x] I separate portable interface metadata from platform-specific build metadata (`nsi` vs `module.json`; `c_sources` on a manifest fails closed).
- [x] I inventory every current native and Python-backed module by privilege, state, payload size, latency, and failure behavior (`schema/nsi/inventory.json`).
- [x] I migrate pure modules first and prove identical in-process and service-process behavior (`nsi:nanolang/vector2d#add`, `make test-nsi-runtime`).
- [x] I migrate filesystem, logging, process, networking, audio, graphics, GPU, and Python modules in increasing privilege order (class NSI documents under `schema/nsi/modules/`, typed handles, not host-library wrap).
- [x] I keep unsafe implementation details behind generated service boundaries rather than expose host pointers or library objects.
- [x] I give each resource handle an interface type, service identity, generation, rights mask, and lifetime state.
- [x] I update module discovery and package metadata to resolve interface contracts independently from implementations (`sdl` and `glfw` share `nsi:nanolang/graphics`).
- [x] I add contract tests that run one client against every supported implementation of an interface (`client_log_write` vs inproc/mock/local).

Transport-neutral invocation:
- [x] I replace symbol-name-centered RPC with stable interface and method identifiers.
- [x] I define request, response, error, cancellation, deadline, and stream frames.
- [x] I negotiate interface and transport versions before accepting calls (`nl_nsi_session_hello`; transport version 0).
- [x] I support synchronous and asynchronous invocations without changing source-level imports.
- [x] I implement bounded queues and explicit backpressure.
- [x] I make idempotence and retry safety explicit properties of methods (`idempotent` on the method).
- [x] I authenticate callers and validate capabilities before dispatch.
- [x] I make malformed messages fail closed without corrupting the service or caller.

Acceptance:
- [x] I run one unchanged NanoLang client against in-process, local-process, and mock implementations (`tests/nsi_client.nano`, `make test-nsi-runtime`).
- [x] I run one unchanged Nano Forth client through the same generated interface (`tests/nsi_client.fs`).
- [x] I demonstrate an implementation replacement without recompiling the client.
- [x] I test schema evolution across at least one compatible minor version and one rejected breaking version.
- [x] I document the exact trusted computing base for each deployment mode (`docs/NSI_TCB.md`).

### Phase 17 - Capability Runtime and Shared Memory (4.4)

Goal: I will replace ambient authority with explicit, typed, least-privilege
capabilities and move bulk data without weakening isolation.

Capability model:
- [x] I will define unforgeable capability references that cannot be fabricated from integers or host pointers (`src/nsi_cap.c`, `make test-nsi-cap`).
- [x] I will encode object type, service generation, rights, delegation policy, and revocation state in capability tables (`NlCapTable`, `make test-nsi-cap`).
- [x] I will support rights attenuation when delegating capabilities (`nl_cap_attenuate`, `make test-nsi-cap`).
- [x] I will require explicit transfer permission before a service can pass a capability onward (`nl_cap_transfer`, `make test-nsi-cap`).
- [x] I will invalidate stale handles after service restart and prevent generation reuse attacks (`nl_cap_restart`, `nl_cap_invalidate_service`, `make test-nsi-cap test-nsi-fabric`).
- [x] I will map NanoLang resource types to capability ownership and consumption rules (`nl_cap_resource_own`, `nl_cap_resource_consume`, `make test-nsi-cap`).
- [x] I will map Forth handles to validated capability references without exposing raw host addresses (`nl_cap_forth_bind`, `make test-nsi-cap`).
- [x] I will audit every capability creation, delegation, use, revocation, and failure (`nl_cap_audit_*`, `make test-nsi-cap`).

Shared-memory data plane:
- [x] I will keep typed IPC as the control plane and use capability-scoped shared regions for bulk data (`src/nsi_shm.c`, `make test-nsi-shm`).
- [x] I will implement read, write, map, seal, transfer, borrow, return, and revoke rights for shared buffers (`make test-nsi-shm`).
- [x] I will validate offset, length, alignment, lifetime, and direction on every mapping and descriptor (`make test-nsi-shm`).
- [x] I will support zero-copy or bounded-copy paths for audio frames, graphics surfaces, network packets, files, and GPU buffers (`NlShmKind`, `make test-nsi-shm`).
- [x] I will make ownership transfer and completion explicit so buffers cannot be reused while a service owns them (`nl_shm_transfer`, `make test-nsi-shm`).
- [x] I will provide a copying fallback with identical semantics when shared mappings are unavailable (`force_copy`, `make test-nsi-shm`).
- [x] I will benchmark control-message latency, throughput, copies, mappings, and cache effects by payload size (`nl_shm_bench`, `make test-nsi-shm`).

Resource governance:
- [x] I will enforce per-service memory, CPU, handle, queue, file, network, and device budgets (`nl_fabric_set_budget`, `make test-nsi-fabric`).
- [x] I will attach deadlines and cancellation tokens to service requests (`timeout_ms`, `nl_fabric_cancel`, `make test-nsi-fabric`).
- [x] I will define behavior for quota exhaustion, cancellation races, partial results, and abandoned clients (`NL_FAB_ERR_QUOTA`, `NL_FAB_ERR_CANCEL`, idempotent replay, `make test-nsi-fabric`).
- [x] I will expose structured resource accounting without requiring localized prose (`NlAccounting`, `make test-nsi-fabric`).
- [x] I will test hostile clients, forged handles, stale generations, oversized messages, queue floods, and service crashes (`make test-nsi-cap test-nsi-fabric`).
- [x] I will use the SDL editor's walker session and freeze-ISA child as
  capability-runtime clients: buffer and eval rights are unforgeable, eval
  time/memory/queue budgets apply, cancellation aborts a hung eval, and a
  restarted worker invalidates stale session handles
  (`editor.walker` / `editor.freeze` in `src/nsi_fabric.c`, `make test-nsi-fabric`).
  These are fabric-supervised stand-ins. Isolation of `bin/nano_emacs_worker`
  is the 4.1 astronaut side-quest (`docs/NANO_EMACS.md`). I do not claim GNU
  Emacs compatibility.

### Phase 18 - Portable Service Fabric and Supervision (4.4)

Goal: I will host Nano services above ordinary multitasking kernels without
embedding Linux, BSD, or microkernel assumptions in application interfaces.

Supervisor:
- [x] I will implement service discovery, startup ordering, dependency health, readiness, and shutdown (`nl_fabric_start`, `make test-nsi-fabric`).
- [x] I will define restart, retry, fail-request, fail-application, and replacement policies (`NlRestartPolicy`, `make test-nsi-fabric`).
- [x] I will distinguish transient, permanent, protocol, authorization, quota, and implementation failures (`NlFailClass`, `make test-nsi-fabric`).
- [x] I will make retries conditional on declared idempotence and request identity (`request_id`, `make test-nsi-fabric`).
- [x] I will preserve or revoke state explicitly across service upgrade and restart (`nl_fabric_preserve_state`, `make test-nsi-fabric`).
- [x] I will support rolling replacement when interface compatibility permits it (`nl_fabric_replace` + `nl_nsi_compat`, `make test-nsi-fabric`).
- [x] I will propagate deadlines, cancellation, tracing context, and audit identity across service calls (`timeout_ms`, `trace_id`, `audit_id`, `make test-nsi-fabric`).

Portable host adapters:
- [x] I will define a narrow host abstraction for processes, threads, IPC endpoints, shared memory, clocks, entropy, files, networking, devices, and credentials (`NlHost`, `make test-nsi-fabric`).
- [x] I will implement the first complete service-fabric adapter on a mature host kernel selected by measured development cost and security properties (`nl_host_posix` on Darwin/Linux, `docs/NSI_FABRIC.md`).
- [x] I will keep transport and policy behavior identical across host adapters through conformance tests (`nl_host_posix` vs `nl_host_inproc`, `make test-nsi-fabric`).
- [x] I will support local in-process mode for development without weakening production policy declarations (`nl_host_inproc`, `make test-nsi-fabric`).
- [x] I will support process-isolated mode using the host's strongest practical primitives (isolated services get `AF_UNIX` `socketpair`; I do not fork `bin/nano_emacs_worker` in 4.4).
- [x] I will support remote transport without giving remote services local capability authority (`nl_fabric_send_cap_remote` → `NL_FAB_ERR_REMOTE`; no network protocol in 4.4).

Service migration milestones:
- [x] I will migrate logging and diagnostics as the first observable service (`log`, `make test-nsi-fabric`).
- [x] I will migrate filesystem access with path-scoped capabilities (`fs`, `make test-nsi-fabric`).
- [x] I will migrate process execution with executable, argument, environment, and child-control capabilities (`process`, `make test-nsi-fabric`).
- [x] I will migrate networking with endpoint-scoped capabilities (`net`, `make test-nsi-fabric`).
- [x] I will migrate audio with stream-scoped device and shared-buffer capabilities (`audio`, `make test-nsi-fabric`).
- [x] I will migrate graphics and window-system access with surface and input capabilities (`graphics`, `make test-nsi-fabric`).
- [x] I will migrate GPU access with device, queue, memory, shader, and synchronization capabilities (`gpu` typed fabric service, not a CUDA wrap).
- [x] I will migrate Python integration into a typed language-service adapter with no direct Python-object leakage (`python` rejects `PyObject` / host pointers; not a CPython wrap).

Live editor as a fabric client (option C; the dedicated-pipe astronaut is
`docs/NANO_EMACS.md`, not these fabric stand-ins):
- [x] I will stop treating `bin/nano_emacs_worker` as a special-case pipe
  daemon in the fabric. The SDL frame is a client. The walker and freeze-ISA
  processes are supervised services with startup, readiness, restart, and
  replacement policies. I still do not host the walker or NanoISA inside the
  frame, and I still do not route editor eval through the NanoVM FFI
  co-process protocol (`editor.walker` / `editor.freeze`, `docs/NSI_FABRIC.md`,
  `make test-nsi-fabric`). The dedicated-pipe worker is the 4.1 astronaut
  (`docs/NANO_EMACS.md`).
- [x] I will grant the walker only the editor capabilities it needs (bound
  buffer copy, queued chrome commands, echo). I will grant freeze-ISA a
  narrower set: compile/run a module and return a result or error, with no
  `ed_*` unless a later 4.4 checkbox explicitly adds buffer capabilities to
  frozen modules (`make test-nsi-fabric`).
- [x] I will apply Phase 17 quotas and cancellation to eval and freeze: a hung
  walker or `nano_vm` grandchild is cancelled or restarted; the frame stays up;
  generation-bumped session handles fail closed after restart
  (`make test-nsi-fabric`).
- [x] I will use the shared-memory data plane for large buffer bind/return when
  the copying RPC is the measured bottleneck, with the copying fallback when
  mappings are unavailable (`nl_fabric_bind_large`, `make test-nsi-fabric`).
- [x] I will test: kill the walker, kill the freeze child, exhaust eval quota,
  present a stale handle after restart, and confirm the frame survives and
  refuses the stale handle. I will document the editor as a live client of the
  fabric. I will not claim GNU Emacs compatibility (`docs/NSI_FABRIC.md`,
  `make test-nsi-fabric`).

### Phase 19 - Effects, Deployment Policy, and Deterministic Replay (4.5)

Goal: I will connect declared program effects to deployable least-privilege
policy and make nondeterministic execution recordable, replayable, and auditable.

Effects to policy (MAC `task_2bd5c3128983683b78134ad9257b7d3b`):
- [x] I will define the relationship between source effects, module requirements, NanoISA traps, service methods, and capabilities.
      `docs/NSI_EFFECTS.md`, `schema/nsi/effect_map.v0.json`, `make test-nsi-policy`.
- [x] I will emit a complete effect and capability inventory for each program.
      `nl_effect_inventory_from_rows`, `make test-nsi-policy`.
- [x] I will generate a reviewable deployment manifest from that inventory.
      `nl_deploy_manifest_json`, `make test-nsi-policy`.
- [x] I will reject deployments whose granted capabilities do not cover declared effects.
      `nl_deploy_check`, `make test-nsi-policy`.
- [x] I will report unused grants so policy can converge toward least privilege.
      `nl_deploy_unused_count`, `make test-nsi-policy`.
- [x] I will support explicit administrator overrides without silently widening source declarations.
      Override deploys; `source_declarations_widened` stays false. `make test-nsi-policy`.

Record and replay (MAC `task_36c1ce545c4d12a0a2c520dd146ea5f1`):
- [x] I will define a versioned trap journal containing sequence, capability, method, arguments or hashes, result, timing, service generation, and implementation version.
      Journal version 0. `make test-nsi-journal`.
- [x] I will record time, entropy, file, network, user-input, process, GPU, audio, and service nondeterminism at the boundary where it enters a NanoVM.
      `NlJournalKind`, `nl_journal_record`, `make test-nsi-journal`.
- [x] I will replay a NanoVM without invoking original services when the journal contains all required events.
      `nl_journal_replay`, `make test-nsi-journal`.
- [x] I will validate replay order, argument identity, capability identity, and result schema.
      `nl_journal_validate`, `make test-nsi-journal`.
- [x] I will support deterministic service mocks and configurable fault injection.
      `nl_journal_mock`, `nl_journal_inject_fault`, `make test-nsi-journal`.
- [x] I will expose replay checkpoints for debugger reverse navigation where state capture permits it.
      Sequence checkpoints (`nl_journal_checkpoint`, `nl_journal_seek`). I do not snapshot full VM heaps.
- [x] I will sign and hash journals when they are used as audit evidence.
      SHA-256 content hash and HMAC-SHA256 with a deployment key. Not PKI. `make test-nsi-journal`.
- [x] I will define redaction and encryption so replayability does not require publishing sensitive payloads.
      Redacted export drops payloads and keeps hashes. Repeating-key XOR can seal remaining payloads. Not AES.

Observability and provenance (MAC `task_860edd8c08843bf90ad9559d0b821cef`):
- [x] I will assign trace IDs across NanoVM, router, service, and kernel-adapter boundaries.
      `nl_obs_record_span`, `nl_fabric_last_trace`. Host adapter, not a kernel. `make test-nsi-obs`.
- [x] I will emit structured metrics and traces through an implementation-neutral telemetry interface.
      `nl_obs_emit`, `make test-nsi-obs`.
- [x] I will record source, NanoISA module, interface, service implementation, policy, and output provenance.
      `nl_obs_provenance`, `make test-nsi-obs`.
- [x] I will test that localized logs do not alter stable audit fields or replay behavior.
      `nl_obs_localize_log`, `make test-nsi-obs`.

### Phase 20 - One IR: NanoISA-only compilation (5.1)

Goal: I emit one portable product — a verified `.nvm` v2 module — and I treat
C, LLVM, Wasm, RISC-V, and GPU targets as translators of that module. Native
means structured AOT, not a bytecode blob plus `nano_vm`. Contract:
`docs/NANOISA_ONLY.md`. I do not start this phase by deleting
`transpiler.nano`.

- [x] I recorded the 5.1 compilation contract in `docs/NANOISA_ONLY.md`
      (`task_87bcff8dad43407884c4dc9e06837f98`): frontend twice (C seed and
      `src_nano`), one IR, translators as host tools, honest bootstrap by
      comparing `.nvm`, host ABI distinct from `CALL_EXTERN` / `nano_cop`.
      Walk, file fate, linking, debug, and equivalence are in that document.

Ownership and proposal closure:

- [x] I freeze one affine ownership contract before extending either compiler
      (`task_4ac22044ffda9f93b336a85573293bc2`). It resolves the current
      contradictions around at-most-once use versus mandatory cleanup,
      by-value consumption versus borrowing, moves, explicit discard, nested
      resources, collections, branches, loops, returns, and error paths.
- [ ] I replace the C seed's identifier-state prototype with path-sensitive
      ownership analysis and a rule-by-rule conformance corpus
      (`task_c4e2f078cef8c4e461f0de3711c8a2b9`).
      The unmerged `c53e27a7` prototype is not that result: its 256-place
      table silently omits excess owners and its expression walker ignores
      unhandled AST forms. I must cover those boundaries before adopting it.
- [ ] I implement the same resource syntax, analysis, and diagnostics in
      `src_nano`; the self-hosted compiler does not inherit correctness from
      the C seed (`task_20048de825616195b9f2bc492231a851`).
      I first pin declaration, move, consuming-call, use-after-move, and
      unresolved-scope decisions across the C seed and bootstrap stages in
      `make test-affine-selfhost`; path-sensitive control flow follows the C
      seed conformance corpus above rather than inventing a second contract.
- [ ] I encode and verify ownership facts in `.nvm` v2, preserving them through
      serialization, linking, reconstruction, `nvm2c`, and every shipped
      translator (`task_ed70242ac4d83be7b2327da7ece387ad`).
- [ ] I migrate real file, socket, GPU, and capability/service handles only
      after that contract and IR are enforceable
      (`task_d03c232dc067e75cbc2fb2b7fb84ee46`).
- [ ] I gate 5.1 on one affine acceptance matrix across both frontends,
      NanoISA, NanoVM, and AOT C
      (`task_28f2fb4b1f3c8a5ce93df628bb569d76`).
- [ ] I take the bounded One-IR slice of `PASSIVE_PARALLELISM_DESIGN.md` into
      5.1: verified purity/independence, deterministic serial semantics, and
      NanoISA eligibility metadata. Scheduler optimization, async I/O, SoA,
      and hardware speedup claims remain outside this task
      (`task_90b123edcc301b464a031c55e4ba1a11`).
- [x] I postponed `ROW_POLYMORPHIC_RECORDS_DESIGN.md`: the C-seed prototype is
      not a language contract without self-hosted parity, a stable ABI,
      dual-frontend conformance, `.nvm` round trips, VM/AOT equivalence, and
      performance evidence (`task_a39aac00600aa77b55ad92ac70a2d1bf`).

Compiler product:
- [x] I restore trial-deleted child counts for newly deferred VM cycle roots
      (`task_72433f501ddd4736a45e6244c44ae4fa`). Roots queued during collection
      can be reached by the old candidate graph while remaining buffered for
      the next pass. I restore their touched subgraphs before collecting old
      white objects, then normalize deferred root colours in a separate pass.
      My existing positive VM lifecycle/collector suite passes 272579 checks.
      Ordinary supervised execution of my retained 555412-byte full compiler
      shadow module now passes, resolving the baseline assertion under
      `task_fdf43892a1104b1facddc2553af390af`. Fresh current-main closure and
      canonical cutover remain separate; see `evidence/vm-deferred-cycle-counts.md`.
- [x] I publish replaced heap edges before releasing their former values, because release can synchronously collect cycles. My existing VM suite passes 272579 checks, including ownership and cycle checks (`task_493552bb4a7144188299474314bde470`; [evidence](evidence/vm-heap-edge-publication.md)). The complete compiler shadow assertion remains open under `task_fdf43892a1104b1facddc2553af390af`; this repair does not resolve it.
- [x] I retain stack context for assertion failures under VM debug mode or module debug metadata, as I already do for ordinary runtime errors. My VM suite passes 272595 checks, including assertion function/source context and unchanged ordinary output (`task_10e6ea8cf0f34321b9dff21594eb9e8a`).
- [x] I retain checked filesystem foreign signatures in the complete
      dependency shadow closure, including `fs_walkdir` string arrays and
      scalar filesystem operations. I preserve owning artifact identity and
      the existing VM/native ABI without excluding dependency shadows
      (`task_41323f26030d452f92bbdbf69a0a8704`).
- [x] I supervise standalone VM shadow execution with a verified module,
      completion handshake and bounded parent deadline. I reject early exit,
      failed execution and incompatible CLI modes before canonical publication
      can depend on this host prerequisite
      (`task_457e55fa59e146878cee92cabc201b6f`).
- [x] I initialize every generated list parameter annotation before compiled
      metadata extraction. PR487 exposed unset signature pointers in the three
      list parameter allocations; I retain a poisoned-allocation regression
      and pass a fresh three-stage native bootstrap, 24 metadata C methods,
      three import methods and the foreign compiler-path regression
      (`task_402e6b8289fc4f58b79ef5559a68dce3`).
- [ ] I investigate GCC 13’s strict `-O1` sanitizer-build diagnostic for
      `nanocore_export.c` `sbuf_appendf`: inlined `vsnprintf` reports a null
      format string. My normal strict build passes; this is a configuration
      diagnostic, not a demonstrated runtime failure. I retain the log and
      use unsuppressed `-O0` sanitizer checks for the independent metadata task.
      MAC `task_927d53891d204f2fb4e1974eb8c3edc2`.
- [x] I preserve complete recursive function parameter/result annotations
      and nested TypeInfo edges in compiled module metadata, including shared
      graph references. My generated-C roundtrip, 24 C methods, parsed lifetime
      checks, two ordinary-import callback cases and controlled sanitizer
      ownership checks pass. Native callback semantics remain separate. MAC
      `task_1c3c02fe49414bce919adde6666824f5`. Evidence:
      `docs/evidence/module-signature-metadata.md`.
  - [x] I retain complete function and parameter metadata when extracting a
        module, deep-copying and releasing annotations independently of parser
        and environment lifetime. My real extraction/emission route and 100
        controlled copy/free cycles pass; legacy checker leaks remain tracked
        by `task_00c47a5d65d04c48914864ec0de553d6`.
- [ ] I enforce the same full signature check for qualified imported calls:
      `cb.apply` currently accepts a string[][] callback where int[][] is
      required, while an ordinary imported `apply` rejects it. I retain this
      distinct checker gap under `task_e05a42e2e09b47cc9c53fa6923eeeaef`;
      metadata serialization does not claim to repair call dispatch checks.
- [x] I lower selected shadows from the same bound Parser to a separate
      NanoISA module (`task_57fc2c62eb504a579dc2ca37ce72e2e8`), after compiled
      signature metadata lands. I preserve ordered selection, module identity,
      callable user main, local scopes and global initialization; reachable
      unsupported operations fail instead of disappearing from the test scope.
  - [x] I correct the architecture document's stale emitter status and label
        the measured seed / first canonical generation / second canonical
        generation comparison without claiming C-seed/self-hosted codegen equality.
  - [x] I add an explicit first-shadow suffix emitter with empty/invalid
        selection, imported duplicate names, global state and assertion tests,
        retaining existing program-emitter bytecode/parity gates.
  - [x] I verify the emitted shadow module in NanoVM and retain the next full
        compiler-shadow blocker. Supervised execution and the driver cutover
        remain separate prerequisites: deadlines and completion protocol must
        reject an early exit(0), and normal native C shadows remain until then.
- [x] I declare the actual temporary-local capacity of NanoISA `__init__`
      (`task_3ed43e842d704748af03f82883793652`) before adding range expressions.
      I retain initializer-local state and pass exact C-seed bytecode, VM and
      native tests for filled-array globals and count/fill ordering. My full
      emitter gate passes 86 comparisons and 70 Python methods; C-seed/Stage2
      emitters produce identical fixture assembly. See
      `docs/evidence/nanoisa-initializer-frame.md`.
- [x] I restore C-seed NanoVM lexical bindings after a `for` loop
      (`task_be93f56b6a5848ebbd23ef30ccd7bfd4`) while retaining allocated slots.
      Nested same-name variables and bounds referring to outer locals pass
      exact C-seed/self-hosted bytecode and VM/native execution controls.
- [ ] I infer C-seed loop element metadata from inline and computed arrays
      (`task_911317d4234c46049c3dea1a2d0a153d`). The current `AST_FOR` checker
      only reads identifier receivers, so inline string/bool literals give an
      integer loop variable and reject valid bodies. I retain the failing
      expanded probe and use explicit typed locals as the current control.
- [x] I preserve C-seed native range bounds once in source order and compile
      valid early-return array loops without inapplicable vectorization hints
      (`task_c3d168ec8de94d18a1f63445f2c6cea0`). My collision-checked argument
      temporaries preserve user bindings; paired native/VM controls pass.
- [ ] I keep unreachable-code warnings nonfatal in self-hosted checking
      (`task_47e61dea383042808ccc1a0c89bef064`). My `check_block` currently creates W0002
      through the error constructor, so Stage1/Stage2 reject a valid early-return
      fixture accepted by C seed. I preserve the warning and original fixture.
- [x] I snapshot both self-hosted native range bounds before entering the loop
      (`task_8e80a672c285487dbbad07dd1dbfc9b9`). A fresh three-stage bootstrap
      and both paired methods pass across C seed, Stage1 and Stage2, including
      failed-shadow output preservation, source order and nested lexical scope.
      See `docs/evidence/selfhost-native-range-bounds.md` for the explicit
      shadow budget and separate unreachable-warning boundary.
- [ ] I measure my compiler-shadow deadline after range emitter growth
      (`task_628759a2daf743b9bf13c9a7fea2ced0`). A fresh bootstrap reached the default
      ten-second shadow deadline without an assertion diagnostic; explicit
      sixty-second execution advances. I retain both logs, measure the cause
      and keep deadline tests unchanged. A timeout alone is not a correctness failure.
- [ ] I carry contextual element types through nested empty-array appends
      (`task_d5ed194093434b5cbfc2e3ec6bc2d37a`). After range lowering, my full compiler
      shadow probe reaches `substitute_union_field_type` and rejects its nested
      string appends to `[]`. I retain negative type controls and require exact
      C-seed bytecode plus VM/native execution before rerunning the closure.
- [ ] I preserve exact `float_to_string` formatting across interpreter, native and VM
      (`task_29976241a36244f7b0ce4ad75cb10b3f`). My C-seed interpreter/native runtime
      appends `.0` for whole floats; my VM cast and self-hosted native path omit it.
      I retain signed-zero checks independently and require explicit formatting
      parity before claiming this builtin agrees across backends.
- [x] I lower typed F64 opcodes in native AOT with strict float operands
      (`task_fd4c63cf9f3e46f09ece380ce00c7a58`). The exact C-seed/self-hosted scalar fixture
      now verifies and runs in NanoVM and native after PR513. I retain signed
      zero, comparison tags and VM division behavior; the integrated paired
      emitter gate passes 86 comparisons and 81 Python methods.
- [x] I retain unary-minus provenance for exact float lowering
      (`task_ef26778894c441c2b1002128fd0a8c37`), before float dependency-shadow task
      `task_8bc58d33e73f4e24a474d3724d862c94`. I preserve the distinction from
      subtraction explicitly through schema/parser metadata. Negative-zero and
      exact opcode checks pass, as do fresh native bootstrap and three-compiler
      execution; see `evidence/nanoisa-scalar-floats.md`.
- [x] I lower scalar float literals, results and comparisons in dependency
      shadows (`task_8bc58d33e73f4e24a474d3724d862c94`), retaining source lexemes
      and the unchanged canonical dependency-shadow regression. I require
      C-seed bytecode, VM/native execution and output-preservation controls.
      Direct returns of float formatting must use inline builtin lowering,
      rather than ordinary function tail-call resolution. Float call results must
      also match the declared return type before inline or tail-call emission.
- [x] I lower range `for` loops required by my full compiler shadow closure
      (`task_ef6adaee5e3644c8a8218ede4b01e6b3`). I retain range evaluation order,
      lexical scope and nested break/continue/return behavior. My integrated
      gate passes 86 existing comparisons and 74 Python methods; C-seed and
      Stage2-built emitters produce identical fixture assembly. The full
      shadow probe advances to the separately recorded empty-array append
      context boundary without exclusions. See `docs/evidence/nanoisa-range-for.md`.
- [x] I retain a reproducible full VM bytecode bootstrap gate: C-seed compiler
      bytecode is input, then two VM-executed compiler generations compile the
      same clean source and immutable host closure. I compare raw generation
      outputs, verify both and execute a product of the final compiler. Native
      C shadow generation and the NanoISA-only architecture remain separate
      (`task_36ceaa830d7d46ba8a5471326f525aac`).
      Clean pin `1277bce2` produces identical 352,236-byte Stage 1/2 outputs;
      the permanent gate passes in 601.921 seconds with unchanged host hashes
      and a verified executable final-compiler product. Evidence:
      `docs/evidence/vm-bytecode-fixedpoint.md`.
- [x] I measure and bound the VM compiler bootstrap's repeated declaration
      scans and cycle collection (`task_36ceaa830d7d46ba8a5471326f525aac`).
      My historical 1,800-second probe reached native-shadow C generation
      before its diagnostic timeout. Shared declaration classification now
      preserves the tested bytecode/shadow semantics and completes two real VM
      compiler generations in 297.05 and 296.60 seconds on clean `1277bce2`.
      Their raw outputs match exactly; the full permanent gate verifies them,
      checks host closure stability and executes the final compiler's product.
      This closes the bounded VM budget gate, not the full NanoISA-only
      architecture. Evidence: `docs/evidence/vm-bootstrap-budget.md` and
      `docs/evidence/vm-bytecode-fixedpoint.md`.
  - [x] I share the existing linear local-declaration classification across
        typechecking, NanoISA lowering and native-shadow C generation, replacing
        the three repeated transpiler scans without importing the full
        typechecker. I preserve parser index semantics and compare generated
        C/assembly and initialization behavior. My 86 comparison checks and
        64 Python methods pass, as does a fresh three-stage native bootstrap.
        A 40-worker fixture produces identical 37,116-byte generated C before
        and after this extraction; native initialization/mutation checks pass.
        Evidence: `docs/evidence/shared-declaration-ownership.md`.
  - [x] I precompute per-let global ownership once from ordinary/unsafe block
        statements and function parameters, preserving IDs, declaration order,
        initializer effects and exact fixture bytecode. I compare the old
        classification on nested and shadow scopes and measure the compiler
        route; this bounded emitter repair does not close the full VM gate.
        The retained 400-worker VM-emitter workload produces identical 205,847
        assembly bytes in 14.18 seconds before and 0.86 seconds after this
        change. Existing 86 comparison checks and 63 methods, plus the new
        bytecode-emitter ownership method, pass. The full canonical compiler
        still publishes and verifies. Evidence:
        `docs/evidence/selfhost-declaration-ownership.md`.
- [x] I declare truthful parser record, enum and union counts in self-hosted
      NanoISA assembly (`task_14dca63e4c1146e59ae0a1649ba29060`). My emitted
      aggregate IDs already follow parser order; omitted `.types` bounds prevent
      native nominal field proofs in the complete compiler. I preserve those
      proofs and test unused declarations, exact bounds and VM/native execution
      before repeating the complete compiler artifact gate (companion to
      `task_250092bed54749ad988f06af5b88c228`). PR #471 supplies the header;
      86 comparison checks and 63 methods pass, including exact serialized
      counts, C-seed instruction parity, VM/native execution and invalid-bound
      refusal preserving prior output. The full native product and VM compiler
      bootstrap gates remain separate.
- [x] I grow checked assembler symbol, label and patch tables for my complete
      compiler artifact (`task_74ee50b905d242e68c92abf41b427e15`). PR #465 replaces
      fixed capacity with checked growth and explicit allocation failure.
      My 2,691 ISA checks, 210 roundtrip checks, allocation fault harness and
      sanitizer checks pass; complete compiler assembly verifies and roundtrips.
      Evidence: `docs/evidence/assembler-compiler-publication.md`.
- [x] I preserve full compiler string literals through assembly publication
      (`task_c77ac0644fda463a8a2d0ae7dd735908`). PR #465 retains comment markers
      inside quoted literals and dynamically decodes long strings with checked
      allocation. The retained 1,246,064-byte compiler assembly publishes,
      verifies and roundtrips exactly. Evidence:
      `docs/evidence/assembler-compiler-publication.md`.
- [x] I reject incompatible map key/value types at typed boundaries
      (`task_d0438e26b84147cdb9fd16b654c44a6a`). I compare both tags for declared map
      values in returns, bindings, calls, assignments, globals, record construction
      and conditional arms, and preserve prior output after rejection. My boundary
      matrix checks all twelve unequal pairs through both shared-checker drivers;
      field projection metadata remains task `160826784e8a4aa4ac9d5e589a54c814`.
- [x] I lower the string-valued maps required by my compiler
      (`task_d32f896911e1447da9c6b69059f6d6ea`), preserving key/value metadata and ownership.
      Full emission now reaches `HashMap<string,string>`; the retained probe is
      `/tmp/nanolang-edges-fullcompiler-probe.log`.
- [x] I lower string prefix and suffix tests used by my compiler
      (`task_891724de0cb5454eb0a575b099aba188`), preserving exact string operands and boolean results.
      Full emission reaches `str_starts_with` after `is_alnum`; the probe is
      `/tmp/nanolang-alnum-fullcompiler-probe.log`.
- [x] I lower the existing `is_alnum` call required by my compiler
      (`task_9e5bf751b8614b599f0bff8e1bdc2a77`), preserving its precise operand/result contract.
      Full emission reaches it after `str_concat`; the probe is retained in
      `/tmp/nanolang-concat-fullcompiler-probe.log`.
- [x] I select actual nested generic union instances instead of matching an
      enclosing type by substring (`task_b25eef03a0874cf8b0c7f0fa23c05289`).
      My C registration and emission share recursive specialization names;
      `Box<Result<int,string>>` copies and payloads execute under all three
      native compiler stages. Other constructor contexts remain task633.
      Evidence: `docs/evidence/native-nested-generic-identity.md`.
- [x] I lower the existing two-string `str_concat` builtin required by my
      compiler (`task_9ab740ec43654cedb0c1de224856735d`), preserving argument
      order and typed refusals. Full emission reaches it after publication calls.
- [x] I retain native foreign module metadata for root sources outside my
      repository (`task_c6b698326e0f4e6296299ddfdf172ebd`). Source-ancestry root
      discovery preserves `/` as its own parent; when no source root exists,
      native metadata uses my established runtime root. Three native link
      regressions pass; see `docs/evidence/native-external-root-metadata.md`.
- [x] I place native module objects before their link libraries and omit
      already supplied canonical runtime sources (`task_59b46df66b0d480b83cf9cde4a416c07`).
      The artifact facade exposes duplicate cJSON symbols and unresolved SHA256
      during the three-stage bootstrap.
- [x] I lower the exact `nl_nanoisa_assemble_text_save` and
      `nl_nanoisa_last_error` artifact signatures used by compiler publication
      (`task_3560b1472ae64d73b0449b7df6933914`), preserving integer status and
      zero-argument string diagnostics. Full emission reaches these after paths.
- [x] I apply my existing `nano_aot_runtime.o` host-link contract to the
      canonical artifact regression (`task_e56bc32177f04739ae5a8935ead863b1`).
      Its initial native invocation omitted the required runtime exports.
- [ ] I propagate required foreign header search paths through transitive
      C-seed module compilation (`task_c00a44f21a2841068b3be742d1b2ccb9`). My
      module compiler currently omits a dependency manifest's header directory.
- [ ] I reclaim shared static-array containers and their owned elements at an
      alias-safe interpreter lifetime boundary
      (`task_5eba51e216e343549c8ca7846713b6a3`). My current environment cleanup
      ignores static arrays; append alias tests do not establish leak freedom.
- [x] I preserve aliases when my shadow interpreter appends to an empty
      array (`task_adb9b837ce4144a68a9d91ea00749ae0`). I initialize shared
      static storage in place and preserve pop/remove compatibility. See
      `docs/evidence/interpreter-empty-array-alias.md`.
- [x] I let my module builder source pass its own conservative PCH capture
      scan without changing the runtime marker or rejection policy
      (`task_c4dc3150e78c4afda7d58fd490801a05`). Four checks preserve canonical PCH capture and refusal cases.
- [x] I emit enum array literals with integer dynamic storage and preserve
      that representation during native access and append
      (`task_0e901805aea04c8aa644995602612a64`).
- [x] I lower the exact path artifact contracts required by my compiler
      (`task_81e682a57a3d431e845b0f41f140352e`), preserving immutable library
      identities, owner bindings and exact signatures across VM and native
      execution. See `docs/evidence/selfhost-artifact-imports.md`.
- [x] I validate array/list append element types before emitting bytecode
      (`task_e354c7131ebf40e3b177ba5a6cab1aba`); malformed append types now refuse publication.
- [ ] I define boolean List syntax and backend parity before claiming support
      (`task_9556b493260246be8e271b27e29cdf02`); my C parser rejects `List<bool>`.
- [x] I lower boolean arrays required by my compiler, preserving bool/int
      separation and typed bytecode tags (`task_76a8de9b6d84450faf717fd0333e0cee`).
      Fourteen opcode comparisons and VM/native execution pass; see
      `docs/evidence/selfhost-boolean-arrays.md`.
- [x] I keep filled-array compiler temporary slots anonymous so legal source
      names cannot be shadowed by lowering (`task_9f05edac81994a818fd0126c1bf90372`).
- [x] I evaluate native filled-array count then fill exactly once, including
      zero and rejected negative counts (`task_dbdd78cd11a6412aa0904d82ea013434`).
      I capture both operands before the guard; three regression methods pass.
- [x] I reject negative filled-array lengths consistently across backends
      (`task_50c84fa002c14b3fa880d3d260bb8a33`). My evaluator, both bytecode
      frontends and native C constructor reject them after both operand effects.
- [x] I lower typed count/fill `array_new` construction with ordered operands
      (`task_6b4240883d7d461c8266257aaffb2471`). Sixteen exact
      opcode comparisons and VM/native execution pass; see
      `docs/evidence/selfhost-filled-arrays.md`.
- [x] I retain concrete generic union types during C-seed native emission at globals,
      inline call arguments, mutable assignment and nested union construction
      (`task_633f2402ec5944cfba0911a56a9f4eb1`). I retain owned constructor
      and specialization context; executable payload values and nominal refusals
      pass. See `docs/evidence/native-generic-constructor-context.md`.
      My existing nested Result and marker controls remain in
      `tests/test_instantiated_ownership.py`.
- [ ] I substitute nested generic payloads and initialize union globals in
      my self-hosted native stages (`task_85a8db6e186440eaad80442bfc133dd8`).
      The stronger constructor fixture exposes `nl_Box_T` in `Envelope<Plain>`
      and an invalid aggregate `= 0` initializer after a fresh bootstrap.
- [x] I apply concrete generic union constructor context before accepting
      nominal array payloads (`task_dd2be49bc494483f9bb18646a0013055`).
      I reject wrong record identities at local/global, argument, return,
      assignment and nested payload boundaries before output publication.
      Native emission context remains separate. Evidence:
      `docs/evidence/concrete-union-array-contracts.md`.
- [ ] I reconcile the legacy raw `string_from_char` host alias: my AOT
      adapter accepts it, while my VM resolves only canonical
      `vm_string_from_char`. I preserve the canonical emitted contract.
      (`task_dbe0c69106984f22b59c241f3be08919`).
- [x] I track native `string_from_char` allocations for cleanup
      (`task_d2b7c2616e2148a1871c25e1a7ac127d`). Unsuppressed LeakSanitizer
      previously reported 16 leaked bytes from eight calls. My owned-string
      cleanup now passes leak and allocation-balance checks.
- [x] I lower `string_from_char` through the existing scalar string contract
      (`task_292f60fa2cfa431990494d079cc8630c`); eight exact opcode comparisons
      and VM/AOT byte conversion checks pass. Host-result cleanup remains
      separate; see `docs/evidence/selfhost-string-from-char.md`.
- [x] I release returned record-array allocations in native AOT output
      (`task_0916ab0afb014b5984d69fbb11b0432d`). Unsuppressed LeakSanitizer
      previously reported a 14,344-byte leak without setters. Repeated return
      and alias cases now pass leak and allocation-balance checks; see
      `docs/evidence/native-returned-value-cleanup.md`.
- [x] I check fixed nominal record-array assignment contracts across
      annotations, record and union fields, parameters and returns
      (`task_8c736631e97043729bf465a2d6bdc2d5`). My C frontend rejects a
      homogeneous array of the wrong record type and preserves prior output.
      Generic union constructor substitution remains separate. Evidence:
      `docs/evidence/nominal-record-array-contracts.md`.
- [x] I retain nominal record identity on checked array access before local
      type scopes end (`task_911461243652466fb0af1fb706d4ac01`). Direct VM
      projection now selects the right field when records reuse field names;
      typed, `at` and `array_get` fixtures retain nested arrays and shadows.
      Evidence: `docs/evidence/nested-record-projection.md`.
- [x] I preserve record element storage when appending through an array field
      (`task_12805797d36043cd875792788f330520`). I use declared element metadata
      and addressable snapshots. Native execution now passes the original
      nested projection fixture; `docs/evidence/native-record-array-append.md`.
- [x] I represent nonempty record-array literals as dynamic arrays in my C
      seed, preserving child order and nominal element identity
      (`task_a5fc558cbfa34f14b4d580923de4209c`). Native/VM execution,
      alias/order/collision cases and mixed-record rejection pass; evidence:
      `docs/evidence/cseed-record-array-literals.md`.
- [ ] I prevent ordinary user record names from colliding with internal native
      compiler-schema typedefs (`task_1d90fd4257bb43c599a79ab11cfa7aec`). The
      fixture's `NSType` currently produces conflicting C declarations.
- [x] I lower typed list element replacement through my existing array ABI
      (`task_0b7d7221b3fd44d5a8055a84e72397a1`), preserving operand order,
      shared identity and owned payloads. Fourteen opcode comparisons and
      VM/AOT checks pass; AOT returned-array cleanup remains separately
      tracked. See `docs/evidence/selfhost-list-set.md`.
- [x] I align enum-array access type metadata and equality bytecode
      (`task_d2b541318e964ea4946542bffe3990ae`). Typed locals, returns, fields,
      globals and array producers retain enum scalar transport; eight exact
      comparisons and native/VM/AOT execution pass. Evidence:
      `docs/evidence/enum-array-metadata.md`.
- [x] I lower declared enum values/types from preserved self-host AST metadata
      (`task_3ef1df55630e4199b825ff5d8a0043f1`), including signed values and
      exact scalar transport. Fourteen bytecode comparisons and VM/AOT
      execution pass; enum-array access metadata remains separately tracked.
      See `docs/evidence/selfhost-enum-values.md`.
- [x] I propagate unsafe-block termination through C-seed bytecode control flow
      (`task_ed2788c9071249d2a8c8ce3fbec37a74`); recursive analysis now
      preserves exact nested-return bytecode comparisons.
- [x] I preserve lexical local bindings during self-hosted block lowering
      (`task_64cc83f059264bf983a449c3312cde4f`), including unsafe/if/while
      shadowing, outer mutation, and monotonically allocated local slots.
- [x] I lower scoped unsafe blocks through my existing supported host ABI,
      retaining control flow and stack balance
      (`task_750341a5ccb04cffa9b2e0cc92e1f7d6`). Twelve opcode comparisons
      and VM/AOT execution cover nested scopes and loop exits; see
      `docs/evidence/selfhost-unsafe-blocks.md`.
- [x] I retain HashMap field generic types through C-seed builtin inference
      (`task_160826784e8a4aa4ac9d5e589a54c814`). My checked field owns its complete
      map annotation, shared inference borrows declared tags and native emission
      selects the matching helper. I test all four scalar pairs, nested/imported
      receivers, and structured errors that preserve prior artifacts.
      See `docs/evidence/map-field-operations.md`.
- [x] I lower the supported string/int map field in compiler `CollectResult`,
      preserving map identity through record fields, calls and returns
      (`task_95a9982edcd04a1dbb57c45639cd7230`). Ten C-seed bytecode comparisons
      and VM/AOT execution pass; direct receiver C-seed inference remains
      separate. See `docs/evidence/selfhost-map-record-fields.md`.
- [x] I resolve selected generic array payload metadata before checking an
      enclosing typed array boundary (`task_17ace696e309484bbff5a59acc2891db`).
      My first check of `array_push payload.values` now retains the concrete
      record element type. The existing valid constructor fixture and wrong-record
      refusals pass together; see `docs/evidence/selected-array-context.md`.
- [x] I preserve nominal record element metadata through direct array access
      and chained field projection in my C seed, including record-array fields
      (`task_8ddcdb5c824e4a8eb6cc0e2d1bc9ebe3`). Both accessors execute under
      native/VM paths; wrong fields reject. Evidence:
      `docs/evidence/cseed-record-array-projection.md`.
- [x] I retain enum member names and signed explicit/implicit values in
      self-host AST metadata instead of discarding them during parsing
      (`task_e66b50097fe343e3b78e6b750a5c7315`). NanoISA enum lowering depends on
      these fields, without adding a token-rescan implementation.
- [x] I lower compiler `array<Symbol>` and its nested `NSType` record fields,
      retaining record-array tags through construction, empty literals, field
      access and calls (`task_8e7c52236ee645d3b877266acf84ce19`).
      All 86 checks and 29 integration methods pass with VM/AOT evidence.
      `NSType` is a record whose kind is an integer; source enums remain
      separate. `docs/evidence/selfhost-record-arrays.md` records scope.
- [x] I lower finite nested compiler record and typed-list field shapes,
      including `CompilerDiagnostic` and its nested source location. I reject
      cyclic shapes and mismatched nested values; 86 checks and 28 integration
      methods pass with VM/AOT construction, projection, calls and mutation
      (`task_2c3445862de74affa5d2bf624c8c5986`).
      `docs/evidence/selfhost-nested-record-shapes.md` records the boundary.
- [x] I infer scalar array literal tags from computed element types; direct
      `[(int_to_string 7)]` now retains its string element tag. The old module
      fails its VM assertion; repaired C-seed/VM/AOT parity and malformed
      refusals pass within 86 checks and 26 integration methods
      (`task_0a57dab79901497b8e5a6a5ae87df9fb`).
- [x] I lower scalar-array-bearing records required by canonical `MergeResult`,
      preserving typed fields through construction, access, calls and returns
      (`task_468477ec7699402a8dfb92830c5ed40f`). All 86 checks and 25
      integration methods pass, including VM/AOT array identity and empty
      element tags. Nested records and other array element kinds remain
      refused; `docs/evidence/selfhost-array-record-fields.md` records scope.
- [ ] I carry record and map globals through standalone AOT with runtime
      tags, field/map access and lifetime roots; generic scalar global values
      do not establish this support (`task_95796f5f49564ed4a911fd05a1aac5b4`).
      A string-valued global map reproduces the native refusal; VM passes.
      `/tmp/nanolang-map-context-repro/full.nvm` retains the failing module.
- [x] I preserve declared global constructor contexts in my C seed, accepting
      typed map construction and emitting the string tag for empty string
      arrays (`task_026e73d59e9e45b0b732b43883feea9a`).
- [x] I infer array and typed-list element access results in expression
      contexts instead of requiring an explicit intermediate local. I reject
      invalid containers, indices and mismatched typed accessors; 86 checks
      and 23 integration methods pass
      (`task_1c4da3f9cf804bbf95005ad3f603ef26`).
- [ ] I bind imported global values and their selective/qualified aliases
      across canonical checking, native and NanoISA emission without confusing
      same-name declarations from different owners
      (`task_2713a842846b417fbfd6aa4b8059d0dd`). Owner-local storage does
      not establish imported-value binding support.
- [x] I lower supported owner-local globals and ordered runtime initialization
      through `__init__`, retaining every initializer as an executable root
      (`task_5a019d3f83cc43d1beb7ace8c8c7388c`). Scalar/string-array globals
      match my C seed and run in VM/AOT; flat-record/map globals match and run
      in VM. Imported-value binding and aggregate AOT remain separate required
      tasks. All 86 checks and 21 integration methods pass;
      `docs/evidence/selfhost-global-initialization.md` records the boundary.
- [x] I expose an explicit executable API that roots main and preserves its
      bound direct-call closure, including recursion. I refuse globals and
      unresolved function values until their lowering preserves initialization
      and references (`task_ee84482d7ca74757ade0ed4f60728cbc`). Whole-source
      APIs and full native shadow scope remain available. I do not skip named
      imports. All 86 bytecode checks and 19 integration methods pass;
      `docs/evidence/selfhost-program-closure.md` records the bounded scope.
- [x] I escape compiler string literals in NanoISA assembly; after scalar
      array results, real compiler emission first reaches quotes/control
      characters (`task_98ec870924bf4b82b9ba5e2591576488`).
- [x] I keep conservatively emitted scalar-array push helpers valid under
      strict native compilation when only one array kind is pushed
      (`task_2dedfc5f181f41d4951eed8927648306`).
      - [x] I keep the same optional helpers warning-clean under Clang, which
        still diagnoses an unused `static inline` helper under `-Werror`
        (`task_4c32ccf895f5435998cd4982c7dd4087`).
- [x] I preserve declared scalar element tags for empty array locals,
      assignments, arguments and returns; a typed empty string local currently
      differs from my C seed (`task_0443ff4ed6224f5683301055555f4209`).
- [x] I lower supported scalar array results; real compiler emission first
      refuses `array<string>` after scalar conversion lowering
      (`task_4ea96b68ae4b43f7a0cfc16cd7c19649`). I require C-seed and VM/native
      parity for direct/tail returns and preserve unsupported-shape refusal.
      - [x] I expose the lowerer's precise refusal through the driver instead
        of collapsing every failure to `outside the pinned subset`, and prove
        that rejected output is still not published.
- [x] I resolve ordinary and qualified calls through the canonical module
      bindings when lowering a merged Parser, preserving result types, void
      calls, tail returns and raw-source invocation isolation
      (`task_39dd3479c5174b299fb8555c6da8b0af`).
- [x] I lower compiler-required `string_to_int` after map transport; real
      compiler emission first refuses that builtin
      (`task_62d9f8ab389e4299b1b16a33ed591630`). I require C-seed and VM/native parity.
- [x] I expose one parsed-program lowering entrypoint for my canonical
      compiler after module merging and typechecking, retaining the raw-source
      wrapper and testing identical assembly and state reset
      (`task_4dfcc41fa8e147848662f6e3965746d9`).
- [x] I lower the compiler-required `HashMap<string,int>` result/local/call
      shapes and operations with C-seed and VM/native parity
      (`task_2c74662f99d44f7faac1fe7427e44325`). Real compiler emission first
      reaches this unsupported result type after host-import lowering.
- [x] I lower declared host extern imports and the compiler-required `getenv`
      builtin in my self-hosted NanoISA emitter (task_cc32b93698db405696235e96ce8fc194).
      I preserve explicit supported signatures and ABI metadata, compare the
      C-seed output, execute VM/native calls and refuse unsupported imports.
- [x] I lower loop `break` and `continue` in my NanoISA emitter
      (task_942fb307760a4460a49eb49986049103), including nested-loop targets and
      out-of-loop refusal. Twelve named bytecode checks and both VM/native
      executions pass, including nested loops and unreachable statements.
      Real compiler emission passes `parse_options` and next refuses
      `getenv`, which needs host-builtin lowering (2026-09-16).
- [x] I lower void-result functions and zero-result call statements in my
      self-hosted NanoISA emitter (task_862889b9369e4511a465fe838e69224e), including
      bare and implicit returns. I require C-seed bytecode comparisons and
      VM/native execution, without claiming complete compiler emission.
      My gate passes 86 baseline checks, 14 void-function bytecode checks,
      VM/native execution and four result-count refusals (2026-09-16).
      The driver next stops at its unresolved imported `nanoisa_emit_nasm`.
- [x] I infer projected string-field expressions and supported builtin string
      results in my NanoISA emitter (task_8e367aeda3394b0bb1ed1c37f56edeef).
      I select generic bool equality instead of I64_EQ and retain rejection
      of mixed scalar comparisons. Ten new named bytecode checks and both
      VM/native executions pass; four mixed-type forms remain refused.
      Real compiler emission passes `c_source_output_path` and next reaches
      the loop-control boundary above (2026-09-16).
- [x] I lower boolean fields and record-valued returns for my supported flat
      records in the self-hosted NanoISA emitter (task_7eb936723f594d5b81c6cd307dc25e6d).
      I compare C-seed bytecode, execute VM/native construction, projection
      and returned records, retain nested-record refusal, and retry compiler
      emission to identify the next unsupported form. This is a compiler-subset
      slice; matching compiler bytecode across bootstrap stages remains open.
      My focused gate passes 86 existing comparisons and ten new named-function
      checks, both VM/native executions, and nested-record refusal (2026-09-16).
- [ ] I make `--emit-nvm` the self-hosted compiler's only backend output.
      `-o binary` is `nvm2c` then `cc`, a tool pipeline, not a language phase.
- [x] Cut A pin: `src_nano/compiler/nanoisa_codegen.nano` emits `.nasm` for
      integer `add`/`main` (`tests/nanoisa/fixtures/cut_a_add.nano`).
      Function bytecode matches the C seed (`make test-nanoisa-src-nano`,
      10 passed). I still pretty-print C to build the compiler. The full
      dual of `codegen.c` is not this pin.
- [x] Cut A control: the same pin matches `choose` (`if`) and `loop_sum`
      (`while`/`set`/`let mut`) bytecode with the C seed.
      `make test-nanoisa-src-nano` (14 passed).
- [x] Cut A strings: the same pin matches `greeting` (`PUSH_STR`) and
      `glue` (`STR_CONCAT`) bytecode with the C seed. Comparison resolves
      string operands, not only pool indices. `make test-nanoisa-src-nano`
      (18 passed). I still pretty-print C to build the compiler.
- [x] Cut A arrays: the same pin matches `len3` (`ARR_LITERAL`/`ARR_LEN`)
      and `first` (`ARR_GET`) bytecode with the C seed. I still pretty-print
      C to build the compiler. Nested arrays, `array_set`, and `array_push`
      are not this pin. `make test-nanoisa-src-nano` (22 passed).
- [x] Cut A structs: the same pin matches `getx` (`AGG_PACK`/`AGG_GET`)
      bytecode with the C seed for an `int`-field record. I still
      pretty-print C to build the compiler. Nested records, string
      fields, `AGG_SET`, and unions are not this pin.
      `make test-nanoisa-src-nano` (24 passed).
- [x] Cut A bool: the same pin matches `is_pos` (`bool` result, `I64_GT_S`)
      bytecode with the C seed. I still pretty-print C to build the
      compiler. `bool` is i64 0/1 in NanoISA. Option types are not this pin.
      `make test-nanoisa-src-nano` (26 passed).
- [x] Cut A bool ops: the same pin matches `yes`/`no` (`PUSH_BOOL`),
      `invert` (`BOOL_NOT`), `both` (`BOOL_AND`), and `either`
      (`BOOL_OR`) bytecode with the C seed. I still pretty-print C to
      build the compiler. Short-circuit evaluation is not this pin.
      `make test-nanoisa-src-nano` (36 passed).
- [x] Cut A cond: the same pin matches `pick` (`cond` as `JMP_FALSE`/`JMP`
      with one `RET`) bytecode with the C seed. I still pretty-print C
      to build the compiler. Statement `if` stays `choose`.
      `make test-nanoisa-src-nano` (38 passed).
- [x] Cut A print: the same pin matches `say` (`PRINT` of i64), `shout`
      (`PRINTLN` of i64), and `mutter` (`PRINT` of a string) bytecode
      with the C seed. Print does not leave a value. I still
      pretty-print C to build the compiler.
      `make test-nanoisa-src-nano` (44 passed).
- [x] Cut A assert: the same pin matches `prove` (`ASSERT`) bytecode
      with the C seed. I still pretty-print C to build the compiler.
      `make test-nanoisa-src-nano` (46 passed).
- [x] Cut A array_push: the same pin matches `grow` (`ARR_PUSH` of
      `array<int>`) bytecode with the C seed. String arrays are not
      this pin. I still pretty-print C to build the compiler.
      `make test-nanoisa-src-nano` (48 passed).
- [x] Cut A str_contains: the same pin matches `has_hi`
      (`STR_CONTAINS`) bytecode with the C seed. I still pretty-print
      C to build the compiler.
- [x] Cut A int_to_string: the same pin matches `digits`
      (`CAST_STRING`) bytecode with the C seed. I still pretty-print
      C to build the compiler. `make test-nanoisa-src-nano` (52 passed).
- [x] Cut A string arrays: the same pin matches `names`
      (`ARR_LITERAL` tag 5, `ARR_PUSH`, `ARR_LEN`) and `head_s`
      (`ARR_GET` of `array<string>`) bytecode with the C seed.
      Nested arrays stay refused. I still pretty-print C to build
      the compiler. `make test-nanoisa-src-nano` (56 passed).
- [x] Cut A string eq: the same pin matches `same` (`EQ` of two
      strings) and `diff` (`NE` of two strings) bytecode with the
      C seed. I still pretty-print C to build the compiler.
      `STR_EQ` as a separate opcode is not this pin.
      `make test-nanoisa-src-nano` (60 passed).
- [x] Cut A at/str_length: the same pin matches `via_at` (`at` as
      `ARR_GET`) and `slen` (`str_length` as `STR_LEN`) bytecode
      with the C seed. I still pretty-print C to build the compiler.
      `make test-nanoisa-src-nano` (64 passed).
- [x] Cut A str_substring: the same pin matches `slice`
      (`STR_SUBSTR`) bytecode with the C seed. I still pretty-print
      C to build the compiler. `STR_TRIM` and the rest of the string
      library are not this pin. `make test-nanoisa-src-nano` (66 passed).
- [x] Cut A List<int> empty: the same pin matches `blank_l`
      (`list_int_new` as `ARR_NEW 1`, `list_int_length` as `ARR_LEN`)
      bytecode with the C seed. I still pretty-print C to build the
      compiler. `list_int_push` as a void statement is not this pin.
      `make test-nanoisa-src-nano` (68 passed).
- [x] Cut A List<int> push: the same pin matches `grow_l`
      (`list_int_push` as `ARR_PUSH` then `POP`, `list_int_get` as
      `ARR_GET`) bytecode with the C seed. I still pretty-print C to
      build the compiler. Typed lists other than `List<int>` are not
      this pin. `make test-nanoisa-src-nano` (70 passed).
- [x] Cut A char_at: the same pin matches `ch` (`char_at` as
      `STR_CHAR_AT`) bytecode with the C seed. I still pretty-print C
      to build the compiler. Out-of-range index is `-1`, matching the
      VM. `str_char_at` is the same opcode.
      `make test-nanoisa-src-nano` (72 passed).
- [x] Cut A List<string> empty: the same pin matches `blank_s`
      (`list_string_new` as `ARR_NEW 1`, `list_string_length` as
      `ARR_LEN`) bytecode with the C seed. I still pretty-print C to
      build the compiler. The C seed uses tag 1 for every `list_T_new`.
      `make test-nanoisa-src-nano` (76 passed).
- [x] Cut A List<string> push: the same pin matches `grow_s`
      (`list_string_push` as `ARR_PUSH` then `POP`, `list_string_get`
      as `ARR_GET`) bytecode with the C seed. I still pretty-print C to
      build the compiler. Record lists and `list_T_set` are not this pin.
- [x] Cut A string-field records: the same pin matches `get_s`
      (`AGG_PACK`/`AGG_GET` of a record with an `int` field and a
      `string` field) bytecode with the C seed. I still pretty-print C
      to build the compiler. Nested records and bool fields are not
      this pin. Lists of records are not this pin.
      `make test-nanoisa-src-nano` (78 passed).
- [x] Cut A record lists: the same pin matches a `List<Tok>` identity
      (`list_Tok_new` as `ARR_NEW 1`, `list_Tok_push` as `ARR_PUSH`
      then `POP`). Nested records stay refused. `list_T_set` is not
      this pin. I still pretty-print C to build the compiler.
      `make test-nanoisa-src-nano` (80 passed).
- [ ] I implement NanoISA lowering in `src_nano` as the dual of
      `src/nanovirt/codegen.c` for the compiler subset, not only the pin.
- [ ] I compile `src_nano` to `.nvm` with the C seed, then with the
      self-hosted emitter.
- [ ] Stage 3 compares `stage1.nvm` and `stage2.nvm`. Matching native
      binaries from `nvm2c`+`cc` is a translator test, kept separate.
- [ ] I freeze `transpiler.nano` as bootstrap-only once the emitter compiles
      the compiler, then I delete it from the product compiler. Git history
      keeps it.
- [ ] I rename `CompilerPhase_PHASE_TRANSPILER`; the pipeline ends at NanoISA.

`nvm2c` as canonical AOT:
- [ ] I make C11 the canonical ahead-of-time portability backend from NanoISA.
      A generated process does not require `nano_vm`, `nano_cop`, or
      `nano_vmd` to compute.
- [x] My C seed emits `ARR_NEW TAG_STRUCT` for an empty `List<record>`, so its
      NanoISA element kind does not depend on a later push
      (`task_2ef5249b949443adab10c45e16a70c5b`, `make test-nanovirt`).
- [ ] `nvm2c` covers the compiler subset: functions, structs, loops, arrays,
      strings, modules, and a declared host ABI for `extern`.
- [ ] I map `CALL_EXTERN` to that host ABI or I refuse the module. I do not
      emit a co-process client and call it AOT.
- [ ] `wrapper_gen` remains a packaged-interpreter path. It is not "native"
      in 5.0 documentation or CLI defaults.
- [x] I ship a `nvm2c` tool (seed in C). `make nvm2c` writes `bin/nvm2c`.
      `make test-nvm2c` runs the library and the CLI. Generated C does not
      name `nano_vm`. I may later write `nvm2c` in myself.
- [x] `nvm2c` translates i64 comparisons, `JMP`/`JMP_FALSE`, and `TAIL_CALL`.
      `choose` and `loop_sum` from the Cut A pin compile and run without
      `nano_vm`. Goto is the translator fallback; recovered `if`/`while` is
      not this item. `make test-nvm2c` (65 passed).
- [x] `nvm2c` translates `PUSH_STR`, `STR_CONCAT`, and `STR_LEN`.
      `greeting` and `glue` from the Cut A pin compile and run without
      `nano_vm`. I do not link `nano_vm`. Embedded NULs and the rest of
      the string library stay refused. `make test-nvm2c` (89 passed).
- [x] `nvm2c` translates `ARR_LITERAL`, `ARR_GET`, and `ARR_LEN` for
      `array<int>`. `len3` and `first` from the Cut A pin compile and run
      without `nano_vm`. Nested arrays and the rest of the array library
      stay refused. `make test-nvm2c` (102 passed).
- [x] `nvm2c` translates `AGG_PACK` and `AGG_GET` for `int`-field records.
      `getx` from the Cut A pin compiles and runs without `nano_vm`.
      Nested records, string fields, `AGG_SET`, variants, and tuples
      stay refused. `make test-nvm2c` (110 passed).
- [x] `nvm2c` translates `bool` results as i64 0/1. `is_pos` from the
      Cut A pin compiles and runs without `nano_vm`.
      `make test-nvm2c` (120 passed).
- [x] `nvm2c` translates `PUSH_BOOL`, `BOOL_NOT`, `BOOL_AND`, and
      `BOOL_OR` as i64 0/1. `yes`, `invert`, `both`, and `either` from
      the Cut A pin compile and run without `nano_vm`.
      `make test-nvm2c` (145 passed).
- [x] `nvm2c` runs Cut A `pick` (`cond` shape) without `nano_vm`.
      Join points copy temps so both arms share one `RET`. Goto is
      the translator fallback. `make test-nvm2c` (155 passed).
- [x] `nvm2c` runs Cut A `say`, `shout`, and `mutter` without `nano_vm`.
      Printing arrays and records stays refused.
      `make test-nvm2c` (177 passed).
- [x] `nvm2c` runs Cut A `prove` (`ASSERT`) without `nano_vm`.
      False asserts abort the process. `make test-nvm2c` (188 passed).
- [x] `nvm2c` runs Cut A `grow` (`ARR_PUSH` of `array<int>`) without
      `nano_vm`. String arrays stay refused.
      `make test-nvm2c` (196 passed).
- [x] `nvm2c` runs Cut A `has_hi` (`STR_CONTAINS`) without `nano_vm`.
- [x] `nvm2c` runs Cut A `digits` (`CAST_STRING`) without `nano_vm`.
      Casting arrays stays refused. `make test-nvm2c` (214 passed).
- [x] `nvm2c` runs Cut A `names` and `head_s` (`array<string>`)
      without `nano_vm`. Nested arrays stay refused.
      `make test-nvm2c` (225 passed).
- [x] `nvm2c` runs Cut A `same` and `diff` (`EQ`/`NE` of strings)
      without `nano_vm`. Array equality stays refused.
      `make test-nvm2c` (243 passed).
- [x] `nvm2c` runs Cut A `via_at` and `slen` without `nano_vm`.
      `make test-nvm2c` (253 passed).
- [x] `nvm2c` runs Cut A `slice` (`STR_SUBSTR`) without `nano_vm`.
      `STR_TRIM` stays refused. `make test-nvm2c` (261 passed).
- [x] `nvm2c` runs Cut A `blank_l` (`ARR_NEW` of `array<int>`) without
      `nano_vm`. `make test-nvm2c` (266 passed).
- [x] `nvm2c` preserves array identity across `ARR_PUSH` then `POP` so
      void `list_int_push` mutates the local the way the VM heap does.
      Cut A `grow_l` exits 7 without `nano_vm`.
      `make test-nvm2c` (271 passed).
- [x] `nvm2c` runs Cut A `ch` (`STR_CHAR_AT`) without `nano_vm`.
      Index 0 of `"hi"` exits 104. Out of range is `-1`.
      `make test-nvm2c` (281 passed).
- [x] `nvm2c` runs Cut A `blank_s` (`ARR_NEW` of a string list) without
      `nano_vm`. Empty length is 0.
- [x] `nvm2c` preserves string-array identity across `ARR_PUSH` then
      `POP` so void `list_string_push` mutates the local the way the VM
      heap does. Cut A `grow_s` length is 2 without `nano_vm`. The C
      seed still emits `ARR_NEW 1`; I classify by the pushed value.
      `make test-nvm2c` (291 passed).
- [x] `nvm2c` runs Cut A `get_s` (string field of a record) without
      `nano_vm`. Nested records stay refused.
      `make test-nvm2c` (299 passed).
- [x] `nvm2c` runs a Cut A list of records (`ARR_PUSH` of `nrec_t`)
      without `nano_vm`. Nested records stay refused.
      `make test-nvm2c` (306 passed).
- [x] I bound every append while formatting native direct calls and size the
      call expression for the full supported arity. Wide ordinary and tail
      calls pass the focused `nvm2c` sanitizer test.
- [x] I keep my generated map-root support warning-clean under strict C11 on
      Darwin even when a program has no root-reset or collection safepoint.
      `task_de5e8502bf025d94b103fa4a4ce734db` repairs the release-gate
      regression where Apple Clang rejects unused `nroot_reset` and
      `nmap_collect_if_needed` helpers. The focused no-safepoint fixture and
      all 2,389 structured-C checks pass on Darwin (2026-09-17).
- [x] I keep dynamic record-array support warning-clean when a program creates
      an empty record array but never grows it. The same Darwin audit task
      records the post-PR483 `nrarr_reserve` strict-Clang regression separately
      from record-array allocation and ownership semantics. The empty-array
      fixture and all 2,390 structured-C checks pass on Darwin (2026-09-17).

Module richness:
- [ ] I store local names, not only slot numbers.
- [ ] I preserve frontend purity, affine-use, generic, effect, and
      exhaustiveness facts as NanoISA metadata.
- [ ] I recover structured `if`/`while`/`return` from `JMP` for
      reconstruction. Goto is a translator fallback, not the claim.
- [ ] I define general and restricted compute profiles with verifier-enforced
      feature sets. GPU targets use the restricted profile.

Reconstruction:
- [ ] I spike a second high-level surface from the same `.nvm` (NanoLang or
      another HLL). If that surface is only an interpreter, the spike failed.
- [ ] I publish the finding in `docs/NANOISA_HL_ROUNDTRIP.md`: sufficient,
      insufficient, or blocked on named metadata.

Other translators:
- [ ] I implement LLVM IR as a NanoISA translator rather than a NanoLang AST backend.
- [ ] I implement WebAssembly as a NanoISA translator rather than a NanoLang AST backend.
- [ ] I reintroduce LLVM and Wasm only behind those translators, with full
      applicable-language coverage.
- [ ] I evaluate JVM bytecode, SPIR-V, PTX, OpenCL, and Metal as NanoISA
      translators. Existing `nanoc --target ptx` / `opencl` AST backends
      become translators under the restricted profile or they leave the
      product compiler.
- [ ] I run the same NanoISA module through each applicable target for
      semantic-equivalence testing (VM, AOT C, and each shipped translator).

Acceptance (from `docs/NANOISA_ONLY.md`):
- [ ] `src_nano` emits `.nvm` as its only compiler product.
- [ ] `nvm2c` builds a process that does not link `nano_vm`.
- [ ] Stage 1 and Stage 2 `.nvm` files match.
- [ ] A pinned suite matches on `nano_vm` and AOT C.
- [ ] `transpiler.nano` is gone from the product compiler.

### Phase 22 - Hardened Operating Environment (6.0)

Goal: I will package the language, VM, services, capabilities, policy, and
supervision layers as a complete operating environment. Kernel choice remains
a deployment decision below the stable Nano service ABI. This follows 5.1
(NanoISA-only compilation). Signing and kernel adapters are 6.0, not 4.0
and not 5.1.

System image and lifecycle:
- [ ] I will define signed manifests for NanoISA modules, service interfaces, implementations, capabilities, and policy.
- [ ] I will build deterministic system images from a locked dependency and service graph.
- [ ] I will verify signatures, hashes, interface compatibility, and policy before activation.
- [ ] I will sign `.nvm` module artifacts themselves, not only their manifests, carrying the signature and its key identifier in a dedicated section of the v2 module format.
- [ ] I will verify a module's signature at load, in `nvm_deserialize`, immediately after the existing CRC32 check and over the same byte range, so every loader inherits it from one place.
- [ ] I will make enforcement a runtime policy rather than a build-time one -- off, warn, or require -- so the same binary can refuse unsigned modules in a locked-down deployment and accept them on a development machine without rebuilding.
- [ ] I will define where verification keys come from and how trust in them is established, since a signature check is only worth the provenance of the key that satisfies it.
- [ ] I will implement atomic service and system upgrades with rollback.
- [ ] I will implement health monitoring, crash-loop control, degraded operation, and recovery policy.
- [ ] I will define administrative capabilities for inspection, update, backup, restore, and shutdown.
- [ ] I will make boot, startup, steady state, upgrade, failure, and shutdown auditable.

Scoping note on module signing: this is deliberately 6.0 work, not 4.0 or 5.0. The
mechanism is cheap -- Ed25519 signing and verification are already available
from the OpenSSL that every binary links today, the v2 module format's section
directory and feature bits have room for a signature without a format break,
and `nvm_deserialize` already has the gate where a check belongs, right after
the CRC32. What is not cheap is deciding where verification keys come from and
who is trusted to issue them, and that question belongs with the capability and
policy work rather than ahead of it. Signing an artifact nobody can establish
provenance for buys very little.

The CRC32 in the current format is an integrity check, not an authenticity one:
it detects corruption and is trivially recomputed by whoever edited the file.
It should not be mistaken for a security property in the meantime.

Kernel and isolation adapters, deliberately last:
- [ ] I will specify the minimum kernel contract for address spaces, scheduling, IPC, shared memory, clocks, entropy, interrupts, credentials, and device access.
- [ ] I will implement and test a Linux adapter using processes, Unix sockets, descriptor passing, shared memory, namespaces, seccomp, and Landlock where available.
- [ ] I will implement and test a 5BSD adapter using its capability descriptors, process and jail isolation, MAC policy, keyvault, and auditable event interfaces where available.
- [ ] I will evaluate and prototype an seL4 adapter using protection domains and capability IPC.
- [ ] I will evaluate FreeBSD Capsicum and other capability-oriented hosts as additional adapters rather than forks of the service model.
- [ ] I will map graphics, audio, GPU, network, storage, and input hardware privileges to narrowly scoped services on each host.
- [ ] I will run the same service-contract conformance suite on every supported kernel adapter.
- [ ] I will document which properties are enforced by NanoISA verification, NanoVM, the capability runtime, the service supervisor, and the host kernel.

Security validation:
- [ ] I will threat-model bytecode, verifier, VM, router, service, shared-memory, capability, update, replay, and kernel-adapter boundaries.
- [ ] I will fuzz every untrusted binary and message parser.
- [ ] I will test compromised services, confused deputies, capability leaks, stale handles, replay attacks, rollback attacks, denial of service, and malicious peripherals.
- [ ] I will use formal methods for the portable capability and message semantics where the model is tractable.
- [ ] I will not call the environment secure merely because its components are isolated; I will state which properties are tested, proved, inherited, or assumed.

Release acceptance:
- [ ] I will boot or launch a complete signed service graph on at least two materially different kernel substrates.
- [ ] I will run unchanged NanoLang and Nano Forth applications across those substrates.
- [ ] I will demonstrate least-privilege filesystem, network, graphics, audio, GPU, and Python services.
- [ ] I will demonstrate crash containment, restart-safe handle invalidation, upgrade rollback, deterministic replay, and auditable provenance.

### Phase 21 - Multi-Language NanoISA Laboratory (4.6)

This 4.6 phase sits after 5.0/6.0 in this file for historical section
order. The release graph still has 4.6 before 5.0.

Goal: I will test whether NanoISA is genuinely language-neutral by compiling a
small set of deliberately different languages to the same verified IR. I will
not collect syntax for its own sake. Each frontend must expose a distinct
architectural weakness or prove a distinct capability.

Shared frontend contract (MAC `task_e62d1cd35b49296604012df95de7911b`):
- [x] I define the shared input, output, source-location, typed-function,
      layout, constant, import, effect, capability, diagnostic, and metadata
      interface in `docs/FRONTEND_CONTRACT.md`.
- [x] I require every frontend to emit the same versioned NanoISA module,
      round-trip it through the shared serializer, and pass the same verifier.
- [x] I give every frontend the same typed route to service contracts,
      capabilities, isolated FFI, debugger metadata, profiling, and target
      translators.
- [x] I put language-specific parsing, desugaring, and type analysis before the
      module boundary and language-neutral verification and optimization after
      it.
- [x] I preserve purity, exhaustiveness, ownership, inferred-type, and effect
      facts as optional namespaced metadata that cannot define behavior.
- [x] I require a bounded feature subset, architectural pressure, exclusions,
      tests, shared fixture, and completion claim before a frontend starts.
- [x] I reject frontend-specific opcodes unless a reviewed proposal defines a
      reusable primitive and its behavior across every shared NanoISA tool.
- [x] I define the cross-frontend conformance harness: equivalent programs use
      shared NanoISA libraries and versioned services, with explicit exclusions
      for fixtures outside a bounded language subset.
- [x] I will define a frontend interface for source locations, typed functions, layouts, constants, imports, effects, capabilities, and diagnostics.
      `NlFrontendFacts`, `docs/NANOISA_FRONTEND.md`, `make test-frontend-contract`.
- [x] I will require every frontend to emit the same versioned NanoISA module format and pass the same verifier.
      `nl_frontend_accept` calls `nvm_verify`; format_version must be v2.
- [x] I will give every frontend access to the same service contracts, capability model, FFI isolation, debugger metadata, profiler, and target translators.
      `nl_frontend_toolchain`.
- [x] I will separate language-specific desugaring and type analysis from language-neutral NanoISA optimization.
      `nl_frontend_phase_is_language_specific`.
- [x] I will preserve language-specific facts such as purity, exhaustiveness, ownership, and effect information as optional metadata.
      Optional fields on `NlFrontendFacts`; unknown effects fail closed.
- [x] I will define bounded implementation and test goals before starting each frontend.
      `nl_frontend_goal`; Scheme, ML, Actor, Dataflow, Object, Shell, and Logic
      are implemented (`make test-scheme`, `make test-ml`, `make test-actor`,
      `make test-dataflow`, `make test-object`, `make test-shell`,
      `make test-logic`).
- [x] I will reject frontend-specific opcodes unless they represent a reusable primitive that survives review against the other languages.
      `nl_frontend_opcode_allowed` is exactly `isa_get_info`.
- [x] I will run cross-frontend programs against shared NanoISA libraries and service interfaces.
      NanoLang and Forth callers of one library via `nl_frontend_accept_linked`.

Nano Scheme (MAC `task_6647a64cc76edac6e3d0f62c228d98c4`):
- [x] I will implement a small Scheme frontend as the first post-Forth language experiment.
      `src/scheme/scheme.c`, `docs/SCHEME.md`, `make test-scheme`.
- [x] I will support lexical scope, closures, first-class procedures, recursive data, and interactive evaluation.
- [x] I will implement proper tail calls and verify constant frame depth under deep recursion.
      `(sum 10000 0)` at frame depth <= 3.
- [x] I will evaluate continuations only after ordinary closure and exception semantics are stable.
      Closures are tested; exceptions are not. `call/cc` fails closed.
- [x] I will use Scheme to stress allocation, callable representation, tail calls, dynamic values, and live code publication.
      1000-cons tail list, `CALL_INDIRECT` closures, session `define` replacement.
- [x] I will run a pinned subset of a recognized Scheme test suite and document intentional exclusions.
      `tests/scheme/test_scheme.c`, `tests/scheme/r5rs_pin.scm`, `docs/SCHEME.md`.

Nano ML (MAC `task_3eed929292a80ed58dd3a8db1ed701b6`):
- [x] I will implement a compact ML-family frontend with static inference, algebraic data types, pattern matching, immutable values, and higher-order functions.
      `src/ml/ml.c`, `docs/ML.md`, `make test-ml`.
- [x] I will use ML to test generic instantiation, aggregate layouts, exhaustive matching, closures, and module signatures.
      `id : a -> a`; pair `fst`; exhaustive `option`; `fn` / `fun` closures; `signature`.
- [x] I will preserve inferred type and exhaustiveness facts in NanoISA metadata where target-independent optimization can use them.
      Schemes interned in the module string pool; `NlFrontendFacts.exhaustiveness = 1`.
- [x] I will run shared aggregate and service-interface programs under both NanoLang and Nano ML.
      NanoLang-labeled assembler `CALL_MODULE` of `ml_fst`; `nl_frontend_accept_linked`.

Nano Actor (MAC `task_0850b9adc62c593b8e4e180e070efcee`):
- [x] I will implement an Erlang, Elixir, and Gleam-inspired actor frontend.
      `src/actor/actor.c`, `docs/ACTOR.md`, `make test-actor`.
- [x] I will support isolated actors, typed mailboxes, pattern-matched messages, monitors, links, supervision trees, deadlines, and cancellation.
      Isolated `VmState` per actor; int tag+payload mailboxes; `monitor`/`link`;
      `supervise one_for_one`; `recv after 0`; `cancel`. Nonzero deadlines fail closed.
- [x] I will first execute actors as isolated NanoVM contexts in one host process.
      Each actor has its own `VmState`; handlers are verified NanoISA in a shared module.
- [ ] I will then move unchanged actors across service-process boundaries through the Phase 18 transport.
- [x] I will test crash containment, mailbox ordering, supervision, hot code replacement, and restart-safe capabilities.
      `tests/actor/test_actor.c`; `cap:` fails closed so capabilities do not cross restarts.

Nano Dataflow (MAC `task_9cb85a523c197b9e2c80ddcffe9ed31a`):
- [x] I will implement a deterministic dataflow and workflow frontend with typed nodes, streams, backpressure, and explicit effects.
      `src/dataflow/dataflow.c`, `docs/DATAFLOW.md`, `make test-dataflow`.
- [ ] I will map graph dependencies to local, service-process, and remote scheduling without changing program semantics.
      `place local` is a no-op; `place remote` fails closed. Phase 18 transport is not wired.
- [x] I will use dataflow programs to test shared-memory bulk transfer, provenance, replay, cancellation, retries, and parallel determinism.
      Bounded integer bulk along edges (copy, not NSI shm maps); interned `feed` journal;
      replay; `cancel`; `retry`; fifo vs reverse ready-set.
- [x] I will record every external input required to reproduce a completed workflow.
      Each `feed` is interned as `.string "feed <port> <value>"`.

Nano Object (MAC `task_90023c92e9fb3841aab9fcc71d8cf90d`):
- [x] I will implement a small Smalltalk-like object frontend with message dispatch, object identity, mutable graphs, reflection, and live method replacement.
      `src/object/object.c`, `docs/OBJECT.md`, `make test-object`.
- [x] I will use it to test dynamic dispatch, inline caches, layout evolution, callable handles, image persistence, and debugger reflection.
      Host IC per class+selector; `extend`; `handle`/`sendvia`; `nl_object_last_image`; `classof`/`slots`.
- [x] I will measure specialization and quickening without exposing cache-specific operations in portable NanoISA.
      `nl_object_last_ic_hits` / misses; no cache opcodes.

Nano Shell (MAC `task_ee91ee94749200ab6309e6c05df3dd61`):
- [x] I will implement a capability-safe orchestration shell using structured values rather than text-only pipelines.
      `src/shell/shell.c`, `docs/SHELL.md`, `make test-shell`.
- [x] I will expose processes, files, networks, services, streams, cancellation, and remote execution only through explicit capabilities.
      `need files|proc|net|service|stream|remote`; missing cap fails closed; granted cap still refuses host effects in this subset.
- [x] I will preserve typed values across pipelines and make text parsing an explicit adapter.
      i64 pipes; `"3" | add 1` fails; `parse "3" | add 1` is 4.
- [ ] I will use Nano Shell as the administrative language for service graphs only after capability and policy enforcement are complete.

Nano Logic (MAC `task_69fc7f6660a1976f10606a42d78fd264`):
- [x] I will implement a bounded Datalog or logic frontend for declarative authorization, dependency, and policy rules.
      `src/logic/logic.c`, `docs/LOGIC.md`, `make test-logic`.
- [x] I will support facts, rules, unification appropriate to the selected subset, queries, and deterministic fixed-point evaluation.
      Ground int facts; Horn rules; `lg_unify` / `I64_EQ`; naive least fixed-point.
- [x] I will use it to test choice points or tabling only when those mechanisms are justified by the selected language subset.
      They are not justified. Naive iteration over a finite EDB is enough.
- [x] I will compile deployment and capability policy queries to verified NanoISA or a documented restricted profile.
      Restricted profile: `grant`/`allow` as ordinary predicates; `query allow 7`. Not NSI documents.

Frontend matrix and demonstrations (MAC `task_92c497c72b7aa1fc993d666f66843759`):
- [x] I will maintain a matrix showing how NanoLang, Nano Forth, Nano Scheme, Nano ML, Nano Actor, Nano Dataflow, Nano Object, Nano Shell, and Nano Logic exercise typing, calls, closures, stacks, matching, concurrency, services, replacement, and replay.
      `docs/FRONTEND_MATRIX.md`, `make test-frontend-matrix`.
- [x] I will implement one shared service interface consumed from NanoLang, Nano Forth, Nano Scheme, and Nano ML.
      Shared `add` library via `nl_frontend_accept_linked`.
- [x] I will implement one supervised service in Nano Actor and orchestrate it from Nano Shell.
      `need service` / `service 1` with `nl_shell_set_service` bound to supervised Echo.
- [x] I will apply Nano Logic policy to that service without embedding policy semantics in the application.
      `query allow 1` is evaluated before Shell starts Echo. Deny keeps Echo stopped.
- [x] I will run equivalent computation fixtures across applicable frontends and compare their NanoISA behavior and results.
      Integer 5 from add/plus/pipe/Ping across applicable frontends.
- [x] I will publish measured compile time, module size, instruction mix, allocation, call behavior, and execution time for each frontend.
      Matrix test prints compile_ns, eval_ns, code_size, fns, ins, calls, strings.
- [x] I will keep NanoLang as my native language and describe the others as bounded architecture probes until their own conformance goals are met.
      `nl_frontend_goal(NL_FE_NANOLANG)` pressure names native; docs/FRONTEND_MATRIX.md.

## Project Vision

I am a minimal, LLM-friendly programming language. I exist to fulfill these goals:
- I compile to C for performance and portability.
- I require shadow tests for all code I compile.
- I support both infix (a + b) and prefix ((+ a b)) notation for operators.
- I compile myself.

## Current Status: Phase 11 Complete - Formally Verified + Virtual Machine

Status: PRODUCTION-READY - I have achieved self-hosting, my virtual machine backend is functional, and my core is formally verified.

Current Capabilities:
- 100% Self-Hosting - My compiler compiles itself. I have verified this through a 3-stage bootstrap.
- NanoISA Virtual Machine - I have a custom 178-opcode ISA with a .nvm bytecode format and process-isolated FFI.
- Formally Verified - I have proved my type soundness, progress, determinism, and semantic equivalence in Coq using zero axioms.
- I have a complete compilation pipeline: lexer, parser, type checker, and transpiler or VM codegen.
- I execute shadow tests during compilation using my compile-time evaluator.
- I provide multiple executables: bin/nanoc (C compiler), bin/nano_virt (VM compiler), and bin/nano_vm (executor).
- My type system includes primitives, arrays, structs, enums, unions, generics, tuples, first-class functions, and affine types.
- I have 66 standard library functions covering math, strings, binary strings, arrays, I/O, OS, checked math, and generics.
- I have over 30 FFI modules, including SDL, ncurses, OpenGL, curl, readline, and a Python bridge.
- I have over 90 working examples, ranging from games and graphics to simulations and data analytics.
- I have over 221 test files covering unit, integration, regression, negative, performance, ISA, and VM tests.
- I have produced over 121 markdown files of documentation.
- I consist of approximately 6,170 lines of Coq proofs and 11,000 lines of VM implementation.

## Phase 1 - Lexer Complete

Goal: Transform source text into tokens.

Deliverables:
- [x] Token definitions (nanolang.h)
- [x] Lexer implementation (src/lexer.c - ~300 lines)
- [x] Error reporting with line numbers
- [x] Test suite for lexer (all examples tokenize correctly)
- [x] Handle comments (# style)
- [x] Handle string literals
- [x] Handle numeric literals (int and float)

Completion Date: September 29, 2025

Success Criteria: All met
- I can tokenize all example programs.
- I provide clear error messages for invalid input.
- I work with 15/15 examples.

## Phase 2 - Parser Complete

Goal: Transform tokens into Abstract Syntax Tree (AST).

Deliverables:
- [x] AST node definitions (nanolang.h)
- [x] Recursive descent parser (src/parser.c - ~680 lines)
- [x] Prefix and infix notation support
- [x] Error recovery
- [x] Test suite for parser (all examples parse correctly)
- [ ] Pretty-printer (not implemented - low priority)

Completion Date: September 30, 2025

Success Criteria: All met
- I can parse all example programs.
- I produce a valid AST.
- I provide helpful error messages.
- I work with 15/15 examples.

## Phase 3 - Type Checker Complete

Goal: Verify type correctness of AST.

Deliverables:
- [x] Type inference engine (src/typechecker.c - ~500 lines)
- [x] Type checking rules for all operators
- [x] Symbol table with scoping
- [x] Scope resolution
- [x] Error messages for type errors
- [x] Test suite for type checker (all examples type-check correctly)

Completion Date: September 30, 2025

Success Criteria: All met
- I catch all type errors.
- I reject invalid programs.
- I accept valid programs.
- I provide clear error messages.

## Phase 4 - Shadow-Test Runner & Interpreter Complete

Goal: Execute shadow tests during compilation and provide full interpretation.

Deliverables:
- [x] Test extraction from AST
- [x] Complete interpreter for shadow tests and programs (src/eval.c - ~450 lines)
- [x] Assertion checking
- [x] Test result reporting
- [x] Function call interface
- [x] Test suite for interpreter (15/15 examples pass)

Completion Date: September 30, 2025

Success Criteria: All met
- I execute all shadow tests.
- I report failures clearly.
- I support full program interpretation.
- I execute quickly.

## Phase 5 - C Transpiler Complete

Goal: Transform AST to C code.

Deliverables:
- [x] C code generation (src/transpiler.c - ~380 lines)
- [x] Runtime library integration
- [x] Built-in function implementations
- [x] Memory management (C standard library)
- [x] Test suite for transpiler (15/15 examples compile and run)
- [ ] C code formatter (basic formatting, could be improved)

Completion Date: September 30, 2025

Success Criteria: All met
- I generate valid C code.
- My output compiles with a standard C compiler (gcc).
- I match my own semantics.
- I produce working binaries.

## Phase 6 - Standard Library (Minimal - In Progress)

Goal: Provide common functionality.

Deliverables:
- [x] String operations (concat, split, trim, length, format, etc.)
- [x] I/O functions (print, println, file_read, file_write, file_exists)
- [x] Math functions (abs, sqrt, pow, sin, cos, floor, ceil, round, etc.)
- [x] Data structures (arrays with bounds checking, dynamic array_push, HashMap)
- [x] Documentation (STDLIB.md)
- [x] Shadow tests for built-in functions

Current Status: Comprehensive stdlib with 40+ built-in functions.

## Phase 7 - Command-Line Tools Complete

Goal: User-friendly compiler and interpreter interfaces.

Deliverables:
- [x] bin/nanoc compiler command (src/main.c - ~190 lines)
- [x] bin/nano interpreter command (src/interpreter_main.c - ~180 lines)
- [x] Command-line options (-o, --verbose, --keep-c, --call)
- [x] Help system (--help)
- [x] Version information (--version)
- [x] Error formatting with line numbers
- [x] Makefile for building both tools
- [x] Documentation

Completion Date: September 30, 2025

Success Criteria: All met
- I am easy to use.
- I provide clear error messages.
- I have good help text.
- I follow Unix conventions.
- I support both compilation and interpretation.

## Phase 8 - Self-Hosting COMPLETE

Completion Date: January 2026

Goal: I compile myself.

Documentation: See [planning/](../planning/) for my implementation design notes.

Required Features (6 essential) - ALL COMPLETE:
1. [x] Structs - I use these to represent tokens, AST nodes, and symbols (November 2025).
2. [x] Enums - I use these for token types and AST node types (November 2025).
3. [x] Dynamic Lists - I use these to store collections of tokens and nodes (November 2025).
4. [x] File I/O - I read source files and write C output (November 2025).
5. [x] Advanced String Operations - I use these for character access, parsing, and formatting (November 2025).
6. [x] System Execution - I invoke gcc on my generated code (November 2025).

Bootstrap Implementation:
- [x] I implemented my lexer in myself (December 2025).
- [x] I implemented my parser in myself (December 2025).
- [x] I implemented my type checker in myself (December 2025).
- [x] I implemented my transpiler in myself (December 2025).
- [x] My 3-Stage Bootstrap works perfectly (January 2026):
  - Stage 0: C-based nanoc_c compiles Stage 1.
  - Stage 1: My self-hosted components (parser, typecheck, transpiler).
  - Stage 2: Stage 1 recompiles itself.
  - Stage 3: Verification (Stage 1 output matches Stage 2 output).
- [x] I optimized my performance to be within 2-3x of C.
- [x] My documentation is complete.
- [x] My full test suite is passing (221 tests).

Success Criteria: ALL MET
- [x] I compile myself.
- [x] My bootstrapping process works reliably (make bootstrap).
- [x] My output binaries are functionally equivalent (verified via Stage 3).
- [x] My performance is acceptable (native C performance via transpilation).
- [x] All my tests pass (shadow tests + examples + 221 test files).
- [x] My documentation is complete (121+ docs).

## Phase 10 - NanoISA Virtual Machine COMPLETE

Completion Date: February 2026

Goal: I have a custom virtual machine backend with process-isolated FFI.

Deliverables - ALL COMPLETE:
- [x] NanoISA Instruction Set - 178 opcodes, a stack machine with a RISC/CISC hybrid design.
- [x] .nvm Binary Format - I include sections for code, strings, functions, types, imports, debug info, and module refs.
- [x] Assembler & Disassembler - I have a two-pass text assembler and a disassembler with label reconstruction.
- [x] NanoVM Interpreter - I have a switch-dispatch execution engine with a trap model (~1,844 lines).
- [x] Reference-Counted GC - I use OP_GC_RETAIN/OP_GC_RELEASE with scope lifetime tracked implicitly by the call stack.
- [x] Compiler Backend (nano_virt) - I have a three-pass AST-to-bytecode codegen (~3,083 lines).
- [x] Co-Process FFI (nano_cop) - I isolate external calls in a separate process via a binary RPC protocol.
- [x] VM Daemon (nano_vmd) - I can run as a persistent process to reduce startup latency.
- [x] Native Binary Generation - I embed .nvm and my VM runtime into standalone executables.
- [x] Cross-Module Linking - I use OP_CALL_MODULE with per-frame module tracking.
- [x] Closure Support - I use OP_CLOSURE_NEW with upvalue capture and invoke closures via OP_CALL_INDIRECT.
- [x] Comprehensive Test Suite - I have 470 ISA tests, 150 VM tests, and 62 codegen tests.

Architecture: My trap model separates my pure-compute core (83+ opcodes) from I/O operations, which allows for future FPGA acceleration. I have documented this in docs/NANOISA.md.

Total: I consist of approximately 11,000 lines of C across my ISA, VM, compiler, and co-process components.

## Phase 11 - Formal Verification COMPLETE

Completion Date: February 2026

Goal: I have a mechanized metatheory for my NanoCore in the Rocq Prover (Coq), achieved without axioms.

Deliverables - ALL COMPLETE:
- [x] Type Soundness (Preservation) - I have proved that well-typed expressions evaluate to well-typed values.
- [x] Progress - I have proved that well-typed closed expressions are values or can take a step.
- [x] Determinism - I have proved that evaluation is a partial function.
- [x] Semantic Equivalence - I have proved that my big-step and small-step semantics agree.
- [x] Computable Evaluator - I have a fuel-based reference interpreter with a soundness proof.
- [x] OCaml Extraction - I can extract my reference interpreter for testing against my C implementation.

Statistics: I have approximately 6,170 lines of Coq, 193 theorems/lemmas, 0 axioms, and 0 Admitted proofs.

Verified Language Features: I have verified integers, booleans, strings, arrays, records, variants with pattern matching, closures, recursive functions (fix), mutable variables, while loops, and sequential composition.

I have included more details in formal/README.md.

## Phase 9 - Ecosystem & Polish (Current - v0.3.0 target)

Goal: I am polishing myself for a 1.0 release and building my ecosystem.

Status: In Progress

High Priority:
- [x] I have completed my STDLIB.md documentation. Every builtin in `src/builtins_registry.c` now has an entry, and `tests/check_stdlib_docs.sh` (wired into `make test-quick` and `make test`) fails the build if the two ever drift apart again.
- [x] I have added code coverage metrics (gcov/lcov integration).
- [x] I have created ERROR_MESSAGES.md with examples.
- [x] I have documented my memory management model in MEMORY_MANAGEMENT.md.
- [ ] I will expand my FFI safety documentation.
- [x] I have created GENERICS_DEEP_DIVE.md.
- [x] I have added NAMESPACE_USAGE.md.
- [x] I have split eval.c into modules to manage its size (January 2026).
- [x] I have added performance benchmarks to my CI.
- [x] I have integrated fuzzing (AFL++/libFuzzer) (January 2026).

Medium Priority:
- [x] I have a VS Code extension with DAP debug support (editors/vscode/).
- [x] I have added a --profile flag and --profile-output for structured benchmark JSON.
- [x] I have created LEARNING_PATH.md for my examples (docs/LEARNING_PATH.md).
- [x] I have documented my error handling philosophy (docs/ERROR_HANDLING.md).
- [ ] I will add build modes (--debug / --release).
- [x] I have planned my Unicode support (docs/UNICODE.md).
- [x] I have expanded my negative test coverage from 20 to 36 tests (January 2026).

Low Priority:
- [x] I have established an RFC process for my evolution (January 2026).
- [x] I have a package manager (scripts/nano-pkg.sh, packages.json, docs/PACKAGE_MANAGER.md).
- [ ] I will document my concurrency model.
- [ ] I will provide a formal grammar specification.

Target Completion: Q1 2026

## Completed Language Features

### Core Data Types
- [x] Arrays - I have dynamic arrays with bounds checking (November 2025).
- [x] Structs - I have user-defined composite types (November 2025).
- [x] Enums - I have enumerated types with named constants (November 2025).
- [x] Unions - I have tagged unions and sum types with pattern matching (December 2025).
- [x] Generics - I have monomorphized generic types (December 2025).
- [x] Tuples - I have heterogeneous tuples (December 2025).
- [x] First-Class Functions - I treat functions as values (December 2025).
- [x] Affine Types MVP - I parse `resource struct` and perform basic
      identifier-state checks in the C seed (December 2025).
      Path-sensitive, dual-frontend, One-IR ownership is a 5.1 release gate.

## Future Enhancements

I may add these features after I am fully self-hosting:

### Language Features
- [x] Dynamic arrays/slices — `array_push`, bounds checking
- [x] Generics/templates — monomorphized generic types
- [x] Pattern matching — `match` statement with enum/union dispatch
- [x] Modules/imports — native and FFI modules via module.json
- [x] Error handling (Result type) — `Result<T, E>` in stdlib
- [x] Algebraic data types — tagged unions with `union` keyword
- [x] Tuples — heterogeneous tuples
- [x] Parallel independence blocks — `par { }` annotation
- [ ] WASM as a NanoISA translator (5.1). Direct AST `--target wasm` was retired.
- [ ] Explicit type conversions (`float_to_int`, `int_to_float`) in compiled mode
- [ ] Arrays of structs in compiled mode

### Tooling
- [ ] REPL (Read-Eval-Print Loop)
- [x] Language server (LSP) — `bin/nanolang-lsp` (hover, definition, completion, diagnostics)
- [x] Debugger — DAP server `bin/nanolang-dap` (breakpoints, step, inspect via VS Code)
- [ ] Package manager
- [ ] Build system
- [ ] Documentation generator

### Optimizations
- [ ] Tail call optimization
- [ ] Constant folding
- [ ] Dead code elimination
- [ ] Inlining
- [x] NanoISA VM backend (alternative to C) - Complete (February 2026)

### Ecosystem
- [x] VS Code extension (editors/vscode/ — syntax highlighting, LSP, DAP debug)
- [ ] Vim plugin
- [ ] Emacs mode
- [ ] Online playground
- [ ] Tutorial website
- [ ] Community forum

## Timeline Actual vs Estimated

| Phase | Original Estimate | Actual Time | Status |
|-------|------------------|-------------|---------|
| Phase 0: Specification | - | 1 day | Complete |
| Phase 1: Lexer | 2-3 weeks | 1 day | Complete |
| Phase 2: Parser | 3-4 weeks | 1 day | Complete |
| Phase 3: Type Checker | 3-4 weeks | 1 day | Complete |
| Phase 4: Shadow-Test Runner | 2-3 weeks | 1 day | Complete |
| Phase 5: C Transpiler | 4-5 weeks | 1 day | Complete |
| Phase 6: Standard Library | 3-4 weeks | - | Minimal |
| Phase 7: CLI Tools | 2 weeks | 1 day | Complete |
| Phase 8: Self-Hosting | 8-12 weeks | 3 months | Complete (January 2026) |
| Phase 10: NanoISA VM | - | 1 month | Complete (February 2026) |
| Phase 11: Formal Verification | - | 1 month | Complete (February 2026) |

Total Actual Time (Phases 0-7): 2 days (September 29-30, 2025)

Efficiency: I developed much faster than estimated due to focused effort and AI assistance.

## Milestones

### Milestone 1: First Compilation (Phase 1-5) ACHIEVED
Completion Date: September 30, 2025
- [x] I can compile simple programs.
- [x] I generate working C code.
- [x] My shadow tests execute.
- [x] All 15 of my initial examples are working.

### Milestone 2: Usable Compiler (Phase 6-7) MOSTLY ACHIEVED
Completion Date: September 30, 2025
- [ ] My standard library is minimal.
- [x] I have polished command-line tools (compiler and interpreter).
- [x] My documentation is complete.
- [x] I am ready for simple projects.

### Milestone 3: Self-Hosting (Phase 8)
Target: I compile myself.
- I am rewritten in myself.
- My bootstrap process is working.
- My full test suite is passing.

## How to Contribute

I have included details in CONTRIBUTING.md.

Current Focus: Implementation planning.

Most Needed:
1. Feedback on my specification.
2. Additional example programs.
3. Test cases.
4. Implementation volunteers.

## Success Metrics

### Technical
- All my example programs compile and run.
- My shadow tests catch bugs.
- My generated C code is readable.
- I compile quickly.
- I compile myself.

### Community
- I provide clear documentation.
- I have active contributors.
- I have a growing example library.
- I receive positive feedback.

### Adoption
- Real projects use me.
- LLMs can generate correct code for me.
- I have teaching material available.
- I have community resources.

## Risks and Mitigations

### Risk: Specification Changes
I mitigate this by seeking community review before I begin implementation.

### Risk: Implementation Complexity
I mitigate this through incremental development and extensive testing.

### Risk: Performance Issues
I mitigate this because my C transpilation provides a good baseline for performance.

### Risk: Limited Contributors
I mitigate this by keeping my codebase simple and well-documented.

### Risk: LLM Generation Quality
I mitigate this by iterating on my language design based on my testing with LLMs.

## Communication

### Updates
- My commit messages.
- My release notes.
- My GitHub issues and pull requests.

### Discussion
- My GitHub Discussions (when available).
- My issue tracker for bugs and features.

### Documentation
- I keep my docs in sync with my code.
- I update my examples regularly.
- I maintain my changelog.

## Versioning

I follow semantic versioning (semver):

- 0.x.y: Pre-1.0 development.
- 1.0.0: First stable release (after I compile myself).
- 1.x.0: New features (backwards compatible).
- x.0.0: Breaking changes.

## Release Strategy

### Pre-1.0 Releases
- 0.1.0: My lexer is complete.
- 0.2.0: My parser is complete.
- 0.3.0: My type checker is complete.
- 0.4.0: My shadow-test runner is complete.
- 0.5.0: My C transpiler is complete.
- 0.6.0: My standard library is complete.
- 0.7.0: My CLI tool is complete.
- 0.9.0: My self-hosting beta.

### 1.0 Release Criteria
- I compile myself.
- All my examples compile.
- My documentation is complete.
- My test suite passes.
- My performance is acceptable.
- Breaking changes are unlikely.

## Long-Term Vision

I aim to be:

1. A reference implementation for LLM-friendly language design.
2. A formally verified language with mechanized proofs of type soundness and semantic correctness.
3. A sandboxed execution platform via my NanoISA VM with process-isolated FFI.
4. A teaching tool for programming language concepts.
5. A practical language for systems programming.
6. A proof of concept for my shadow-test methodology.
7. A community project with active contributors.

---

Last Updated: September 16, 2026
Current Phase: I am completing the One IR contract for `v5.1.0`. The published
`v5.0.0` language/runtime release remains described in `docs/RELEASE_5.0.md`.
Next Public Release: `v5.1.0`, after the full Phase 20, clean-tree and platform
gates pass.
My unchecked NanoISA-only architecture milestones remain in
`docs/NANOISA_ONLY.md`; a release tag does not establish their acceptance.
Next Review: the exact release candidate and its published artifacts.
