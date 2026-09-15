# My Roadmap

I keep this document to outline my development journey.

I execute active work from top to bottom. Before implementation begins, I add
the work here as checkable items, including its tests and documentation. I mark
an item complete only after I have verified it. MAC tasks track ownership and
execution; this document records product direction and order.

When I discover a defect, an overclaim, or required work while executing this
document, I add a concrete checkbox here in dependency order before I continue.
Chat is not the ledger. A defect I already fixed in the same session still gets
an `[x]` item so it stays in product history.

**Public GitHub Release `v4.5.0` covers 4.1–4.5.**
https://github.com/jordanhubbard/nanolang/releases/tag/v4.5.0
Tagged `ef32c833` on 2026-09-07. Leftover 4.1, sdl_term, Phase 19, docs,
deck, and user guide are in. I do not call the system internationalized.
JSON/TOON and catalog fallback still use English; guide drafts are
machine-generated. I do not claim a Forth Standard System, GNU Emacs, a
kernel, CUDA, or a CPython wrap. **The next public GitHub Release is
5.0.** The release cut ships the 4.6 laboratory frontends and the audited
language/runtime fixes described in `docs/RELEASE_5.0.md`. The user authorized
this cut without waiting for the fleet hold. Unchecked 5.0 architecture and
acceptance milestones below remain follow-up work, including the NanoISA-only
bootstrap in `docs/NANOISA_ONLY.md`; the release number does not complete them.
6.0 stays out of this bar.

## Active Execution Queue

- [x] **Release-gate launcher capture.** The full suite reaches 219 passing
      implementation tests and one compile failure in
      `tests/unit/test_launcher_capture.nano`. I diagnose its saved compile log,
      preserve child-capture and termination assertions, and rerun the gate.
      Two reproductions exposed shell-normalized exit status 143. I use
      `printf; exec sleep` so the observed PID receives the signal directly.
      Eight consecutive native compilations with dependency/root shadows and
      executable runs pass. The complete suite remains a separate release gate.
- [ ] **Launcher termination output.** I drain child pipes again after reaping
      a stopped child. `pm_kill` currently drains only immediately after sending
      SIGTERM, so output produced by a termination handler can be lost before
      descriptor closure. I require a handler-output regression, not a sleep.
- [ ] **5.0 release integration.** I reconcile the audit-contract branch with
      main, preserve original-file diagnostics and immutable native-cache
      behavior, run clean build/tests and documentation gates, then merge and
      publish 5.0. The fleet dispatch hold does not gate this user-authorized
      release. Unfinished roadmap items remain explicitly unfinished.
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
- [ ] **Release-gate VM example coverage.** I repair the compilation and
      dependency-shadow failures reported by `make test-vm-examples` and
      remove eight stale exclusions that now compile to bytecode. I retain
      the default dependency-shadow contract rather than hiding failures.
      MAC `task_7ee12d8737363c126a040fde905a7114`.
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
      - [ ] I carry callback signatures and adapter contracts through the
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
      - [ ] I connect owner-thread callback execution to suspended VM
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
- [ ] **Follow-up — C arrays of records.** I retain the nominal element
      name and use the dynamic-array representation when lowering record
      literals or record-returning calls inside array literals. My current
      C emitter produces `struct[]` without a record name in both cases.
      MAC `task_967a32569524e07e3c97742cf23234e9`.
- [ ] **Follow-up — MAC commands execute once.** My standard-library command
      wrapper captures stdout with one execution, then executes the same
      command again to obtain its status. I replace this with one execution
      and test a counted side effect, output, failure status, and offline use.
      Release tests use an offline fixture so shadows cannot mutate a live
      task ledger. MAC `task_5f807ded474a473ca5776018c32c636f`.

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
      Older transpiler shadows still call `tokenize_string` with one argument;
      its current signature requires source, filename and diagnostics.
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
- [ ] **5.0 audit — native runtime services.** I run a useful native
      NanoLang service in a separate worker through typed NSI calls, with
      scoped capabilities, restart supervision, tracing, and module packaging.
      I expose mailboxes, monitoring, links, supervision, and upgrade behavior
      through native NanoLang runtime APIs; laboratory frontends remain tests.
      Acceptance exercises real operations, worker death, recovery, denied
      authority, bounded queues, and version compatibility end to end.
- [ ] **5.0 audit — FFI authority.** I resolve declared typed imports by
      module identity, remove ambient symbol fallback from the secure path,
      validate manifests, and isolate untrusted native code. I test symbol
      collisions, undeclared imports, unload lifetime, and crash containment.
- [ ] **5.0 audit — measured ergonomics.** I benchmark human authoring and
      LLM generation/repair with representative programs and recorded compiler
      diagnostics. I use correctness, repair rounds, and author feedback to
      improve imports, operator grouping, shadow policy, and standard-library
      consistency; I publish migrations for changed accepted syntax.
- [ ] **5.0 / Phase 20.** NanoISA-only compilation. Verified `.nvm` is
      the only compiler product. Native AOT does not embed `nano_vm`.
      Public GitHub Release `v5.0.0` after 4.6 and this phase close.
      `docs/NANOISA_ONLY.md`.

- [ ] **5.0 audit — enforceable release evidence.** I require successful
      CI and review before merging, make lint and proof gates blocking, pin
      build dependencies and Actions, and publish reproducible artifacts with
      signed checksums, SBOMs, and provenance. I remove destructive release
      synchronization and predictable temporary files, reconcile task evidence,
      and verify clean installation and rollback before the public release.

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
- [x] I recorded the 5.0 One IR rewrite (`docs/NANOISA_ONLY.md`, Phase 20;
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
| **5.0** | One IR: NanoISA-only compilation | I emit verified `.nvm` as the only compiler product. C11 AOT, LLVM, Wasm, and GPU targets are translators of that module. Self-host proof compares `.nvm`. Native binaries do not embed `nano_vm`. Contract: `docs/NANOISA_ONLY.md`. |
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
5.0 One IR (NanoISA-only compilation; docs/NANOISA_ONLY.md)
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

### Phase 14 - NanoISA-Centered Backends (4.0 spike; rewrite is 5.0)

Goal: 4.0 records that NanoISA is the common IR and that a closed integer
subset can become structured C11 without embedding the VM. The ambitious
rewrite — I emit only `.nvm`; C/LLVM/Wasm are translators; bootstrap
compares `.nvm`; `transpiler.nano` leaves the compiler — is **5.0**.
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

I moved these 4.0-or-later rewrite items to Phase 20 (5.0): frontend facts as
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

### Phase 20 - One IR: NanoISA-only compilation (5.0)

Goal: I emit one portable product — a verified `.nvm` v2 module — and I treat
C, LLVM, Wasm, RISC-V, and GPU targets as translators of that module. Native
means structured AOT, not a bytecode blob plus `nano_vm`. Contract:
`docs/NANOISA_ONLY.md`. I do not start this phase by deleting
`transpiler.nano`.

- [x] I recorded the 5.0 compilation contract in `docs/NANOISA_ONLY.md`
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
- [ ] I gate 5.0 on one affine acceptance matrix across both frontends,
      NanoISA, NanoVM, and AOT C
      (`task_28f2fb4b1f3c8a5ce93df628bb569d76`).
- [ ] I take the bounded One-IR slice of `PASSIVE_PARALLELISM_DESIGN.md` into
      5.0: verified purity/independence, deterministic serial semantics, and
      NanoISA eligibility metadata. Scheduler optimization, async I/O, SoA,
      and hardware speedup claims remain outside this task
      (`task_90b123edcc301b464a031c55e4ba1a11`).
- [x] I postponed `ROW_POLYMORPHIC_RECORDS_DESIGN.md`: the C-seed prototype is
      not a language contract without self-hosted parity, a stable ABI,
      dual-frontend conformance, `.nvm` round trips, VM/AOT equivalence, and
      performance evidence (`task_a39aac00600aa77b55ad92ac70a2d1bf`).

Compiler product:
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
a deployment decision below the stable Nano service ABI. This follows 5.0
(NanoISA-only compilation). Signing and kernel adapters are 6.0, not 4.0
and not 5.0.

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
      Path-sensitive, dual-frontend, One-IR ownership is a 5.0 release gate.

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
- [ ] WASM as a NanoISA translator (5.0). Direct AST `--target wasm` was retired.
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

Last Updated: September 7, 2026
Current Phase: 4.6 complete; 5.0 / Phase 20 next (`docs/NANOISA_ONLY.md`).
The next public GitHub Release is 5.0, covering 4.6 and 5.0.
Next Major Milestone: 5.0 NanoISA-only compilation, then public tag `v5.0.0`.
Next Review: after `make test-frontend-matrix` and `docs/NANOISA_ONLY.md`.
