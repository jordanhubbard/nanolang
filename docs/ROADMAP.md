# My Roadmap

I keep this document to outline my development journey.

I execute active work from top to bottom. Before implementation begins, I add
the work here as checkable items, including its tests and documentation. I mark
an item complete only after I have verified it. MAC tasks track ownership and
execution; this document records product direction and order.

## Active Execution Queue

- [ ] I will make the 3.5 benchmark workloads execute successfully on NanoVM,
  then record repeatable profiles for NanoLang execution, allocation, direct and
  indirect calls, FFI, and the current Forth interpreter.

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
| **4.4** | Capability service fabric | I run modules as supervised least-privilege services with typed capabilities, asynchronous IPC, shared-memory bulk transfer, quotas, cancellation, and restart-safe handles. |
| **4.5** | Effects, policy, and replay | I derive deployment policy from effects, record nondeterministic traps, replay executions deterministically, inject failures, and audit service interactions. |
| **4.6** | Multi-language laboratory | I validate NanoISA with bounded Scheme, ML, actor, dataflow, object, shell, and logic frontends, each chosen to test a distinct semantic pressure. |
| **5.0** | Nano operating environment | I package signed services, startup graphs, upgrades, rollback, health monitoring, and kernel adapters into a complete operating environment. Linux, 5BSD, seL4, and other kernels remain interchangeable substrates below the service ABI. |

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
5.0 operating environment and kernel adapters
```

### Phase 12 - NanoISA v2 (3.5 foundation, 4.0 completion)

Goal: I will make NanoISA a regular, compositional, verified instruction set
for NanoLang, Forth, and future frontends. My portable bytecode will remain
readable. Verification and instantiation may translate it into a faster private
form, including measured superinstructions.

Workflow and evidence:
- [x] I wrote the initial Forth-on-NanoISA architecture contract in `docs/superpowers/specs/2026-08-30-ans-forth-nanoisa-design.md`.
- [x] I added persistent `vm_invoke`, latest-function lookup, and incremental function verification on PR #115.
- [x] I measured static NanoVirt opcode and instruction-sequence frequencies across the repository.
- [x] I added a reproducible NanoISA benchmark and profiling harness before changing execution architecture.
- [x] I record opcode, pair, and triple frequencies, retired instructions, branches, call kinds, stack depths, traps, retain/release traffic, allocations, and FFI byte/latency counters.
- [ ] I benchmark NanoLang execution, allocation, direct and indirect calls, FFI, and the current Forth interpreter; compiled Forth, its compiler, and Forth exceptions remain blocked on the Phase 13 runtime.
- [x] I publish benchmark summaries with hardware, OS, compiler, commit, distributions, retired instructions, and normalized costs.
- [x] I require semantic equivalence and full quality gates for every accepted optimization in `docs/NANOISA_OPTIMIZATION_POLICY.md`.

Portable ISA design:
- [x] I defined `spec/nanoisa.yaml` as the source of truth and generate the active opcode metadata plus v2 stack and ownership metadata from it.
- [x] I use a regular local/stack hybrid: indexed locals for named state and an operand stack for expression evaluation.
- [x] I removed fictional architectural registers from the active NanoISA documentation and v2 schema.
- [ ] I will give each portable instruction one comprehensible meaning and keep operand forms symmetric.
- [x] NanoVirt emits explicit signed integer, floating, comparison, boolean, string-concatenation, and array-arithmetic operations; legacy polymorphic scalar operations remain assembler compatibility instructions but no frontend emits them.
- [x] I added signed and unsigned division, remainder, comparison, shifts, carry, borrow, and wide multiplication primitives required by Forth double cells.
- [x] I added coherent indexed `PICK` and `ROLL` operations alongside the basic stack operations.
- [x] I added byte-addressed little-endian memory loads and stores at 8, 16, 32, and 64 bits; unaligned access is explicitly supported.
- [x] I replaced NanoVirt's language-specific struct, tuple, and union lowering with regular layout-driven `AGG_PACK`, `AGG_GET`, `AGG_SET`, and `AGG_TAG` operations.
- [x] I separated direct function references and heap closures into unambiguous value tags and constructors.
- [ ] I will regularize direct, indirect, tail, imported, and linked calls around verified signatures.
- [ ] I will resolve separate-module calls to callable handles during linking rather than carry module/function pairs through dispatch.
- [ ] I will replace special print, assert, and host operations with typed traps where that improves composition.
- [ ] I will move trimming, case conversion, splitting, replacement, formatting, parsing, and collection algorithms from the ISA into runtime libraries.
- [ ] I will retain only primitive string and aggregate operations justified by representation or measured cost.
- [ ] I will add compact constants, short local forms, and compact general operands without making assembly irregular.
- [ ] I will define a clean extended-opcode space without treating an opcode value as an instruction count.

Execution architecture:
- [ ] I will separate compact serialized bytecode, verified instruction IR, and optimized dispatch IR.
- [x] I decode each function once and dispatch predecoded instructions rather than call the generic decoder for every retired instruction.
- [ ] I build instruction-boundary maps and resolve branches plus direct and tail calls during instantiation; layouts, constants, globals, imports, and linked callable handles remain.
- [ ] I will provide computed-goto dispatch where supported and retain a portable switch fallback.
- [x] I moved generated source locations entirely to side tables and removed executable `DEBUG_LINE` instructions from NanoVirt output.
- [x] I made `--strip-debug` remove all generated runtime debug cost by stripping the side table from code that contains no debug opcodes.
- [x] I removed generated `PUSH_VOID; POP`, unreachable `RET; JMP`, and statements after terminating control flow from NanoVirt lowering.
- [x] I added direct tail-call lowering and frame-replacement execution with verifier and runtime signature checks.
- [ ] I will add profile-selected private superinstructions without exposing frontend bookkeeping as portable opcodes.
- [ ] I will initially evaluate local-field load, local increment, compare-branch, union-tag branch, and tail-call fusions.
- [ ] I will accept a fusion only when maintained NanoLang or Forth workloads justify it.

Runtime representation:
- [ ] I will measure and evaluate split payload/tag operand stacks and globals.
- [ ] I will dynamically size globals from serialized declarations instead of embedding 4,096 values in every VM.
- [ ] I will preinstantiate module constants so string literals do not allocate and search the intern table on every execution.
- [ ] I will replace linear transient-string interning or stop interning transient values.
- [ ] I will consistently use stored string lengths and preserve embedded zero bytes.
- [ ] I will add unboxed homogeneous arrays for integer, float, boolean, and byte elements.
- [ ] I will simplify array mutator stack effects and remove the two-result `ARR_POP` convention.
- [ ] I will replace chained hash-map entries with a measured contiguous implementation.
- [ ] I will fix reference ownership for array removal, closure calls, FFI trap arguments, and marshalled arrays.
- [ ] I will choose and document tracing collection or enforceable cycle restrictions for heap graphs.

Verifier and safety:
- [ ] I will implement a control-flow verifier with instruction-boundary validation.
- [ ] I will infer stack height and types through every basic block and require compatible merge states.
- [ ] I will verify call arity, result shape, aggregate counts, local/global/upvalue bounds, type tags, and import signatures.
- [ ] I will verify return shape, maximum operand depth, frame depth, ownership effects, and explicit termination.
- [ ] I will verify linked-module calls and every opcode family rather than selected operands only.
- [ ] I will eliminate integer-overflow and overlap gaps in code-range and section validation.
- [ ] I will rewrite verified operations to unchecked private handlers where the proof permits it.
- [ ] I will add malformed-bytecode tests and fuzz the decoder, loader, verifier, assembler, disassembler, and co-process protocol.

Module format and tools:
- [ ] I will design a NanoISA v2 module header with format version, ISA version, feature bits, total size, and bounded section directory.
- [ ] I will serialize required code, constants, signatures, globals, imports, layouts, links, metadata, and optional debug sections.
- [ ] I will reject duplicate singleton sections, overlaps, partial records, trailing data, and arithmetic overflow.
- [ ] I will make symbolic functions, imports, fields, types, constants, and labels first-class assembler operands.
- [ ] I will make canonical disassembly lossless and byte-length aware.
- [ ] I will validate complete operand consumption and verify every assembled module.
- [ ] I will correct disassembler import annotations, branch operand roles, label construction, and binary-string handling.
- [ ] I will add an exhaustive coverage check tying every opcode to schema, VM behavior, verifier rules, assembly, disassembly, and tests.
- [ ] I will remove or justify opcodes that no frontend emits, including no-op GC scopes and duplicated closure/string operations.
- [ ] I will choose one coherent flattened or separately linked module model and test it end to end.

FFI and traps:
- [ ] I will resolve imports once into typed call descriptors.
- [ ] I will use generated typed stubs or a general ABI layer for mixed integer and floating signatures.
- [ ] I will make argument limits consistent across imports, traps, direct FFI, and co-process calls.
- [ ] I will pass trap stack ranges instead of copying a fixed array of tagged values where measurement supports it.
- [ ] I will make co-process serialization explicitly little-endian and restore a tested large-payload path.
- [ ] I will measure cold startup, warm calls, scalar calls, strings, arrays, crashes, restarts, and batching.
- [ ] I will batch high-frequency host work rather than cross the process boundary for each element.

Documentation and acceptance:
- [x] I replaced stale NanoISA opcode counts and architecture claims in the active documentation; historical changelog entries remain historical.
- [ ] I will document the portable ISA separately from verified and optimized runtime representations.
- [ ] I will provide readable symbolic assembly examples for NanoLang and Forth.
- [ ] I will record why every public instruction belongs in the ISA rather than a runtime library.
- [ ] I will demonstrate performance changes with distributions, not single timing claims.

### Phase 13 - Forth 2012 on NanoISA (4.1)

Goal: I will implement a standards-oriented Forth system whose colon words are
verified NanoISA functions and whose typed library words use the same import and
co-process machinery as NanoLang.

Foundation:
- [x] I selected Forth 2012 Core and every optional word set as the target.
- [x] I established the persistent NanoVM invocation boundary required by an interactive compiler.
- [x] I added and verified `examples/language/forth/pi.fs` under Gforth for 0, 1, 10, and 50 places.
- [ ] I will pin the exact maintained-standard and test-suite revisions.
- [ ] I will confirm licensing before vendoring third-party conformance files.
- [ ] I will add differential runs against a pinned Gforth release.
- [ ] I will document cells, characters, addresses, division, floats, files, terminals, blocks, limits, and ambiguous-condition behavior.

Compiler and runtime:
- [ ] I will create one mutable `NvmModule` and persistent `VmState` per Forth session.
- [ ] I will add VM-owned data, return, floating-point, and control-flow stacks.
- [ ] I will add a byte-addressable virtual Forth address space with validated allocation and file handles.
- [ ] I will implement dictionary headers, execution tokens, name tokens, early binding, immediacy, and word lists.
- [ ] I will implement nested terminal, evaluated-string, included-file, and block input sources with `SOURCE` and `>IN` restoration.
- [ ] I will compile each colon definition privately to NanoISA, verify it, then publish it atomically.
- [ ] I will compile calls to earlier definitions as stable `OP_CALL` references and `RECURSE` to the reserved current definition.
- [ ] I will compile structured control flow with a checked compile-control stack and branch patching.
- [ ] I will implement `CATCH` and `THROW` by restoring Forth stacks, locals, input sources, and NanoVM invocation state.
- [ ] I will implement typed Forth import declarations that lower to `NvmImportEntry` and `OP_CALL_EXTERN`.
- [ ] I will reject FFI signatures the active ABI cannot call correctly instead of guessing.
- [ ] I will restart an isolated FFI co-process after dynamic import-table mutation.
- [ ] I will make `SEE` disassemble the actual compiled NanoISA function and describe imported words.

Standard word sets, in dependency order:
- [ ] I will implement and test Core.
- [ ] I will implement and test Core Extensions.
- [ ] I will implement and test Exception and Exception Extensions.
- [ ] I will implement genuine double-cell arithmetic and test Double Number and its extensions.
- [ ] I will implement and test String and String Extensions.
- [ ] I will implement and test Search Order and Search Order Extensions.
- [ ] I will implement and test File Access and File Access Extensions.
- [ ] I will implement and test Memory Allocation.
- [ ] I will implement recursive, reentrant Locals and Locals Extensions.
- [ ] I will implement and test Facility and Facility Extensions.
- [ ] I will implement and test Programming Tools and Programming Tools Extensions.
- [ ] I will implement an IEEE binary64 floating stack and test Floating Point and its extensions.
- [ ] I will implement UTF-8 Extended Character and Extended Character Extensions.
- [ ] I will implement Block and Block Extensions against an explicitly disposable image.

Tests, examples, and SDL IDE:
- [ ] I will retain the existing 280 cases as regression tests while replacing their nonstandard harness assumptions.
- [ ] I will run pinned committee Core and optional-word-set tests.
- [ ] I will run licensed Forth-2012 tests and record unsupported or manual cases separately.
- [ ] I will add malformed definitions, multiline definitions, early binding, immediate words, execution tokens, overflow, unsigned output, loop boundaries, exceptions, source nesting, and UTF-8 tests.
- [ ] I will make `pi.fs` pass under my Memory-Allocation and Exception implementations with the exact 50-place output.
- [ ] I will update every file in `examples/language/forth/` to standard behavior.
- [ ] I will update `sdl_forth_ide` to launch the NanoISA-backed Forth executable.
- [ ] I will keep the SDL IDE as a PTY client rather than create a second Forth implementation.
- [ ] I will add build, PTY, file-loading, interpreter-liveness, and graphical smoke coverage.
- [ ] I will publish the precise standard-system label only after tests and required documentation support it.

### Phase 14 - NanoISA-Centered Backends (4.0 and later)

Goal: NanoISA is the common typed and verified boundary between all language
frontends and general execution targets. I will implement each frontend once
and each backend once rather than maintain a frontend-by-backend matrix.

Architecture:
- [x] I selected NanoISA as the common IR for NanoLang, Nano Forth, and future frontends.
- [ ] I will preserve frontend-specific purity, affine-use, generic, effect, and exhaustiveness facts as NanoISA metadata.
- [ ] I will define general and restricted compute profiles with verifier-enforced feature sets.
- [ ] I will make C11 the canonical ahead-of-time portability backend from NanoISA.
- [ ] I will implement LLVM IR as a NanoISA translator rather than a NanoLang AST backend.
- [ ] I will implement WebAssembly as a NanoISA translator rather than a NanoLang AST backend.
- [ ] I will evaluate JVM bytecode, SPIR-V, PTX, OpenCL, and Metal as NanoISA translators.
- [ ] I will use a restricted NanoISA compute profile for GPU targets instead of pretending every general instruction maps to a kernel.
- [ ] I will run the same NanoISA module through each applicable target for semantic-equivalence testing.

Direct backend retirement:
- [x] I removed the immature direct NanoLang-to-LLVM backend and its CLI, build, test, and CI surface.
- [x] I removed the immature direct NanoLang-to-WebAssembly backend and its CLI, build, test, signing, publication, and CI surface.
- [x] I removed LLVM and Wasm from the direct cross-backend CI matrix.
- [x] I retain the retired direct backends in Git history rather than carry dormant implementation files in the active tree.
- [ ] I will reintroduce LLVM and Wasm only behind NanoISA translators with full applicable-language coverage.

### Phase 15 - Internationalization and UTF-8 Neutrality (4.2)

Goal: I remain language-neutral in source, runtime text, diagnostics, and logs,
and I publish a useful translated guide as evidence. English remains the
canonical source until the translation workflow proves otherwise.

Language scope:
- [x] I selected the six largest languages by total-speaker metrics for the initial proof: English, Mandarin Chinese, Hindi, Spanish, Modern Standard Arabic, and French.
- [x] I recorded Ethnologue 2026 total-speaker estimates as the dated selection method and treat the ranking as revisable.
- [ ] I will use BCP 47 language tags and explicit fallback chains.
- [ ] I will distinguish language, script, region, locale, encoding, collation, and text direction rather than treating them as one setting.

UTF-8 language and runtime contract:
- [ ] I will require valid UTF-8 at source, diagnostic, log, module metadata, and documentation boundaries unless a value is explicitly binary.
- [ ] I will define identifier normalization and confusable-character policy.
- [ ] I will implement normalization, Unicode case folding, grapheme iteration, display width, and safe character-indexed operations where the public API claims them.
- [ ] I will keep byte-oriented APIs explicit and separate from code-point and grapheme APIs.
- [ ] I will preserve embedded zero bytes only in binary strings and length-aware protocols.
- [ ] I will test malformed, overlong, truncated, combining, supplementary-plane, emoji-sequence, and bidirectional text.
- [ ] I will audit C, NanoVM, NanoISA, FFI, JSON, TOON, source maps, paths, terminals, SDL text, and file APIs for truncation or locale dependence.

Language-neutral diagnostics and logging:
- [ ] I will give every diagnostic and log event a stable message identifier independent of rendered English text.
- [ ] I will separate structured fields from localized prose and keep machine-readable severity, phase, location, and parameters stable.
- [ ] I will add locale selection through a documented CLI option and environment fallback without changing deterministic machine output.
- [ ] I will implement UTF-8 message catalogs with English fallback and missing-key diagnostics.
- [ ] I will support plural rules, number formatting, dates, lists, quoting, and parameter reordering without concatenating translated fragments.
- [ ] I will keep LLM JSON and TOON diagnostics language-neutral by default, with localized rendering as an explicit layer.
- [ ] I will make logs safe for right-to-left text and resistant to bidi control and terminal escape spoofing.
- [ ] I will test catalog completeness, placeholder compatibility, fallback, invalid UTF-8, and deterministic output.

Translated documentation and user guide:
- [ ] I will make the user-guide builder locale-aware with per-language navigation, canonical URLs, `lang`, `dir`, `hreflang`, and fallback metadata.
- [ ] I will define a translation source format that preserves code, links, anchors, front matter, and untranslatable identifiers.
- [ ] I will add translation memory and source-hash tracking so stale translations are visible rather than silently published.
- [ ] I will publish machine-translated Simplified Chinese, Hindi, Spanish, Modern Standard Arabic, and French guides as explicitly machine-generated drafts.
- [ ] I will preserve English code examples and identifiers while translating explanation and interface prose.
- [ ] I will add language switching that keeps the current page when a translation exists.
- [ ] I will test generated links, anchors, search, code blocks, font fallback, mobile layout, and Arabic right-to-left rendering.
- [ ] I will document how contributors report and correct translations through issues and pull requests.
- [ ] I will credit human reviewers and distinguish reviewed translations from machine-generated drafts.

Acceptance:
- [ ] I will compile and run representative NanoLang programs containing all initial scripts through C and NanoVM paths.
- [ ] I will emit and parse localized diagnostics and logs for all six initial languages.
- [ ] I will build and link-check all six guide editions in CI.
- [ ] I will perform visual checks for Simplified Chinese, Devanagari, Latin, and Arabic scripts on desktop and mobile.
- [ ] I will not call the system internationalized while core diagnostics or logs still require English prose for machine interpretation.

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
- [ ] I will define a versioned Nano Service Interface schema with stable interface, method, type, error, and capability identifiers.
- [ ] I will describe parameter direction, ownership, borrowing, transfer, lifetime, mutability, optionality, and streaming in the schema.
- [ ] I will support records, variants, arrays, strings, binary data, resources, callbacks, asynchronous results, and versioned errors.
- [ ] I will define backward- and forward-compatibility rules for interfaces and wire representations.
- [ ] I will reject ambiguous ABI inference; every foreign boundary will have an explicit typed contract.
- [ ] I will generate NanoLang and Nano Forth bindings from the same interface description.
- [ ] I will generate client stubs, server dispatch, serialization, validation, documentation, mocks, and compatibility tests.
- [ ] I will generate NanoISA imports and typed trap descriptors from service contracts.
- [ ] I will preserve implementation language neutrality: C, C++, Python, Rust, NanoLang, NanoVM, and remote services expose the same contract.

Module refactoring:
- [ ] I will extend module manifests with interface version, required capabilities, isolation policy, resource budgets, restart policy, and implementation adapter.
- [ ] I will separate portable interface metadata from platform-specific build metadata.
- [ ] I will inventory every current native and Python-backed module by privilege, state, payload size, latency, and failure behavior.
- [ ] I will migrate pure modules first and prove identical in-process and service-process behavior.
- [ ] I will migrate filesystem, logging, process, networking, audio, graphics, GPU, and Python modules in increasing privilege order.
- [ ] I will keep unsafe implementation details behind generated service boundaries rather than expose host pointers or library objects.
- [ ] I will give each resource handle an interface type, service identity, generation, rights mask, and lifetime state.
- [ ] I will update module discovery and package metadata to resolve interface contracts independently from implementations.
- [ ] I will add contract tests that run one client against every supported implementation of an interface.

Transport-neutral invocation:
- [ ] I will replace symbol-name-centered RPC with stable interface and method identifiers.
- [ ] I will define request, response, error, cancellation, deadline, and stream frames.
- [ ] I will negotiate interface and transport versions before accepting calls.
- [ ] I will support synchronous and asynchronous invocations without changing source-level imports.
- [ ] I will implement bounded queues and explicit backpressure.
- [ ] I will make idempotence and retry safety explicit properties of methods.
- [ ] I will authenticate callers and validate capabilities before dispatch.
- [ ] I will make malformed messages fail closed without corrupting the service or caller.

Acceptance:
- [ ] I will run one unchanged NanoLang client against in-process, local-process, and mock implementations.
- [ ] I will run one unchanged Nano Forth client through the same generated interface.
- [ ] I will demonstrate an implementation replacement without recompiling the client.
- [ ] I will test schema evolution across at least one compatible minor version and one rejected breaking version.
- [ ] I will document the exact trusted computing base for each deployment mode.

### Phase 17 - Capability Runtime and Shared Memory (4.4)

Goal: I will replace ambient authority with explicit, typed, least-privilege
capabilities and move bulk data without weakening isolation.

Capability model:
- [ ] I will define unforgeable capability references that cannot be fabricated from integers or host pointers.
- [ ] I will encode object type, service generation, rights, delegation policy, and revocation state in capability tables.
- [ ] I will support rights attenuation when delegating capabilities.
- [ ] I will require explicit transfer permission before a service can pass a capability onward.
- [ ] I will invalidate stale handles after service restart and prevent generation reuse attacks.
- [ ] I will map NanoLang resource types to capability ownership and consumption rules.
- [ ] I will map Forth handles to validated capability references without exposing raw host addresses.
- [ ] I will audit every capability creation, delegation, use, revocation, and failure.

Shared-memory data plane:
- [ ] I will keep typed IPC as the control plane and use capability-scoped shared regions for bulk data.
- [ ] I will implement read, write, map, seal, transfer, borrow, return, and revoke rights for shared buffers.
- [ ] I will validate offset, length, alignment, lifetime, and direction on every mapping and descriptor.
- [ ] I will support zero-copy or bounded-copy paths for audio frames, graphics surfaces, network packets, files, and GPU buffers.
- [ ] I will make ownership transfer and completion explicit so buffers cannot be reused while a service owns them.
- [ ] I will provide a copying fallback with identical semantics when shared mappings are unavailable.
- [ ] I will benchmark control-message latency, throughput, copies, mappings, and cache effects by payload size.

Resource governance:
- [ ] I will enforce per-service memory, CPU, handle, queue, file, network, and device budgets.
- [ ] I will attach deadlines and cancellation tokens to service requests.
- [ ] I will define behavior for quota exhaustion, cancellation races, partial results, and abandoned clients.
- [ ] I will expose structured resource accounting without requiring localized prose.
- [ ] I will test hostile clients, forged handles, stale generations, oversized messages, queue floods, and service crashes.

### Phase 18 - Portable Service Fabric and Supervision (4.4)

Goal: I will host Nano services above ordinary multitasking kernels without
embedding Linux, BSD, or microkernel assumptions in application interfaces.

Supervisor:
- [ ] I will implement service discovery, startup ordering, dependency health, readiness, and shutdown.
- [ ] I will define restart, retry, fail-request, fail-application, and replacement policies.
- [ ] I will distinguish transient, permanent, protocol, authorization, quota, and implementation failures.
- [ ] I will make retries conditional on declared idempotence and request identity.
- [ ] I will preserve or revoke state explicitly across service upgrade and restart.
- [ ] I will support rolling replacement when interface compatibility permits it.
- [ ] I will propagate deadlines, cancellation, tracing context, and audit identity across service calls.

Portable host adapters:
- [ ] I will define a narrow host abstraction for processes, threads, IPC endpoints, shared memory, clocks, entropy, files, networking, devices, and credentials.
- [ ] I will implement the first complete service-fabric adapter on a mature host kernel selected by measured development cost and security properties.
- [ ] I will keep transport and policy behavior identical across host adapters through conformance tests.
- [ ] I will support local in-process mode for development without weakening production policy declarations.
- [ ] I will support process-isolated mode using the host's strongest practical primitives.
- [ ] I will support remote transport without giving remote services local capability authority.

Service migration milestones:
- [ ] I will migrate logging and diagnostics as the first observable service.
- [ ] I will migrate filesystem access with path-scoped capabilities.
- [ ] I will migrate process execution with executable, argument, environment, and child-control capabilities.
- [ ] I will migrate networking with endpoint-scoped capabilities.
- [ ] I will migrate audio with stream-scoped device and shared-buffer capabilities.
- [ ] I will migrate graphics and window-system access with surface and input capabilities.
- [ ] I will migrate GPU access with device, queue, memory, shader, and synchronization capabilities.
- [ ] I will migrate Python integration into a typed language-service adapter with no direct Python-object leakage.

### Phase 19 - Effects, Deployment Policy, and Deterministic Replay (4.5)

Goal: I will connect declared program effects to deployable least-privilege
policy and make nondeterministic execution recordable, replayable, and auditable.

Effects to policy:
- [ ] I will define the relationship between source effects, module requirements, NanoISA traps, service methods, and capabilities.
- [ ] I will emit a complete effect and capability inventory for each program.
- [ ] I will generate a reviewable deployment manifest from that inventory.
- [ ] I will reject deployments whose granted capabilities do not cover declared effects.
- [ ] I will report unused grants so policy can converge toward least privilege.
- [ ] I will support explicit administrator overrides without silently widening source declarations.

Record and replay:
- [ ] I will define a versioned trap journal containing sequence, capability, method, arguments or hashes, result, timing, service generation, and implementation version.
- [ ] I will record time, entropy, file, network, user-input, process, GPU, audio, and service nondeterminism at the boundary where it enters a NanoVM.
- [ ] I will replay a NanoVM without invoking original services when the journal contains all required events.
- [ ] I will validate replay order, argument identity, capability identity, and result schema.
- [ ] I will support deterministic service mocks and configurable fault injection.
- [ ] I will expose replay checkpoints for debugger reverse navigation where state capture permits it.
- [ ] I will sign and hash journals when they are used as audit evidence.
- [ ] I will define redaction and encryption so replayability does not require publishing sensitive payloads.

Observability and provenance:
- [ ] I will assign trace IDs across NanoVM, router, service, and kernel-adapter boundaries.
- [ ] I will emit structured metrics and traces through an implementation-neutral telemetry interface.
- [ ] I will record source, NanoISA module, interface, service implementation, policy, and output provenance.
- [ ] I will test that localized logs do not alter stable audit fields or replay behavior.

### Phase 20 - Hardened Operating Environment (5.0)

Goal: I will package the language, VM, services, capabilities, policy, and
supervision layers as a complete operating environment. Kernel choice remains
a deployment decision below the stable Nano service ABI.

System image and lifecycle:
- [ ] I will define signed manifests for NanoISA modules, service interfaces, implementations, capabilities, and policy.
- [ ] I will build deterministic system images from a locked dependency and service graph.
- [ ] I will verify signatures, hashes, interface compatibility, and policy before activation.
- [ ] I will implement atomic service and system upgrades with rollback.
- [ ] I will implement health monitoring, crash-loop control, degraded operation, and recovery policy.
- [ ] I will define administrative capabilities for inspection, update, backup, restore, and shutdown.
- [ ] I will make boot, startup, steady state, upgrade, failure, and shutdown auditable.

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

Goal: I will test whether NanoISA is genuinely language-neutral by compiling a
small set of deliberately different languages to the same verified IR. I will
not collect syntax for its own sake. Each frontend must expose a distinct
architectural weakness or prove a distinct capability.

Shared frontend contract:
- [ ] I will define a frontend interface for source locations, typed functions, layouts, constants, imports, effects, capabilities, and diagnostics.
- [ ] I will require every frontend to emit the same versioned NanoISA module format and pass the same verifier.
- [ ] I will give every frontend access to the same service contracts, capability model, FFI isolation, debugger metadata, profiler, and target translators.
- [ ] I will separate language-specific desugaring and type analysis from language-neutral NanoISA optimization.
- [ ] I will preserve language-specific facts such as purity, exhaustiveness, ownership, and effect information as optional metadata.
- [ ] I will define bounded implementation and test goals before starting each frontend.
- [ ] I will reject frontend-specific opcodes unless they represent a reusable primitive that survives review against the other languages.
- [ ] I will run cross-frontend programs against shared NanoISA libraries and service interfaces.

Nano Scheme:
- [ ] I will implement a small Scheme frontend as the first post-Forth language experiment.
- [ ] I will support lexical scope, closures, first-class procedures, recursive data, and interactive evaluation.
- [ ] I will implement proper tail calls and verify constant frame depth under deep recursion.
- [ ] I will evaluate continuations only after ordinary closure and exception semantics are stable.
- [ ] I will use Scheme to stress allocation, callable representation, tail calls, dynamic values, and live code publication.
- [ ] I will run a pinned subset of a recognized Scheme test suite and document intentional exclusions.

Nano ML:
- [ ] I will implement a compact ML-family frontend with static inference, algebraic data types, pattern matching, immutable values, and higher-order functions.
- [ ] I will use ML to test generic instantiation, aggregate layouts, exhaustive matching, closures, and module signatures.
- [ ] I will preserve inferred type and exhaustiveness facts in NanoISA metadata where target-independent optimization can use them.
- [ ] I will run shared aggregate and service-interface programs under both NanoLang and Nano ML.

Nano Actor:
- [ ] I will implement an Erlang, Elixir, and Gleam-inspired actor frontend.
- [ ] I will support isolated actors, typed mailboxes, pattern-matched messages, monitors, links, supervision trees, deadlines, and cancellation.
- [ ] I will first execute actors as isolated NanoVM contexts in one host process.
- [ ] I will then move unchanged actors across service-process boundaries through the Phase 18 transport.
- [ ] I will test crash containment, mailbox ordering, supervision, hot code replacement, and restart-safe capabilities.

Nano Dataflow:
- [ ] I will implement a deterministic dataflow and workflow frontend with typed nodes, streams, backpressure, and explicit effects.
- [ ] I will map graph dependencies to local, service-process, and remote scheduling without changing program semantics.
- [ ] I will use dataflow programs to test shared-memory bulk transfer, provenance, replay, cancellation, retries, and parallel determinism.
- [ ] I will record every external input required to reproduce a completed workflow.

Nano Object:
- [ ] I will implement a small Smalltalk-like object frontend with message dispatch, object identity, mutable graphs, reflection, and live method replacement.
- [ ] I will use it to test dynamic dispatch, inline caches, layout evolution, callable handles, image persistence, and debugger reflection.
- [ ] I will measure specialization and quickening without exposing cache-specific operations in portable NanoISA.

Nano Shell:
- [ ] I will implement a capability-safe orchestration shell using structured values rather than text-only pipelines.
- [ ] I will expose processes, files, networks, services, streams, cancellation, and remote execution only through explicit capabilities.
- [ ] I will preserve typed values across pipelines and make text parsing an explicit adapter.
- [ ] I will use Nano Shell as the administrative language for service graphs only after capability and policy enforcement are complete.

Nano Logic:
- [ ] I will implement a bounded Datalog or logic frontend for declarative authorization, dependency, and policy rules.
- [ ] I will support facts, rules, unification appropriate to the selected subset, queries, and deterministic fixed-point evaluation.
- [ ] I will use it to test choice points or tabling only when those mechanisms are justified by the selected language subset.
- [ ] I will compile deployment and capability policy queries to verified NanoISA or a documented restricted profile.

Frontend matrix and demonstrations:
- [ ] I will maintain a matrix showing how NanoLang, Nano Forth, Nano Scheme, Nano ML, Nano Actor, Nano Dataflow, Nano Object, Nano Shell, and Nano Logic exercise typing, calls, closures, stacks, matching, concurrency, services, replacement, and replay.
- [ ] I will implement one shared service interface consumed from NanoLang, Nano Forth, Nano Scheme, and Nano ML.
- [ ] I will implement one supervised service in Nano Actor and orchestrate it from Nano Shell.
- [ ] I will apply Nano Logic policy to that service without embedding policy semantics in the application.
- [ ] I will run equivalent computation fixtures across applicable frontends and compare their NanoISA behavior and results.
- [ ] I will publish measured compile time, module size, instruction mix, allocation, call behavior, and execution time for each frontend.
- [ ] I will keep NanoLang as my native language and describe the others as bounded architecture probes until their own conformance goals are met.

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
- [x] Reference-Counted GC - I use scope-based auto-release with OP_GC_SCOPE_ENTER and OP_GC_SCOPE_EXIT.
- [x] Compiler Backend (nano_virt) - I have a three-pass AST-to-bytecode codegen (~3,083 lines).
- [x] Co-Process FFI (nano_cop) - I isolate external calls in a separate process via a binary RPC protocol.
- [x] VM Daemon (nano_vmd) - I can run as a persistent process to reduce startup latency.
- [x] Native Binary Generation - I embed .nvm and my VM runtime into standalone executables.
- [x] Cross-Module Linking - I use OP_CALL_MODULE with per-frame module tracking.
- [x] Closure Support - I use OP_CLOSURE_NEW and OP_CLOSURE_CALL with upvalue capture.
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
- [x] Affine Types - I use these for resource management (December 2025).

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
- [x] WASM backend — `--target wasm` emits WebAssembly binary
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

Last Updated: February 20, 2026 (Post-VM + Formal Verification Update)
Current Phase: Phase 9 - Ecosystem & Polish (Phases 10-11 complete in parallel)
Next Major Milestone: v1.0 Release (target: Q3 2026)
Next Review: After Phase 9 completion
