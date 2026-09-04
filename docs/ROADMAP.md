# My Roadmap

I keep this document to outline my development journey.

I execute active work from top to bottom. Before implementation begins, I add
the work here as checkable items, including its tests and documentation. I mark
an item complete only after I have verified it. MAC tasks track ownership and
execution; this document records product direction and order.

## Active Execution Queue

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
5.0 operating environment and kernel adapters
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
co-process machinery as NanoLang.

Foundation:
- [x] I selected Forth 2012 Core and every optional word set as the target.
- [x] I established the persistent NanoVM invocation boundary required by an interactive compiler.
- [x] I added and verified `examples/language/forth/pi.fs` under Gforth for 0, 1, 10, and 50 places.
- [x] I pinned Forth 2012, Jackson test-suite v0.15.0,
      forth200x `91f1ed9c756aac27f57e939c270b5f2c84262427`, and Gforth 0.7.3
      in `docs/FORTH_2012.md` and `tests/forth/pins.json`.
- [x] I confirmed Jackson-suite notices permit vendoring with copyright
      retained, refused a wholesale forth200x copy until a per-file inventory
      exists, and refused to vendor GPL Gforth; I have not vendored any of
      them yet.
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
- [ ] I will sign `.nvm` module artifacts themselves, not only their manifests, carrying the signature and its key identifier in a dedicated section of the v2 module format.
- [ ] I will verify a module's signature at load, in `nvm_deserialize`, immediately after the existing CRC32 check and over the same byte range, so every loader inherits it from one place.
- [ ] I will make enforcement a runtime policy rather than a build-time one -- off, warn, or require -- so the same binary can refuse unsigned modules in a locked-down deployment and accept them on a development machine without rebuilding.
- [ ] I will define where verification keys come from and how trust in them is established, since a signature check is only worth the provenance of the key that satisfies it.
- [ ] I will implement atomic service and system upgrades with rollback.
- [ ] I will implement health monitoring, crash-loop control, degraded operation, and recovery policy.
- [ ] I will define administrative capabilities for inspection, update, backup, restore, and shutdown.
- [ ] I will make boot, startup, steady state, upgrade, failure, and shutdown auditable.

Scoping note on module signing: this is deliberately 5.0 work, not 4.0. The
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
