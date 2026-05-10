# Proposal: Replace NanoISA with a Strict Subset of WebAssembly

**Status:** Draft / proposed — not approved
**Author:** code-review session, 2026-05-09
**Decision needed by:** Project owner (Jordan)
**Related beads:** nl-3du, nl-d7n, nl-2tg (phase-1 prerequisites)

## TL;DR

NanoISA today is a hand-written 94-opcode stack VM with its own bytecode format, FFI co-process protocol, daemon, verifier, and disassembler — about 15.7k lines of C across `src/nanoisa/`, `src/nanovm/`, `src/nanovirt/`, and `src/nanocore_*.c`. It exists to serve two goals: **isolation** (FFI in a separate process, deterministic execution) and **provability** (small enough that NanoCore semantics fit in Coq with no axioms).

WebAssembly already provides both: it has a sandboxed execution model with a documented trap semantics, multiple production runtimes (`wasmtime`, `wasm3`, V8), and multiple mechanized formalizations (WasmCert-Coq, WasmCert-Isabelle). The existing `nanoc --target wasm` backend is the right delivery mechanism — it just needs to be completed.

This proposal: **make Wasm the bytecode format**, delete NanoISA, retarget NanoCore proofs at WasmCert-Coq, and ship a runtime that delegates to `wasmtime` (production) or `wasm3` (embedded). Phased so we never burn boats.

## Motivation

Three concrete problems with the current architecture:

1. **The proof story doesn't connect to the compiler.** `formal/*.v` proves a small λ-calculus is sound and deterministic. The C/LLVM/WASM backends emit code that has no formal relationship to those proofs. Soundness on the source side is a separate artifact from "what `nanoc` actually emits."
2. **NanoISA is parallel, not synergistic, with the LLVM/Wasm/RISC-V backends.** `--target wasm`, `--target llvm`, `--target riscv` already exist and produce real artifacts. NanoISA is a fourth target with its own bytecode, runtime, and isolation protocol — and right now it carries almost all the maintenance burden of the other three combined (15.7k lines vs. ~2k lines for the wasm backend).
3. **Recent Backend Matrix fixes (commit edf4ceb) had to paper over WASM/LLVM gaps with stubs** because the wasm backend was incomplete and the test harness only checked "didn't crash." See nl-3du, nl-d7n, nl-2tg. The right response to those gaps is to invest in the wasm backend, not to keep NanoISA as a parallel runtime.

## Current state

```
src/nanoisa/        ~955 lines    Custom 94-opcode ISA (94 of 179 slots used)
src/nanovm/        ~5,500 lines   Stack VM, GC, FFI, daemon, co-process protocol
src/nanovirt/      ~3,973 lines   Codegen from AST to NanoISA bytecode
src/nanocore_*.c   ~1,139 lines   "NanoCore subset" exporter + extraction support
src/wasm_backend.c   ~948 lines   Wasm backend (incomplete — see nl-3du)
─────────────────────────────────
Total VM stack    ~12,517 lines   (excluding wasm backend which we keep)
```

The `bin/` outputs of all this work: `nano_virt`, `nano_vm`, `nano_cop`, `nano_vmd`. Plus the `.nvm` bytecode format, its source-map sidecar, and the `--strip-debug`, `--isolate-ffi`, daemon, etc. CLI surface.

The existing `--target wasm` is incomplete: per nl-3du, calls to `println`, `print`, string builtins, and the AST node kinds for strings/structs/unions/match/arrays are stubbed to `i64.const 0`. The cross-backend test driver (`tests/cross-backend/run-all.sh`) accepts "wasm3 ran without crash" as a pass, so this is invisible in CI (nl-2tg).

## Proposal

Delete NanoISA. Make Wasm the bytecode. Use `wasmtime` (or `wasm3` for embedded contexts) as the runtime.

### Wasm subset to target

**Use:**
- Single, non-shared linear memory
- Numeric types: `i32`, `i64`, `f32`, `f64`
- Structured control flow: `block`, `loop`, `if`, `br`, `br_if`, `br_table`, `return`, `unreachable`
- Functions, locals, globals with typed signatures
- `call`, `call_indirect`, tables (for first-class functions)
- Imports / exports (the FFI boundary)
- Source maps (already implemented in `src/wasm_backend.c`)

**Exclude (initially):**
- SIMD / `v128` (already half-supported, not on the critical path)
- Threads / atomics (out of scope for proofs)
- The Wasm GC proposal (not in `wasm3`; fresh in `wasmtime` — keep ARC in linear memory)
- The exception-handling proposal (algebraic effects already lower to plain control flow)
- Multi-memory, memory64, tail calls (add later if needed)
- WASI Preview 2 (still churning — use Preview 1 for now)

### Runtime delivery

- **Production sandboxed execution:** `wasmtime` — JIT, AOT, mature, formally specified
- **Embedded / minimal footprint:** `wasm3` — interpreter, ~64 KB binary, no JIT
- **Browser / playground:** existing CodeMirror playground already uses Wasm
- **Direct native:** existing C backend stays as the no-runtime path

`nano_vm hello.wasm` becomes a thin wrapper that invokes one of the above. `.nvm` becomes either an alias for `.wasm` or is deprecated outright.

## Phased migration

Each phase is independently shippable. None of them require deleting code from prior phases, so we can pause at any boundary.

### Phase 1 — Make `--target wasm` actually work
**Prerequisite for everything below. Doesn't commit to the larger plan.**

- Fix nl-3du: implement real codegen for strings, structs, unions, match, arrays in `src/wasm_backend.c`. Use linear memory + ARC (mirror the C backend's representation).
- Fix nl-d7n: either rename codegen to use existing `nl_*` runtime symbols, or add `nano_*` thunks in `src/runtime/`.
- Fix nl-2tg: cross-backend driver must compile, run, and assert output for wasm targets, with explicit `xfail` markers for known-unsupported features.
- Acceptance: every program in `tests/cross-backend/*.nano` produces matching `.expected` output under `wasmtime` and (where relevant) `wasm3`.

### Phase 2 — Make `wasmtime` the official runtime
**Reversible. NanoISA still runs in parallel.**

- `nano_vm hello.wasm` becomes a thin wrapper around `wasmtime run --invoke main`.
- `nanoc --emit-nvm` becomes an alias for `--target wasm` (with a deprecation warning).
- `--isolate-ffi` becomes "host imports run in a subprocess sandbox" — `wasmtime` already supports this via component isolation.
- Daemon mode (`nano_vmd`) becomes `wasmtime serve` or a precompiled-module cache.
- `examples/playground/` consolidates: only Wasm artifacts, no `.nvm`.

### Phase 3 — Retarget the proofs
**The biggest research-grade lift. Independent of Phase 4 — repo can ship without this.**

- Add `formal/Codegen.v`. Prove `compile : SourceTerm → WasmTerm` preserves semantics, using **WasmCert-Coq** as the target. This is a lockstep simulation proof.
- Existing `formal/Soundness.v`, `Progress.v`, `Determinism.v` remain valuable — they prove the source language is well-behaved. `Codegen.v` extends the chain to "what the compiler emits."
- `formal/Equivalence.v` keeps its current role (big-step ↔ small-step on the source). Resolve the existing `Admitted` for the tuple sub-case along the way.
- Update the README to describe the proof chain accurately.

### Phase 4 — Delete
**Only after Phases 1–2 are stable. Phase 3 is not a prerequisite for deletion.**

- Delete `src/nanoisa/` (~955 lines).
- Delete `src/nanovm/` (~5,500 lines).
- Delete `src/nanovirt/` (~3,973 lines).
- Delete `src/nanocore_subset.c`, `nanocore_export.c` (~1,139 lines) — replaced by direct codegen-to-Wasm in `src/wasm_backend.c`.
- Delete `bin/nano_virt`, `nano_vm`, `nano_cop`, `nano_vmd` from build targets.
- Drop the `.nvm` format entirely.
- Net deletion: ~11–13k lines, depending on how much support code goes with it.

## What we keep

- **`src/wasm_backend.c`** — promoted from "one of N backends" to the canonical bytecode emitter.
- **`src/c_backend.c`** — stays as the production native-binary path. `nanoc -o hello` still produces a self-contained ELF/Mach-O.
- **`src/llvm_backend.c`** — stays. Useful for users who want LLVM optimization passes or specific targets.
- **`src/riscv_backend.c`, `src/ptx_backend.c`** — stay.
- **`src/runtime/`** — stays. The `nl_*` runtime is shared by C and Wasm targets.
- **All non-VM compiler infrastructure** — lexer, parser, typechecker, FFI module system, optimizer passes, LSP, DAP, doc generator, sign/verify, source maps.
- **`formal/Soundness.v`, `Progress.v`, `Determinism.v`** — unchanged. They prove the source language. `formal/Equivalence.v` keeps its current role (with the `Admitted` cleanup as a follow-on).

## What we lose / have to rebuild

| Today | After Phase 4 |
|---|---|
| `nano_cop` co-process FFI | Wasm imports + host subprocess sandbox (wasmtime config) |
| `nano_vmd` daemon | `wasmtime` precompiled-module cache |
| `.nvm` source-map debug info | `.wasm.map` (already implemented) |
| Custom verifier (`nanoisa/verifier.c`) | Wasm validation in `wasmtime` / `wasm-validate` |
| Custom disassembler (`nanoisa/disassembler.c`) | `wasm-tools dump` / `wabt`'s `wasm2wat` |
| `--strip-debug` (NanoISA-specific) | `wasm-tools strip` |
| Hand-rolled GC integration with stack VM | ARC in linear memory (mirrors C backend) |

The **FFI ergonomics regression** is real. Today you can call arbitrary C from `nano_cop` over a socket protocol. After this proposal, the host has to import each function explicitly, and arbitrary-C requires a host-side bridge (`wit-bindgen` or similar). This is a feature, not a bug, from a sandboxing perspective — but it is more plumbing for users who currently use FFI casually.

## Costs and risks

| Risk | Likelihood | Mitigation |
|---|---|---|
| Phase 1 reveals deeper wasm-backend rot | Medium | Phase 1 is bounded by the existing test programs. If it requires more than ~2 weeks, reassess scope. |
| Phase 3 (proofs) blows out timeline | High | Phase 3 is explicitly independent. The repo can ship Phases 1–2–4 without retargeted proofs and update the README to describe the (narrower) proof story honestly. |
| `wasmtime` dependency adds friction for users | Low | `wasm3` is a single ~64KB embedded interpreter. We can ship a vendored copy. |
| Performance regression vs. current `nano_vm` | Low | `wasmtime` AOT is faster than any hand-written stack VM. `wasm3` interpreter is roughly comparable to the current VM. |
| Wasm GC proposal not stable enough for ARC replacement | High (if we depend on it) | We don't. ARC in linear memory works today (the C backend does it). |
| WASI Preview 2 churn | High (if we depend on it) | We don't. Preview 1 is stable and ubiquitous. |
| FFI users with existing `nano_cop` programs | Low (only one production user — see README) | Document the migration path; provide a compatibility shim if needed during deprecation. |
| Loss of "we wrote our own VM" identity | N/A | The README's voice can be updated. The proof story actually gets *stronger*, not weaker. |

## Alternatives considered

### A. Status quo: keep NanoISA, fix nothing
- Pro: zero work
- Con: WASM/LLVM backends remain broken; proof story stays disconnected from codegen; 12k lines of VM continue to age

### B. Fix wasm/llvm backends, keep NanoISA in parallel
- Pro: smaller scope; preserves all current capabilities
- Con: doubles the long-term maintenance surface; doesn't address the proof-story disconnect; doesn't reduce parallel-target sprawl

### C. Switch NanoISA to track RISC-V
- Pro: real hardware target; existing toolchain
- Con: RISC-V semantics are vastly larger than NanoISA; pinning Coq proofs to a moving target is impractical; loses sandboxing model
- See conversation log for full analysis. Rejected.

### D. Switch NanoISA to LLVM IR
- Pro: real toolchain
- Con: LLVM IR is even larger than RISC-V; bitcode versioning unstable; only Vellvm formalization exists and uses axioms; not designed for sandboxing
- Rejected.

### E. This proposal: NanoISA-as-strict-subset-of-Wasm
- Pro: small enough to formalize (and is); production runtimes exist; sandbox is built in; trap model maps directly; massive code deletion
- Con: proof retargeting is real work; FFI ergonomics regress slightly
- **Recommended.**

## Open questions

1. **WasmCert-Coq vs. WasmCert-Isabelle.** Coq matches existing `formal/` work, but Isabelle's WasmCert may be more complete. Worth a survey before committing Phase 3.
2. **Embedded targets.** Is `wasm3` good enough for what users want from `nano_vm` today? Need to benchmark on the existing test suite.
3. **Daemon mode use cases.** `nano_vmd` is currently used for fast-startup. Is `wasmtime`'s precompiled-module cache a sufficient replacement, or do users need something more daemon-shaped?
4. **`.nvm` format users in the wild.** The README states "production by exactly one person." Confirm: is anyone else relying on `.nvm` such that we need a long deprecation tail, or can we move quickly?
5. **Phase 3 ownership.** Proof retargeting is a research-grade undertaking. Is there capacity, or should the repo ship Phases 1–2–4 with a narrower (but accurate) proof story?

## Decision needed

Approve / reject Phase 1 (the wasm-backend completion work — beads nl-3du, nl-d7n, nl-2tg). Phase 1 is valuable independently, and approving it does *not* commit to the rest of this proposal.

Phases 2–4 are a separate decision once Phase 1 lands and we have empirical signal on:
- How much work the wasm backend actually needs (Phase 1's scope)
- Whether `wasmtime` / `wasm3` performance is acceptable (Phase 2 prerequisite)
- Whether anyone depends on the `.nvm` format such that deprecation needs a long tail (Phase 4 prerequisite)
- Whether there's capacity for proof retargeting (Phase 3 prerequisite)

## Appendix: concrete impact

**Lines deleted (Phase 4):** ~11,500
**Lines added (Phases 1–2):** ~500–1,500 (wasm backend completion + runtime wrapper)
**Net codebase reduction:** ~10,000–11,000 lines (~10% of repo)

**Binaries removed:** `nano_virt`, `nano_vm`, `nano_cop`, `nano_vmd`
**Binaries added:** none (relies on `wasmtime`/`wasm3`)
**External dependencies added:** `wasmtime` (recommended) or `wasm3` (vendorable)

**CLI surface changes:**
- `nanoc --emit-nvm` deprecated → alias for `--target wasm`
- `nano_vm` repurposed as wasmtime/wasm3 wrapper
- `nano_virt`, `nano_cop`, `nano_vmd` removed (all wrapped by `wasmtime` features)
- `--isolate-ffi` reinterpreted as "use wasmtime's subprocess sandbox" (semantics preserved, implementation different)
