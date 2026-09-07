# NanoLang 4.5

I am NanoLang 4.5. This tag is `v4.5.0`. The last public GitHub Release
before it is `v4.0.0`. Product work for 4.1, 4.2, 4.3, 4.4, and 4.5
landed on one branch and ships as one release. I did not cut public
tags for 4.1–4.4. `docs/RELEASE_4.4.md` is the 4.4 product boundary as
it stood on `main` before Phase 19, not a GitHub Release.

I compile to C. I still compile to C. NanoISA v2 and NanoVM v2 from 4.0
remain the verified bytecode path. What I added after that is a Forth
session with Core evidence, message catalogs and six-language guide
drafts, a Nano Service Interface, unforgeable capabilities, a POSIX
service fabric, an SDL editor whose walker runs in a child process, and
a v0 library that maps effects to deployment policy and records traps
for replay.

I am a language. I am also a secure runtime: contracts, capabilities,
supervision, and a journal sit between a program and the host. I host
that runtime on an ordinary POSIX kernel.

**I do not claim a Forth Standard System, GNU Emacs compatibility, a
kernel, a CUDA or CPython wrap, or that the system is internationalized.**

## What I Shipped

### 4.1 — Forth Core evidence, not a Standard System

- A NanoISA Forth session that compiles colon definitions to verified
  bytecode. The pins stay Forth 2012, Jackson v0.15.0, Gforth 0.7.3.
- Jackson Core and Core Ext suites are vendored. `make test-forth-coreext`
  and `make test-forth-jackson` record what passes.
- Passing a suite is evidence for those tests. I still do not claim Core,
  Core Ext, or a Standard System as a banner.
- The precise label is `docs/FORTH_STANDARD_SYSTEM.md`.
- `INCLUDED` / file-access is a recorded gap, not a silent skip. See
  `docs/FORTH_2012.md`.

### 4.2 — Catalogs and guide drafts, not i18n-complete

- UTF-8 message catalogs for `en`, `zh`, `hi`, `es`, `ar`, and `fr`
  (`catalogs/messages/`, `src/catalog.c`). Human stderr looks up the
  process locale. JSON and TOON stay English.
- The user-guide builder emits six editions with `lang` / `dir` /
  `hreflang`. Drafts under `userguide/i18n/` are machine-generated. I do
  not call those drafts reviewed translations.
- Identifiers stay ASCII. Invalid UTF-8 in source fails closed (`CSRC01`).
- `make test-catalog`, `make test-locale-catalog`.

### 4.3 — Nano Service Interface v0

- Stable ids for interfaces, methods, types, errors, capabilities, and
  parameters. Direction, ownership, lifetime, mutability, optionality, and
  streaming fail closed. Typed payloads and versioned errors load in the
  same document. `nl_nsi_compat` answers whether an older client can call
  a newer document.
- `src/nsi_gen.c` emits NanoLang, Forth, Python, Rust, and C++ stubs,
  dispatch, frames, validation, docs, mocks, and NanoISA imports.
- Module manifests carry a portable `nsi` block. `module.json` stays build
  metadata.
- `src/nsi_runtime.c` invokes by method id over in-process, mock, and
  local-process adapters.
- `make test-nsi`, `make test-nsi-gen`, `make test-nsi-runtime`,
  `make test-nsi-manifest`. Authority: `docs/NSI.md`, `docs/NSI_TCB.md`.

### 4.4 — Capabilities, POSIX fabric, isolated editor

- Unforgeable `NlCap` tokens. Fabricating one from an integer or a host
  pointer fails closed. Rights attenuate. Transfer revokes the source.
  Restart bumps generation (`src/nsi_cap.c`, `make test-nsi-cap`).
- Capability-scoped shared memory with a copy fallback
  (`src/nsi_shm.c`, `make test-nsi-shm`).
- A supervisor and `NlHost` POSIX / in-process adapters. Scoped log, fs,
  process, net, audio, graphics, gpu, and python *service slots* — typed
  regions and rights, not device drivers and not language wraps
  (`src/nsi_fabric.c`, `docs/NSI_FABRIC.md`, `make test-nsi-fabric`).
- The SDL editor is a fabric client of `editor.walker` and `editor.freeze`
  stand-ins. The live frame talks to `bin/nano_emacs_worker` over a
  length-prefixed pipe. The frame does not `dlopen` the interpreter. The
  worker does not link SDL. Crash-restart keeps buffers. `C-x C-z`
  freeze-defun runs `nano_vm` as a grandchild (`docs/NANO_EMACS.md`,
  `make test-nano-eval`, `make test-nano-emacs-worker`).
- `modules/sdl_term/mvp.nano` is a compile-only smoke: externs live in
  `unsafe`. `make test-sdl-term-mvp`.

### 4.5 — Effects to policy, trap journal, observability

- A versioned effect map (`schema/nsi/effect_map.v0.json`) connects
  source effect rows, NanoISA traps, NSI methods, and capabilities.
  `State` has no host capability. `Err` maps to `TRAP_ERROR` and does
  not invent an NSI method.
- I emit an inventory from declared effects, generate a reviewable
  deployment manifest, reject uncovered grants, and count unused grants.
  An administrator override deploys anyway and does not widen the source
  declaration (`src/nsi_policy.c`, `make test-nsi-policy`).
- A versioned trap journal records time, entropy, file, network,
  user-input, process, GPU, audio, and service events. Replay returns the
  recorded result and does not call the original service. Mocks and fault
  injection are explicit. Checkpoints are sequence numbers, not heap
  snapshots (`src/nsi_journal.c`, `make test-nsi-journal`).
- SHA-256 hashes a journal. HMAC-SHA256 with a deployment key
  authenticates it. That is not a PKI claim. Redacted export replaces
  payloads with hashes. Repeating-key XOR can seal remaining bytes for a
  local key; that is not AES.
- Traces copy across NanoVM, router, service, and host spans. Provenance
  records source, module, interface, implementation, policy, and output.
  Localized log text does not change audit fields (`src/nsi_obs.c`,
  `make test-nsi-obs`).
- The journal and policy APIs are a tested C library. They are not
  hooked into every `vm.c` trap in this tag. Authority: `docs/NSI_EFFECTS.md`.

### Still true from 4.0

NanoISA v2, the verifier that fails closed on unknown stack effects, the
v2 module format, fuzzed parsing surfaces, cycle collection, and measured
dispatch remain the bytecode contract. See `docs/RELEASE_4.0.md`.

## Evidence

Verified with the commands named above, plus:

- `make test-nano-eval` and `make test-nano-emacs-worker`.
- `otool -L bin/nano_emacs` (Darwin) / `ldd bin/nano_emacs` must not show
  an interpreter dylib. The worker binary must not link SDL.
- `make test` is the release pipeline. CI runs it on x64 and arm64 with
  sanitizers, coverage, documentation, and benchmarks.

I distinguish proved (Coq NanoCore), tested (these suites), and assumed
(anything I did not run).

## What I Have Not Done

- Forth 2012 Standard System, File-Access / `INCLUDED`, and later Forth
  200x proposals.
- Reviewed human translations. Catalogs and drafts exist; JSON/TOON are
  English.
- A kernel. POSIX is the host.
- GNU Emacs. The SDL frame is Emacs-shaped, not compatible.
- CUDA or CPython as wrapped runtimes. Those names are fabric slots.
- Wiring the trap journal into every NanoVM trap in `vm.c`.
- Heap snapshots as checkpoints. Sequence numbers only.
- AES, PKI, or module signing.
- Header-file dependencies in `Makefile.gnu` (GitHub issue #211).
- 4.6 frontends. 5.0: one verified `.nvm` as the only compilation
  contract (`docs/NANOISA_ONLY.md`). LLVM and Wasm return only as
  NanoISA translators.

## Where to read next

- User guide: [Secure Runtime](../userguide/guide/08_secure_runtime.md)
- Developer overview: [docs/presentation/README.md](presentation/README.md)
- LinkedIn draft: [LINKEDIN_4.5.md](LINKEDIN_4.5.md)
