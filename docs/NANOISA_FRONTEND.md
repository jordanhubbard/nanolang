# NanoISA frontend contract

I compile more than one language. They do not each get an instruction
set. They emit the same verified `.nvm` v2 module and they pass the
same verifier. That is this contract. 4.6 starts here. NanoLang stays
my native language. The others are bounded probes until their own
goals are met.

I do not claim a Forth Standard System, GNU Emacs, a kernel, or that
the laboratory languages are product compilers.

## What a frontend must produce

A frontend is finished with a program when it has filled this interface
and handed a module to `nl_frontend_accept`:

| Slot | What it is |
| --- | --- |
| Source locations | DEBUG entries and a source path. A module without locations fails closed. |
| Typed functions | Name, arity, locals, result tag and count. Untyped functions fail closed. |
| Layouts | Struct, enum, and union counts the verifier already checks. I do not add a second layout language. |
| Constants | The module string / constant pool. Names live there. |
| Imports | `.import` / `OP_CALL_EXTERN` with a declared signature, or none. |
| Effects | Optional facts: `IO`, `Err`, `State`. Unknown names fail closed. |
| Capabilities | Optional `cap:` identifiers. Anything else fails closed. |
| Diagnostics | Shared diagnostic ids. A frontend-private error vocabulary fails closed. |

Language-specific facts — purity, exhaustiveness, affine-use, effects —
are optional metadata on that interface. They are not new opcodes.

## Shared, not copied

Every frontend uses:

- NanoISA v2 and `nvm_verify` / `nvm_verify_linked`
- NSI contracts
- unforgeable capabilities
- FFI isolation (`nano_cop`)
- DEBUG metadata
- the existing profiler (`NANO_PROFILE` / opcode instrumentation)
- translators of `.nvm` (`nvm2c` today; LLVM and Wasm only as 5.0
  translators)

`nl_frontend_toolchain` records that those surfaces exist. I do not
give a frontend a private copy of any of them.

## Pipeline

Language-specific work stops before the ISA:

1. Desugar (language)
2. Typecheck (language)
3. Emit `.nvm` (shared format)
4. Verify (shared)
5. Optimize NanoISA (shared)

`nl_frontend_phase_is_language_specific` is true only for desugar and
typecheck. An optimizer that runs on a language AST is in the wrong
place.

## Opcodes

An opcode is allowed only if `isa_get_info` knows it. I reject
frontend-specific opcodes unless they are a reusable primitive that
survives review against the other languages. `0xFF` is not a Scheme
opcode. It is a hole.

## Who is started

NanoLang and Nano Forth are implemented. Forth is in the laboratory
because it is a language, an assembler, and a compiler in one design —
a proof of NanoISA, not a product direction back to 1970.

Nano Scheme and Nano ML are implemented as bounded laboratory frontends
(`docs/SCHEME.md`, `make test-scheme`; `docs/ML.md`, `make test-ml`).
Actor, Dataflow, Object, Shell, and Logic have published bounded goals
in `nl_frontend_goal`. They are not implemented. Emitting a module
labeled as one of them fails closed until that frontend's checkboxes in
`docs/ROADMAP.md` Phase 21 are done.

## Cross-frontend programs

A NanoLang module and a Forth module may `CALL_MODULE` the same
library. `nl_frontend_accept_linked` runs the shared verifier on that
link. Later frontends join that table. They do not grow a second
linking story.

## Tests and code

- `src/nanoisa/frontend.h`, `src/nanoisa/frontend.c`
- `make test-frontend-contract`
- `src/scheme/scheme.c`, `docs/SCHEME.md`, `make test-scheme`
- `src/ml/ml.c`, `docs/ML.md`, `make test-ml`
- `docs/NANOISA.md` remains the ISA. This page is the language boundary.

Authority for 5.0 compilation (`.nvm` as the only compiler product) is
`docs/NANOISA_ONLY.md`. This contract does not delete `transpiler.nano`.
