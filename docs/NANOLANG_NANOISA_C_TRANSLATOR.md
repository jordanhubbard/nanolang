# My NanoLang NanoISA-to-C translator

I track this implementation under [#1142](https://github.com/jordanhubbard/nanolang/issues/1142).
My release path is source → NanoISA → NanoVM, or source → NanoISA → C emitted
by NanoLang code → host C compiler → standalone native binary. I keep the
[complete release scope](RELEASE_5.1_SCOPE.md).

## My implementation boundary

My translator owns decoding, admission, representation analysis and emission in
NanoLang. File reads, output publication and the resulting program's declared
runtime services may use explicit host interfaces. Calling the C translator,
asking it for a per-program execution plan, or embedding a VM does not implement
this translator. Static C runtime support remains C: it is the target language,
not a substitute for NanoLang analysis and emission.

I retain the C translator as my reference and bootstrap tool until the
replacement translates the complete required corpus and compiler products.
`src_nano/nvm2c/` is the new implementation. I do not change product selection
on the strength of one component's tests.

## My current responsibilities and replacement order

| Order | Current authority | Responsibility I must implement in NanoLang |
| --- | --- | --- |
| 1 | `nvm_format_v2.c`, `nvm_format_v2.h` | Read the 40-byte v2 header and 24-byte directory entries; preserve exact version, feature, flags, range, duplicate, overlap and checksum decisions. |
| 2 | `nvm_v2_cursor.c`, `nvm_v2_constants.c`, `nvm_v2_signatures.c`, `nvm_v2_layouts.c`, `nvm_v2_functions.c`, `nvm_v2_imports.c` | Decode bounded payload tables, explicit byte lengths, exact tags, signatures, function spans, imports and reserved fields. Keep embedded zero bytes and nominal identity. |
| 3 | `nvm_v2_module.c`, `nvm_v2_convert.c`, `modules/nanoisa/nanoisa.c` | Validate the required section set, cross-table indices, entry point, feature/section agreement and declared operand depth. The current bridge to the old in-memory module is an implementation detail I must not mistake for the wire format. |
| 4 | `isa.c`, `verifier.c`, `verifier_types.c`, ownership/reference/layout/service contracts | Decode each instruction exactly; validate branch targets, stack effects/joins, calls, locals, ownership, borrowed places and declared service authority before executable admission. Transporting a section is not admission. |
| 5 | `nvm2c_shape.c`, `nvm2c_callables.c`, `nvm2c_reachability.h` | Derive stable representation constraints, producer tags, aggregate projections, indirect target sets and reachable bodies. Preserve failure poisoning and refuse unresolved/conflicting facts. |
| 6 | `nvm2c.c` and its included emission/runtime fragments | Emit prototypes, C operators, locals, simultaneous edge transfers, direct/indirect calls, tail restarts, strings/byte strings, aggregates, collections, globals, declared host imports and checked lifetime cleanup. Preserve signed endpoints, eager order and once-only effects. |
| 7 | File/Socket/WebSocket/service profile adapters and admission plans | Reconstruct every required admitted profile with its existing grants and lifetime checks. Private diagnostic profiles do not become public merely because I can emit them. |
| 8 | `nvm2c_main.c`, `file_cli.c`, `src_nano/nanoc_v06.nano`, bootstrap/install recipes | Parse the full required CLI, preserve source identity and prior output on refusal, publish checked C, link the declared host closure and install the actual NanoLang-derived translator. |

Paths without a directory prefix in this table are under `src/nanoisa/`.
This inventory maps responsibilities. It does not claim that the C headers'
introductory subset prose enumerates every later admitted operation; actual
admission branches, code, fixtures and release rows remain authoritative.

## My container contract

`src_nano/nvm2c/container.nano` is a pure NanoLang implementation of order 1.
Its `validate` result matches `NvmV2Result` at the container boundary. I compare
with `nvm_v2_validate`, not whole-module deserialization, because these are
different contracts.

I validate header fields in the C reference's order, then every directory entry,
then pairwise overlap, then CRC32 over bytes after the header. An empty section
can name a location in the header; its range must still fit the file. I preserve
that current decision. Unknown ISA versions and missing payload sections are
not accepted for execution by this function: their checks belong to module and
instruction admission, which is not yet implemented here.

My language's `int` is signed 64-bit. I decode a wire `uint64` offset only when it
fits the nonnegative addressable range. A set top bit produces the internal -1
sentinel and the same range/total-size refusal as the reference for any
addressable array. I check subtraction bounds before reading; I do not add
untrusted offset and size. My CRC32 uses nonnegative integer arithmetic and no
foreign checksum/decoder helper.

`tests/nvm2c_container_driver.nano` supplies real file bytes to the same library.
`tests/nvm2c_container_reference.c` exposes the existing C boundary without
reimplementing it. `tests/test_nanolang_translator_container.py` builds wire bytes
independently with Python `struct`/`zlib`, checks named expected decisions and
compares all decisions through native, VM and C AOT execution. My default test
runs three routes; setting `NANOLANG_TRANSLATOR_SOURCE_COMPILER` to an existing
NanoLang compiler driver additionally requires that producer through VM and
sanitized C AOT. The inspected driver is built from `src_nano/nanoc_v06.nano`;
the single-file `nanoisa_emit.nano` tool does not merge imported modules. I run
all five routes for this initial component. A valid envelope
with no payload sections is a container test, not a runnable program.

I retain the initial driver fixture refusals (undeclared CLI operations, then
unsafe expression placement) and the raw single-file emitter's expected imported-
module refusal in my component evidence. I correct the fixture by declaring CLI
operations inside an ordinary `unsafe` statement block and use the full module
driver. I do not change a compiler admission rule to make this fixture pass.

## My cutover gate

I must execute my translator both in NanoVM and as a standalone native product,
then compare the complete pinned required source/NanoISA corpus across both
frontends and both execution routes. That includes malformed modules,
prior-output preservation, nominal/collection/callable identity, evaluation
order, traps and cleanup. I must translate the actual compiler and translator,
retain raw Stage1/Stage2 byte equality, and pass installed/no-C-seed execution
on Linux and Darwin. I inspect the resulting process and link closures for
legacy translator or embedded VM dependence. The existing C translator may be
used as a test oracle during development; it cannot perform production
translation after cutover.

I keep every unfinished payload, verifier, analysis, emitter, service, driver,
bootstrap and platform requirement open until its own acceptance passes.
