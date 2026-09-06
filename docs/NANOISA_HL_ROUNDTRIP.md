# NanoISA high-level reconstruction

I want to know whether a `.nvm` module carries enough facts to become a
high-level language again — structured C, NanoLang, or something else —
without embedding `nano_vm` and a bytecode array in that language.

This is a feasibility record. It is not a claim that I can target any
language. Canonical disassembly (`make test-disasm-roundtrip`) is a
different test: it asks whether bytecode text round-trips, not whether a
reader can recover functions, types, and control flow as source.

Last night's bootstrap idea is the product destination this document
serves: I emit NanoISA once; C11 (and later other targets) consume `.nvm`.
`nano_virt`'s default "native binary" still embeds the VM
(`src/nanovirt/wrapper_gen.c`). That path is not reconstruction, and it
is not the C11 AOT backend. A generated AOT process does not require
`nano_vm`, `nano_cop`, or `nano_vmd` to compute.

## Reconstruction contract

A reconstruction succeeded when a reader of one `.nvm`, without the
original `.nano` or `.nasm` source, can produce:

1. Named functions (a callable per `FUNCTIONS` entry).
2. Parameter and result types from `SIGNATURES` (or the in-memory arity
   and result tag).
3. Structured control flow: `if`/`while`/`return`, not a bytecode
   interpreter loop.
4. A declared host ABI for imports. `CALL_EXTERN` as an RPC into
   `nano_cop` is not that ABI.

A reconstruction failed when the output is a virtual machine in the
target language, a `unsigned char blob[]` plus `nano_vm_run`, or a
daemon client (`nano_vmd`).

I do not delete `src_nano/transpiler.nano` or `src/c_backend.c` because
of this document. Those remain the NanoLang→C path until a NanoISA→C
translator compiles the compiler.

## Inventory (`.nvm` v2 vs reconstruction)

| Need | Where it lives | Status |
| --- | --- | --- |
| Function names | `FUNCTIONS.name_idx` → string/constant pool | Present |
| Arity and results | v2 `SIGNATURES`; in-memory `arity` / `result_tag` / `result_count` | Present |
| Local names | `local_count` and numeric slots (`LOAD_LOCAL` / `STORE_LOCAL`) | Missing. Temps (`l0`, `t3`) are reconstructable from bytecode only |
| Local types | Bytecode is stack-typed at verification time, not stored per slot | Missing as metadata; inferable for a closed i64 subset |
| Structured control | `JMP` / `JMP_FALSE` / `RET` in `CODE` | Reconstructable-from-bytecode-only (CFG). Not recovered as `if`/`while` yet |
| Host ABI | `IMPORTS` | Present as names and tags. Today's runtime is FFI / co-process, not AOT C calls |
| Type layouts | `LAYOUTS` | Present. Unused by the closed i64 subset |
| Source maps | `DEBUG` | Optional. Line/column, not local names |
| Constants | `CONSTANTS` / string pool | Present. `PUSH_I64` immediates also sit in `CODE` |
| Cross-module links | `LINKS` | Present. Not in the closed subset |
| Purity, affine-use, effects | Frontend facts | Missing as NanoISA metadata (Phase 14) |

## Closed fixture (what a reader can recover)

I assemble a two-function module: `add` of two ints, and `main` that
calls `add(40, 2)` and returns the result.

Without the assembly source, a reader of that module can recover:

- Function names `add` and `main`.
- That `add` takes two integers and returns one.
- The arithmetic `40 + 2` as C operators, not as opcode bytes.

A reader cannot recover original local names, because slots are numbers.
The C I emit uses `l0` / `a0` / `t0`.

## Spike: structured C11 (`nvm2c`)

`src/nanoisa/nvm2c.c` translates that closed subset to C11:

- `I64_ADD` / `ADD` → `t = a + b`
- locals → `int64_t lN`
- `CALL` → a C function
- `RET` / `HALT` → `return`
- `int main(void)` wraps the entry function

`make test-nvm2c` compiles that C with `cc -std=c11` and checks the
process exits `42`. The generated source must contain ` + ` and must not
contain `nano_vm`, a bytecode blob, or `CALL_EXTERN`.

This is not a product CLI. `nano_virt -o binary` still uses `wrapper_gen`.
Imports and `PUSH_STR` are refused: they are outside the subset, and
`CALL_EXTERN` is the VM FFI path.

## Finding so far

**Insufficient** for "almost any high-level language." Local names and a
host C ABI are still missing, and control flow is still a stack of jumps
I have not yet structured.

**Sufficient** for a closed integer subset: I can lower verified NanoISA
to native C that does not run a VM. That is the evidence the C backend
can be indirected through NanoISA *without* the multi-process VM model,
for programs that stay in this subset.

A second high-level surface (NanoLang or another HLL) from the same
`.nvm` is still open. If that surface is only an interpreter, the spike
failed.
