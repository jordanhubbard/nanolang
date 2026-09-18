# My managed-string LLVM/Wasm subset

I extend my closed literal-string lowering with STR_CONCAT, checked STR_SUBSTR and generic ADD
for two strings. I preserve generic numeric ADD, including enum coercion.
My [byte predicates](NANOISA_MANAGED_STRING_PREDICATES.md) also admit
STR_CONTAINS, STR_STARTS_WITH and STR_ENDS_WITH without allocation.
Mixed string/numeric ADD reports a type error. I retain exact byte lengths,
embedded NUL bytes, unsigned byte ordering and truthy empty strings. These
operations do not complete my required full-language LLVM/Wasm scope.

My caller-selected `NVM_PROFILE_CLOSED_MANAGED_STRINGS` applies ordinary
verification and the existing closed module/signature rules before admitting
these operations. The translator first tries its literal profile, then the
managed profile. The original CLOSED_SCALAR and CLOSED_LITERAL_STRINGS API
selectors retain their previous decisions. Tail calls,
other heap values, imports, nominal layouts, ownership/reference and passive
contracts remain outside this subset. My checked substring prerequisite is merged;
my [substring contract](NANOISA_MANAGED_SUBSTRING.md) defines the matched extension.
My [decimal conversion contract](NANOISA_MANAGED_STRING_INT.md) adds managed
string CAST_INT with C-locale decimal prefix parsing and signed saturation.
My [scalar formatting contract](NANOISA_MANAGED_SCALAR_FORMAT.md) admits exact
non-floating CAST_STRING. My [binary64 formatter](NANOISA_MANAGED_BINARY64_FORMAT.md)
now adds exact closed C-locale/default-rounding `%g` output for floating operands;
the temporary floating-module formatting exclusion is removed. My
[binary64 parser](NANOISA_MANAGED_BINARY64_PARSE.md) admits string CAST_FLOAT
with exact decimal/hexadecimal rounding and an explicit cross-host NaN payload
policy. My [legacy companion](NANOISA_LEGACY_BINARY64_PARSE.md) shares that
parser while preserving the evaluator strict-cast/prefix-helper distinction.

## My ownership and errors

I use module-local handles, immutable reference-counted strings and a
reclaiming allocator. LOAD and DUP retain; STORE transfers and releases its
previous destination. Calls transfer arguments into callee locals, and returns
transfer their result before cleaning remaining roots. Consumed operands are
released after their instruction. Branch conditions, comparisons, assertions,
implicit returns and reported errors follow the same ownership rules.

My managed helper checks preserve the first error in a private instance latch.
Each function returns a value/status pair and releases its frame before
propagating an error. I guard unsafe conversions before evaluation. Only the
outer legacy entry traps; no live managed frame traps internally. Prior global
writes persist on error. Instance globals remain owned across repeated entries.

For modules using this runtime, my freestanding exports are:

| Export | Result |
|---|---|
| `nano_try_entry() -> i64` | High 32 bits contain status; low 32 bits contain signed entry-result bits on success, otherwise zero. |
| `nano_entry() -> i32` | Returns the entry result, or traps after frame cleanup. |
| `nano_dispose() -> i32` | Releases global roots and runtime storage. It is terminal and idempotent, including before first entry. |

Status values are OK=0, TYPE=1, ASSERT=2, MEMORY=3, BUSY=4, DISPOSED=5 and
STATE=6. Nested entry/disposal while active is refused without changing the
suspended invocation. I reset the first-error latch only after successful
begin. The native `main` wrapper disposes before returning or trapping; reusable
native and Wasm hosts dispose explicitly. Engine faults, process termination
and machine-stack exhaustion are outside recoverable language errors.

## My target runtime package

Building `nvm2llvm` now requires Clang and LLVM `opt`, in addition to the C
compiler. `NMS_RUNTIME_CLANG` and `NMS_RUNTIME_OPT` select those build tools;
`NMS_NATIVE_CLANG_FLAGS` supplies explicit native toolchain flags when needed.
For example, this Linux ARM64 host's Clang development build needs
`--gcc-install-dir=/usr/lib/gcc/aarch64-linux-gnu/13` to select its complete
installed support files under strict warnings.

I generate native-64 and wasm32 runtime IR from the same portable C source.
The generated manifest records source/generator hashes, compiler version,
flags, target triple/layout and exact IR hashes. I regenerate on each build
request so changed toolchain flags cannot silently reuse stale target IR.
Emitted `.ll` contains its runtime definitions; linking needs no hidden project
object. Native support requires the declared 64-bit pointer/size ABI and normal
malloc/free. Wasm uses my reclaiming allocator without libc/WASI/host imports.
Its memory pages can be reused but cannot shrink until instance disposal by
the host engine. Language disposal releases allocations within those pages.

The CLI accepts `--runtime-target native|wasm32`; existing C APIs default to
native, and `nvm2llvm_emit_target` makes target selection explicit. `nvm2wasm`
selects wasm32 before emission. I reserve `nano_try_entry`, `nano_dispose` and
the `nano_runtime_` prefix against custom entry collisions. Target/profile/name
refusals preserve prior output.

I test the implementation with `make test-llvm-managed-strings`. My contract
and evidence are in [managed concat](NANOISA_MANAGED_CONCAT.md) and
[its checkpoint evidence](evidence/managed-string-concat.md). The broader
managed-runtime task remains open for portable conversions and broader applicable-language coverage.
