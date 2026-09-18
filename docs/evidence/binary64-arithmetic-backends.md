# My scalar arithmetic backend policy evidence

I completed stage 2 of task `task_50090616040044dfa3bba1ab93a9f6d1` on base
`e5d82ef7`, after stage-1 PR730. This task remains open for source stage 3;
full reconstruction and array task3717 remain open. My preimplementation stage
entry is `e2eed1ef`; reviewed production `8dc1340a` routes the four scalar binary
operations through the selected canonical arithmetic NaN policy.

I cover VM typed and all twelve generic FLOAT/FLOAT or mixed INT/FLOAT arms,
native typed/known/boxed numeric paths, and LLVM typed/generic paths shared by
Wasm. Exact tag checks, integer promotion, negation, comparison, transport and
array implementation paths retain their boundaries. DIV tests either signed
zero before arithmetic and returns exact positive zero. LLVM helpers normalize
result bits with integer operations and have no fast/reassoc/contract flags.
Native output embeds the exact guarded stage-1 header. The normative schema
meanings changed without changing opcode values or wire version.

My first build failed because a local helper-emission flag was out of scope in
the footer. I retained `/tmp/nanolang-binary64-arithmetic-backends-build.log`.
Correction `ece589b5` stores the flag in the emitter buffer; the corrected build
passed before execution gates. Generated schema text was produced by that build
and recorded unchanged at `415594a1` after the focused run. This records the
actual generated artifact, not a claim that its text was committed beforehand.

I rebuilt VM dispatch with actual `-ffp-contract=off -fno-fast-math` flags and
retained the command log. The switch-dispatch binary uses the same compile/link
commands with NANO_NO_COMPUTED_GOTO and a separate object/output path. My focused
native-C commands explicitly use these FP flags, O2 strict warnings and
ASan/UBSan with nonrecovering errors. Broader native fixtures use their existing
flags; the volatile helper boundary was separately qualified under contraction
enabled in stage 1. I do not infer driver flags from helper names.

My frozen backend/harness gates passed:

- Four GCC/managed methods: 11.031 seconds.
- Three Clang scalar methods: 10.910 seconds.
- Three switch-dispatch scalar methods: 9.929 seconds.
- Existing VM suite: 274541 checks; checked allocation/recovery controls passed.
- Existing native suite: 2422 checks; shape constraints: 1365 checks.
- Twenty-one adjacent generic arithmetic, tagged arithmetic and numeric-union
  methods: 103.325 seconds.

The scalar cases compare the same module in VM, generated C, unoptimized LLVM,
O2 LLVM, linked LLVM machine code, Wasmtime and import-free Node. They observe
exact result bits for signed qNaNs/sNaNs, differing operand payloads/order, invalid
infinity operations, zero-divisor precedence, signed zeros, ties, subnormals,
overflow, division and a contraction-sensitive chain. Known/boxed generic and
mixed numeric routes are included. Original input bits and unary negation remain
unchanged. The managed case uses actual allocation/cleanup and zero live-object
observers on native LLVM, Wasmtime and Node; its native IR is ASan-instrumented.
LLVM scalar linked code uses the existing harness's normal compilation; I do not
claim the entire scalar LLVM runtime was sanitizer-built.

After every gate above terminated, I strengthened the typed/known/boxed method
with an explicit FLOAT-tag assertion before boxed extraction. Final frozen
harness `6dbe6afa` passed that changed method on GCC/default dispatch in 6.894
seconds and Clang/switch dispatch in 6.606 seconds, including every LLVM/Wasm
route. Production and executable tool hashes remained unchanged throughout.
The native Make relink retained the identical nvm2c hash. No test was edited
while its run was active. The first compile error is the only failed invocation
in this child.

[My manifest](binary64-arithmetic-backends.json) retains frozen/final source and
tool hashes plus raw log hashes. Logs begin `/tmp/nanolang-binary64-arithmetic-`;
per-command scalar fixtures remain under `/tmp/nano-arithmetic-backends-*`.
Managed fixtures follow the existing cleanup harness, with results in the gate
log. Immutable PR720 producer/import hashes still match; those compilers were
not rebuilt or used to claim this new policy.

Stage 3 must still repair and qualify main/optimized interpreter scalar routes,
both legacy C emitters, ordinary f64_* user-name coexistence, and a fresh bootstrap
with actual source producers. Until that stage passes I make no source-wide
policy completion claim or arithmetic reconstruction admission. No historical
PR679 or frozen product artifact was executed or modified.
