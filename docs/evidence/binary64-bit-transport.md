# My exact binary64 transport evidence

I tested production checkpoint `cdee11e6` from base `53b43377` on Linux AArch64.
My [manifest](binary64-bit-transport.json) retains compiler/tool hashes and log
hashes. My implementation follows the [preimplementation contract](../BINARY64_BIT_TRANSPORT_CONTRACT.md)
for tasks `eadc85b5000f43e3b375ce274ff5ac7a` and
`aafecef0efca4d8f8115df1487e93a6d`.

I passed these gates:

- Fresh two-stage bootstrap, including both stage smoke checks.
- 2,878 NanoISA checks, 274,541 VM checks, 2,422 native checks and 1,365 shape checks.
  These ran before the final source-only bound-name guards; native/VM production
  remained unchanged across that amendment.
- 110 distinct binary64 patterns in both inverse directions, including both
  zeros, finite/subnormal endpoints, infinities and signed quiet/signaling NaN
  payloads. One verified module per small chunk ran through VM, ordinary C,
  LLVM interpretation/O2/native, Wasmtime and import-free Node. Generated C
  passed GCC and Clang ASan/UBSan. Calls, globals, locals and both join branches
  retain exact bits after checked extraction.
- Portable switch-dispatch execution of the same three transport methods
  (5.491 seconds); my normal VM uses computed goto.
- Managed LLVM native ASan and Wasm/Node/Wasmtime retain all 17 edge patterns
  beside allocated strings, with zero live managed objects after each entry.
- Eleven interpreter bit-pattern observations through public function calls.
- Three final source methods: GCC 10.211 seconds; Clang 9.676 seconds. I tested
  C-seed NanoVirt and Stage1/Stage2 canonical output with VM and generated-native
  execution, plus all three legacy C compiler outputs. Clang instrumentation
  applies to generated native C; compiler-selected legacy C toolchains remain
  their configured defaults. Forty type/arity/name refusal routes preserve
  previous outputs. Positive controls cover ordered mutable/immutable globals,
  direct returns and once-only operands in both directions.
- Twelve final combined transport/adjacent methods (41.527 seconds), including
  unchanged float arithmetic, checked numeric casts and source float formatting.
- Six existing facts/canonical-text methods, schema/purity drift checks and a
  primary-opcode collision rejection control.

Nine compiler/runtime/translator hashes were unchanged after the final source
and adjacent gates. I did not modify the frozen product acceptance tree.

I retain failures as history. My first source gate exposed legacy global call
initializers becoming zero; the corrected startup lowering passed fresh source
acceptance. A first unboxed/boxed join fixture met the existing profile refusal;
my admitted transport control uses explicit checked extraction before its
float join. Missing generated `string.h` was corrected before native acceptance.
I also retain setup logs for the switch link-command parser, absent facts helper,
unsupported `nanoisa disasm` spelling (corrected to `dump`), and the intentionally
stopped redundant bootstrap inherited by an initial interpreter-test recipe.
These are not successful gates or unexplained product-failure attributions.

I preserve numeric casts and ordinary arithmetic semantics. My source profile
explicitly refuses calls through same-named bound values, and I do not claim
broader callback-resolution parity. Existing generic boxed/concrete joins and
heap/profile boundaries are unchanged. Both reconstruction targets still refuse
`F64_FROM_BITS` and preserve previous outputs; full executable float reconstruction
and the broader reconstruction parent remain open. I make no Darwin or full
product-platform acceptance claim from this Linux transport gate.

## Current-main integration

I integrated main through PR719 without conflicts at `a61f6585`. My bit transport,
C/source checker, legacy emitter, ordinary native emitter and LLVM lowering files
retain the reviewed production identity. Main contributes separate VM slice
cleanup and managed-array shape/runtime changes. I rebuilt the affected VM and
translators, then all nine transport/source/managed methods passed. The combined
invocation also requested an adjacent shape class without its required
`NMA_LINK_OBJECTS`, so that invocation retains a setup error rather than a green
suite claim. The proper Make target, with the configured native Clang GCC path,
subsequently passed all 18 shape methods in 2.507 seconds. I retain both that
missing-environment log and the unconfigured Clang setup log separately.

My four source compiler/producer hashes remain identical to the fresh bootstrap
qualification; rebuilt execution tools and exact integration logs have their own
manifest entries. A strengthened interpreter check also retained zero floating
exception flags after every bit roundtrip. No failed historical compiler or
carry artifact was replayed, and no full product/reconstruction gate is closed
by this integration.
