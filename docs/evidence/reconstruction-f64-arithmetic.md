# My typed binary64 arithmetic reconstruction evidence

I qualify the four typed operations in
[my contract](../NANOISA_RECONSTRUCTION_F64_ARITHMETIC.md), task
`task_509569917fdd4ce390a67e5d5b1e55f1`. Full reconstruction parent4bd and
release acceptance remain open. I do not add generic FLOAT arithmetic, casts,
heap profiles or a new source-shadow recovery claim.

## My source and tools

I began from main90bd4787, recorded contract2f4de64c, and independently reviewed
productionde150525. My exact FLOAT expressions use existing snapshots, typed
Nano operators and the verbatim shared C arithmetic header. I reference all
embedded helpers without calling them, preserving strict warnings and exact
header bytes after the separately recorded Clang prerequisite below.

I freshly built C-seed, Stage1, Stage2 and runtime tools at `748fba8e`.
`fresh-build.log` records both stage smoke tests, installed compiler checks and
C-seed independence. I later added one discarded-result test at7ca2df00;
production and tools stayed unchanged during each frozen gate.

My corrected production pin is `4f4d713e`. My integrated source pin is
`0e100cd02155cff6576c40c45b4699853ec41a86`, incorporating main2a0d3167 through
PR767. The merge resolved only an additive roadmap conflict; my production and
harness remain identical to4f4d713e. I rebuilt current C-seed, NanoVirt and
runtime tools. Stage1/Stage2 remain the actual fresh748fba8e binaries, not a
claimed final-head bootstrap. Their hashes are unchanged:

- Stage1: `ec838864c80daaa4b95d82bba226675741d2670e072ec1b867307f1522c88dcd`
- Stage2: `ae6d13948ede5570f61e843db7499433da18932e4b998cbc2ca10a4c31a1efb3`

I explicitly selected this checkout's `bin` through `NANO_HL_COMPILER_DIR`.
I did not execute old f38 tools/libraries or historical679 artifacts. These
are Linux AArch64 scoped results; I make no new Darwin or release claim.

## My measured gates

| Frozen gate | Result |
| --- | --- |
| Fresh748fba8e bootstrap and tools | PASS |
| Shared arithmetic direct/emitted helper and target guards | 2 methods PASS, 1.158s |
| Initial four reconstructed arithmetic methods | GCC PASS, 59.031s |
| Frozen7ca2df00 expanded harness | GCC5 PASS, 77.512s; Clang5 with 2 compilation failures |
| Existing transport/comparison/negation controls at7ca | 15 methods PASS, 215.447s |
| Corrected4f4d713e arithmetic | GCC5 PASS, 81.022s; Clang5 PASS, 80.541s |
| Corrected transport/comparison/negation controls | 3 methods PASS, 48.365s |
| Integrated0e100cd0 arithmetic | GCC5 PASS, 78.420s; Clang2 affected full-value/rounding methods PASS, 59.909s |

My five-method harness checks 51 explicit arithmetic input/result triples and
both original input patterns: 153 integer-observed bit checks. It covers all
four typed operations, quiet/signaling signed NaNs and varied payloads,
canonical arithmetic NaNs, both signed-zero divisors including nonfinite
numerators, finite endpoints, subnormal ties, cancellation and signed zero.
A multiply then add control observes the intermediate rounding boundary under
ordinary O2 and a separate O3 contraction-enabled C compilation.

Each positive fixture runs original verified VM/native, reconstructed standalone
C with strict warnings and ASan/UBSan, reconstructed Nano through C-seed,
Stage1 and Stage2 C routes, and canonical NanoVirt/Stage1/Stage2 modules through
VM/native. Canonical dumps show the corresponding typed operations. Helper
arguments/results, overwritten locals, both branch arms and a bounded pure
loop retain snapshots. Test-only instrumentation checks two once-evaluated
calls, discarded arithmetic results and original operand order. My external
shadows validate these fixtures; they are not reconstructed original shadows.

Other-tag and missing-operand analyzer controls remain refusals. Generic FLOAT
arithmetic, casts, truthiness and generic comparisons preserve prior output
files on refusal. I replace the old typed F64_ADD refusal assertion with these
actual positive multiroute checks, leaving adjacent unsupported cases intact.

## My preserved first failure and correction

At7ca, strict Clang rejected single-operation C output because three unused
static helpers from the exact embedded header triggered `-Wunused-function`.
I recorded `task_a7f544ab38b049d09e6afaf112036e8d` and roadmap8fd1501f before
correction. I retain the original full log, ordinary input assembly and emitted
C for both failures. This was a compiler diagnostic, not a runtime incident.

Reviewed4f4d713e adds conditional non-calling references in generated main.
I do not modify shared helper bodies, drop warning flags or weaken assertions.
Fresh corrected GCC/Clang results above qualify that repair. The bounded child
is ready for canonical-merge reconciliation; full reconstruction stays open.

## My sealed artifacts

[My report manifest](reconstruction-f64-arithmetic/report-sha256.json) seals
27 reports, including all first/final logs, failed fixture sources, wrapper
flags and source/tool snapshots. Each of my three preintegration frozen
before/after pairs matches all2120 entries. My final integrated before/after
pair matches all2125 entries. The integrated pair covers current tracked
source/tests/scripts and nine tool paths. I distinguish that source inventory
from the unchanged Stage1/Stage2 bootstrap provenance above.
