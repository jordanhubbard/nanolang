# My bounded Darwin lowering window

I retain this observation under
[#1265](https://github.com/jordanhubbard/nanolang/issues/1265), after my
[coarse phase diagnostic](../darwin-stage1-phase-diagnostic-20261126/README.md)
and [original1800s Stage1 failure](../darwin-stage1-generation-timeout-20261126/README.md).
My original manifest binds37 records, SHA-256
`0df3403881dc5132e7f09dce4c52e682c7876bd7582038cf312c7bd1ba502bc1`.
A post-terminal schema/opcode analysis adds two separately sealed records,
supplement SHA-256
`263ab472dec0411027b9626621b821118c65f75ba748d3d14aa6086c1a262570`.
I leave the original manifest unchanged. My reader verifies39 new records,
35 parent records,68 original-failure records and4683 pinned Git source blobs.
Compiler and whole-release acceptance remain false.

I use a new APFS root and a private observation-only copy of
src_nano/compiler/nanoisa_codegen.nano. Twelve reversible insertion groups add
phase markers only inside nisa_emit_module_indexed_body; one useful direct
shadow is added. Removing those insertions and that shadow restores every
original source byte (SHA-256418e9cd9ad88bf9469c618a60bd01d6a3806199dff86dd41d366c97d2bd7cd4f).
The observed file SHA-256 is
d942fbc7e4c0591f82bfdf98f78d9343009c04e1ef0474f6c462331c90eff60d.
All original assertions and return expressions remain present. The lookup
helpers and their empty-print loops are byte-identical; original print statements
in the observed body remain in place. Probe I/O and
allocation can affect timings; exact reversal is not a semantic-equivalence
proof or permission to ship the overlay.

Original-helper seed/verify preparation passes292.160060/4.582147s under
stricter diagnostic600/60 limits; its command takes305.768664s. I stop the helper
before Stage1 generation. I then allow600s to reach program-lowering-start and
request a120s window inside lowering, with no-generated-C guard intact.
Lowering starts427.759772s after VM launch. The observer requests termination
120.076899s after that marker; the VM exits-15. Total VM observation is547.883336s
including cleanup. This is a distinct targeted observation below the original
1800s qualification limit, not an unchanged full bootstrap retry. Original
compiler shadows30 and ordinary default10 remain unchanged.

My marker offsets below are wall-clock seconds after program-lowering-start,
including observation overhead:

| Marker | Seconds |
| --- | ---: |
| module-enter | 0.137 |
| reset-done | 0.276 |
| declarations-done | 0.546 |
| ownership-done | 3.668 |
| index-start / index-done | 3.788 / 3.909 |
| globals-start / globals-done | 4.019 / 4.489 |
| queue index0: main | 4.598 |
| queue index32: __nano_module___39_sn_owner | 21.551 |
| queue index64: parse_env_int | 115.356 |

A queue marker is before that entry's emission, not a completed-function count.
I observe the per-function emission path after setup, but no whole lowering
return. The interval from queue32 to queue64 is93.804746s. It does not identify
which function or helper accounts for that interval. I select no optimization
from a static lead and do not claim the cause of the original1800s failure.
These runs differ in source probes and preparation; their frontend timings are
not a controlled performance comparison.

The probe seed retains1350 functions, feature bits2 and byte-identical signature,
layout, global and link sections. Its code and constants change as expected.
A schema-bound independent instruction walk checks both raw seeds through the
same57 opcode kinds, with no kind introduced or removed. The extra instructions
include16 marker calls and the progress condition; this check does not establish
performance equivalence or native qualification. I retain the generated schema
bytes and full histograms in the supplement, bound to the tracked source hash.

Source/tool/seed/host/working-input guards pass. Exactly one of4683 tracked files
changes in the diagnostic copy, by the declared overlay; parent sources and
cold images remain unchanged. All seven cold images, nine selected tools, six
new hosts and32 additional frozen-parent files match their bound identities.
The controller is absent, no owned process remains and its reservation releases.
I retain the actual new seed/host bytes, original prefix work, full streams,
request, observer source, baseline/observed NanoLang and reversible binding.
No production NanoLang/C source, owner checkout, installed compiler, global
package/toolchain/SDK or NAS/TCC/migration state changes. The #1262 handoff is
still pending.

I next localize the per-function cost with bounded measurements and original
lookup/duplicate/owner/body/indirect-call/empty-print contracts intact. I still
require a demonstrated repair, fresh original17-step bootstrap/raw equality,
installed/no-C-seed/native36/emitter12 checks, current compiler-source/private,
pure translator/body/runtime/profile/ownership/installed cutover, exact-candidate
CI and all69 release requirements. No tag or release follows from this window.

```sh
python3 docs/evidence/darwin-current-source-20261124/diagnostics/darwin-lowering-window-diagnostic-20261126/read_terminal.py --repo .
```

The reader executes only Python evidence readers and data analysis. It runs no
compiler, VM or product and makes no current process-liveness claim. Without
--repo it explicitly reports zero pinned Git source-blob checks.
