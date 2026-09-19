# My closed mixed Samples runtime qualification

I qualify runtime/native admission at production `675395d7e` and final fixture
`e3beb609f`. Source producers remain unchanged. I have not run the original
Samples/PREFIX source through a widened producer, admitted managed fields in an
owner, or closed the mixed/ownership/product parents.

My [manifest](mixed-samples-runtime/manifest.json) seals 66 reports, 350 retained
artifact hashes and 17 current tool/library inputs. Every run's tracked-source
and actual compiler-tool maps agree before/after. Object inventories explicitly
record build changes rather than claiming all objects were already frozen.
The final Clang run also inventories the actual VM, translator, fixture executable
and compiler wrapper before/after. Generated programs contain the qualified
embedded runtime; no runtime source-path lookup is needed.

## Passing boundaries

| Gate | Result |
| --- | --- |
| Final 12-case GCC native/runtime | PASS, 33.899 seconds; strict C11 O0/O2, ASan/UBSan/LSan, exact regenerated C, allocation failures and recovery |
| Final 12-case Clang native/runtime | PASS, 18.061 seconds, same optimization/sanitizer/fixture boundaries |
| VM lifecycle | 2,006 checks: all four APIs, repeated activation, direct-core true/false ASSERT continuation, typed FLOAT-or-VOID failures, aliases/replacement, complete synthetic shadows, owner forwarding, NaN/zero, lower-index calls and repeated/zero-iteration paths |
| VM heap injection | 13,358 checks; 488 budgets, 440 actual failures, every case/API reaches its uninjected terminal and recovers |
| Separate fresh admission injection | 127 checks: first verifier-allocation failure, all four APIs and output sentinels, generation preservation, max-stack output and native publication, followed by recovery |
| Private composition/provenance | 112,482 / 339 checks; private runtime_admitted remains false |
| Existing profiles | STRING print/fields/joins, multi-caller references, value graphs/results and binary64 PASS, 48.841 seconds |
| General verifier/native | PASS, 186.913 seconds; verifier 96 tests, shape 1,365 checks, native 2,422 checks |

I check empty stacks, frames and reference contexts before any VM collection.
My bounded fixture graph includes all buffered objects and their STRUCT/flat
FLOAT-array children. Exact internal incoming edges must account for every
refcount, and graph nodes must account for every allocation above baseline.
A deliberately retained external reference must fail this predicate; releasing
it restores the predicate. Only then may collection restore the exact baseline,
before destruction. Generated native cleanup checks Nms roots before finish and
disposal; the harness independently checks every tracked allocation afterward.

The VM heap injector covers heap.c allocation sites, not frame-reserve or every
analysis allocation. The separate admission injector forces the first allocation
in verifier.c; existing private-query sweeps cover their documented analysis
objects. Existing graph/result preflight regression controls pass, but I do not
call those new mixed-frame allocation injection. Native sweeps intercept emitted
owner and embedded managed allocations and reach a noninjected terminal within
the fixed 256-attempt ceiling. The sanitizer claim covers generated native
programs; the linked VM gate uses the ordinary built objects. Darwin is unmeasured
in this seal. My compiler-specific fixture policy enables LSan for GCC/non-Apple
Clang and preserves the existing Apple-runtime exception; no Darwin LSan result
is inferred.

## Preserved first terminals

1. `1e98283f0`: compiler build status2, 6.981 seconds, misleading indentation in
   compiler-side opcode emission; no fixture execution.
2. `d6aad596d`: build/private queries pass; runtime allocation-count assertion
   stops before destruction, 1.118 seconds. Allocation count alone did not
   establish external-root retention.
3. `5ab080edb`: case0/API0/repeat0 measures buffered STRUCT refs0 and ARRAY refs1;
   direct-zero assertion stops before collection, 1.117 seconds. The corrected
   graph accounting measures the parent's internal child edge explicitly.
4. `7937fc15b`: all nine generated programs fail strict compilation at a one-line
   cleanup footer, 3.521 seconds; no generated native execution.
5. `c9e4289f1`: new loop fixture's generic ADD is refused at offset251 before that
   module's emission/execution, 1.768 seconds. The reviewed test-only correction
   uses the existing admitted I64_ADD, without changing production acceptance.

Each terminal remains independently sealed. Corrected runs do not relabel them.
The full source, parent ownership and release acceptance remain open.
