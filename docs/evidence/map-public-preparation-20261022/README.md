# Borrowed map preparation only — 2026-10-22

I qualified immutable preparation and fresh import validation against a borrowed
ordinary VmState. No stack allocation, activation, selected dispatch, public VM,
native selector, loader, or publisher admission is enabled. Public refusal remains
asserted. The caller owns the original module, caches, constants and ordinary heap;
they must remain exclusively stable. Output storage must be disjoint from all
caller-owned storage. Accessible stack/global/constants-table overlaps refuse.

## Source and commands

Base e5164fe74; reviewed contract 3e9a51aca. The new private owner holds a single
boundary plan, copied function/instruction/physical-layout rows and counters.
The positive-only reserved boundary wrapper preserves the original constructor's
0/0 route. Preparation checks all decoded/dispatch instructions, including dead
rows, exact code/parameter/constant correspondence, and the same plan's imported
global obligations. It publishes only after complete success. It never changes
VmState, collector metadata, globals, or the caller's stack address/capacity.

The required target is `make test-record-map-public-prepare`, also in test-units.
A source-only Make dry-run succeeded; no prerequisite build was invoked here.
The component used the exact retained inputs in final-environment.json:

```sh
python3 -m unittest -f -v tests.test_record_map_public_prepare.RecordMapPublicPrepare
python3 -m unittest -f -v tests.test_record_map_import.RecordMapImport
python3 -m unittest -f -v tests.test_record_map_boundary.RecordMapBoundary
```

The first two use all eight RECORD_MAP_MODULES inputs (four original map fixtures
and four source-global fixtures). The boundary neighbor uses its original four
only, and the exact Make query closure in boundary-corrected-environment.json.
Compiler/provider/module hashes and final source hashes are retained. Compilation
is isolated O1 ASan/UBSan with provider instrumentation preserved; no frontend or
bootstrap was built. The filesystem had35GiB free and64% free memory at first gate.

## Actual results and limits

- First unchanged-four switch gate:1 method8.718s,220 checks. Seed reservation
 84,068,462 bytes/16,980,536 logical work; C-hosted and retained stages
 84,068,161/16,979,892. Both fit unchanged128MiB/32Mi-work caps.
- Final source and both dispatches:12,495 checks each passed in the combined
 final-neighbors invocation, alongside old import13,164 checks each. That
 invocation subsequently failed only the boundary neighbor's link setup, so it
 is not reported as a completely green five-method invocation. The immediately
 preceding standalone extraction gate passed2 methods16.747s/12,493 each; the
 two additional checks precharge the same32F policy in the whitebox helper test.
- Corrected boundary invocation:1 method8.928s,754 linked/1685 whitebox checks;
 old owner40/row16/retained25,536 and62 allocation positions unchanged.
- Twelve strict GCC16/Clang cases passed. All six legacy allocation/work rows
 match retained baseline bytes exactly; Analysis remains39,840 bytes.

Actual new owner14,512 bytes, instruction80 and field20 are statically asserted.
Full boundary reservation remains live. Stack growth reserves old+new capacity
bytes, but only possible copied old bytes+64 logical overhead as work: realloc
never initializes newly reserved capacity. Normal capacity4096 rounds to1048576;
real odd capacity32769 rounds to524304. Actual no-growth preparation preserves the
already allocated stack and reserves no new stack bytes. Zero-capacity NULL is
valid; nonzero-capacity NULL refuses. No allocator-internal timing bound is claimed.

The independent allocation oracle measured85,653 peak bytes beyond169,453 live
caller/prior-owner bytes, against84,065,265 reserved bytes. This large conservative
reservation includes the unchanged full proof overlap and planned stack overlap;
planned stack allocation does not occur in this slice. All69 reached allocation
positions were denied transiently and persistently, with prior-owner query data,
caller bytes and80 collector-only headers preserved, followed by independent
balanced retries. Exact measured byte/work caps succeed; +1 and UINT64_MAX refuse
without replacing output. MAX refuses before proof allocation; late-cap failures
may allocate and reversibly free proof objects.

Each original is prepared before and after two ordinary legacy executions, with
explicit result consumption. Those runs establish ordinary import state, not new
selected execution. Source-global replacement, buffered aliases and constants
remain intact. Malformed caches, call handles, boundaries, constant hash/terminator,
headers and accessible overlaps refuse then succeed after restoration. A real
one-parameter helper proves the valid-domain STRUCT→ARRAY mismatch against copied
immutable parameter facts refuses in the actual static correspondence helper;
restoration passes. The private helper is not an executable trusted-plan API.

## Retained failures

The expanded harness first failed compilation because its copied legacy baseline
helper was absent and an instruction type name lacked the N prefix. Both were
test-only corrections. The tag255 negative then expected INVALID, while the
original proof correctly returned UNRESOLVED with `I require ordinary parameter
tags.`; the exact expectation was corrected without widening admission.

The combined boundary neighbor initially omitted the compiler-input archive from
a manually narrowed link selector. The exact Make query closure corrected this.
The next attempt mistakenly supplied all eight modules to the older four-module
annotation harness; it correctly refused the source-global field-count mismatch.
The final retry used the original four inputs and passed. Neither older harness
nor product authority was changed to accommodate these selection mistakes.

Raw logs/artifacts remain under /private/tmp/nanolang-map-public-prepare-20261022
and the temporary folders printed by each test. This receipt retains compact
text/status/pin evidence, with trailing blank lines normalized; no binary bundle.
