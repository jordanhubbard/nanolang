# My STRING binding hint and local repair checks

I retain the repair under #1265/#1253, product commit
`2006c758df3c67c27f623ff9f7e1e1baaf56c136`, hosted-gate integration
`f4e220ff1c69ce19000961de98caad3935919974` and corrected local fixture
`d18fed86140b6a4cf983898ac32a079002f6b784`. Original hosted and full-release
qualification remain required.

My source 6e09 diagnostic overlay changes only env/eval counters and reporting.
Its exact reverse patch recovers the original Git bytes. All 1,139 shadows
complete under the original ten-second supervisor. My 2,278 cumulative snapshots
record 91,472 declaration scans: 883 hits and 90,589 misses, with 746,387,154 slot
probes. Assignment adds 8,566 scans / 7,792 misses / 72,875,558 probes. Reversing the
scan barely affects the dominant no-alias work; reported reverse probes are an
upper bound derived from the first forward hit, not exact duplicate selection.

I reserve one bit in existing GCHeader flags for a monotonic fact: this managed
STRING has been published in an environment binding. Never-bound allocations
can skip the alias scan. Previously bound strings retain the original exact
search and retain behavior. Every publication marks, including folded imported
literals, borrowed/identity replacements and direct static/dynamic/list loops.
The bit does not assert current ownership, liveness or environment membership.
GCHeader layout and allocation code remain unchanged; all other flags survive.
My [contract](../../../CSEED_STRING_BINDING_SCAN.md) records the invariant.

My permanent regression checks fresh/seen/unmanaged/recycled allocations,
other flags, all 128 alias positions, multiple aliases, replacement/identity,
borrowed values and checkpoint/scoped retention. It exercises the actual folded
import and a native callback inside static-array, dynamic-array and legacy-list
STRING loops, including a nonempty shadow. Removing the import marker fails its
binding invariant; removing loop markers fails the first loop assertion.

My new harness initially fails on an unused included helper, then its loop
extension uses a non-public constructor. I retain both sources/refusals and
correct only the fixture, preserving warning policy and assertions. The broad
local gate then passes in 42.412655s: new hint/index tests, interpreter, scoping,
STRING reference/carrier, list, checker-context and GC targets, with unchanged
source maps. The final fixture cleanup is separately rerun and passes.

My complete ASan/UBSan build uses detect_leaks=1. Its initial new gate finds
55 bytes: 47 from a temporary static-array literal argument and 8 from the test's
caller-owned dynamic string. I preserve that failure, give the same nonempty
shadow a global-owned argument and explicitly free the caller-owned dynamic
payload. All assertions remain. The corrected leak-enabled gate and existing
index gate pass. I retain full build commands, binaries and final sanitizer
source map; the initial sanitizer runner did not record a pre-build source map.
The corrected test and included env source are byte-checked against the local
candidate. This is local testing, not native release qualification.

A minimal source control reproduces the 47-byte literal leak using actual 6e09
env/eval/module C with otherwise unchanged common code and the repair's
header-declaration-only additions. The first probe lacks main and is refused
before execution; its corrected version passes assertions and LeakSanitizer
reports 47 bytes in three allocations. I track temporary-expression ownership
in #1266 rather than attributing its introduction to the binding hint.

I preserve an original-limit emitter attempt that stops after 1,127 completions,
with signal 14, parent_timeout=true and no recorded assertion failures. A
separate hardware CPU observation completes all 1,139 shadows, flushes, is
reaped cleanly and exits 0 at the original limit: 9.756433114s from parent start.
Its compiler/sampler exit 0 after 40.768457s. All source/tool guards match. The
823 retained samples have no reported loss; binding's first-non-inline weighted
share is 1.53%, versus 13.71% in my unchanged-source prerequisite profile. These
are sampled event weights under different host scheduling, not a controlled
universal speedup, exclusive timing attribution or a Darwin pass. I retain both
ARM PMU event streams and all decoded samples/reports.

My manifest SHA-256 is
`57fc7fef460539654ce6dcb5c9ba16a97fdaa7c11fb88ebd2a0760955640a09a`:
135 records, 182 chunks and 312,864,563 uncompressed bytes. My reader checks all
bytes, 4,676 original / 4,696 candidate source hashes, the exact counter reversal,
marker-removal refusals, sanitizer and prior-source control boundaries, original
emitter refusal, all counter snapshots and decoded sample identities/weights.
It checks unchanged GCHeader text/layout and runs no product or fresh perf
decoder. A changed manifest refuses before reconstruction.

```sh
python3 docs/evidence/c-repairs-linux-20261126/string-binding-hint-20261127/read_string_binding_hint.py \
  --archive docs/evidence/c-repairs-linux-20261126/string-binding-hint-20261127/archive \
  --repo . --workspace /path/on/large/volume
```

Hosted [run46042142609](https://github.com/jordanhubbard/nanolang/actions/runs/46042142609)
is pinned to the local candidate. Its Darwin bootstrap job builds production
tools and passes guards, then refuses my new test's undeclared mkdtemp under
strict C99 before original bootstrap/native execution. My later test-only
_DARWIN_C_SOURCE correction is separate from this local archive and requires
new original hosted qualification. I retain that job separately; other live
jobs continue. Neither this receipt nor a successful diagnostic closes #1265,
#1253, #1262, temporary-array ownership, pure translator/paired/cutover,
frozen CAST_STRING assertions, exact-candidate CI or any unmet 5.1 requirement.

My strict-header correction passes the local Linux regression again; that
compile/run is retained in `darwin-feature-local/` and claims no Darwin execution.
The selected four-file Darwin failure manifest is
`ec0ef664bdc83093051f783240c7e3b1e3b17f992c75fc7a3653caf978490658`;
`read_darwin_refusal.py` checks its bytes, exact old/new test sources and
that bootstrap/native execution was skipped.
