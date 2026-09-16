# Native map root lifetimes

I reproduce PR #303's lifetime defect at
`d7d02e93a129719eb3d45ce43f6f195a0bfcb690` with
`tests/nanoisa/fixtures/map_caller_lifetime.nasm`. The caller creates a map,
calls a function that allocates maps in a loop, then reads its retained map.
ASan reports heap-use-after-free in `nmap_find`: the callee's `nmap_collect`
destroyed that map. This is a runtime reproduction, not only source review.

I register a native root frame for each generated function. Before direct
calls I publish the live operand slots and locals, keeping caller frames
linked while callees run. Before backward branches and self-tail restarts I
refresh the current roots and collect. I unregister frames on every generated
return path. I do not scan stale temporary high-water slots.

Collection includes globals and traverses string arrays, record arrays,
nested records, map fields and tagged values. I traverse current aggregate
contents, not flattened snapshots of old elements. A deduplicated iterative
worklist prevents cycles from causing unbounded recursion. This collector
reclaims map objects and owned map-lookup strings; it is not a claim of
complete native allocation reclamation or a concurrent runtime collector.

Evidence on the integration branch:

- Regular translator gate: 1,723 checks and 1,073 shape checks pass.
- Fresh ASan/UBSan translator gate: the same counts pass; the runner checks
  actual instrumentation in translator and shape objects.
- Existing native/compiler tests plus the initial lifetime matrix: 26 methods
  pass.
- Expanded lifetime gate: four methods pass, including eleven retained-value
  cases, the original caller-map fixture, mutation during a callee's loop,
  backward conditional branches and 20,000 self-tail restarts.
- Every lifetime program executes in NanoVM and generated C with ASan/UBSan.
  Generated C checks a peak of at most 16 map/string owners and zero owners
  after entry cleanup. Leak detection is disabled because other native owner
  families remain separate work.

`make test-one-ir-compiler` now includes this lifetime gate. This repair alone
does not complete the release. MAC: `task_d3310bef8bd541ba9e1e267ee213eb9e`.

## PR #303 reconciliation

I compare its earlier map commits with my existing integration history:
`86454c80` storage matches `9ebc79d9`'s runtime fragment; `7ed63d5e` flow
matches `1e3ede7d`'s translator and shape sources; `a4ae73a5` tagged lookup
matches `3124f5fb`'s translator and shape sources. These selected production
files are byte-identical at each pair, not merely similarly named commits.
My current implementation includes later fixes that I retain.

The final branch commit adds the unsafe collector, a stress test, an older
test execution helper and import-kind plumbing. I retain my existing import
kinds, shape build integration and broader execution helper; I do not restore
its old nested-record rejection or current-frame-only collector. I import its
20,000-iteration map stress program into the VM/native lifetime gate with its
original eight-owner peak bound. I preserve the branch ancestry in the merge.
All five lifetime methods pass after reconciliation. The production translator
and shape sources remain identical to the sanitizer-tested repair checkpoint.
