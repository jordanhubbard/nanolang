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

Evidence on the 5.0.1 release candidate:

- Regular translator gate: 1,761 checks and 1,076 shape checks pass.
- Fresh ASan/UBSan translator gate: the same counts pass; the runner checks
  actual instrumentation in translator and shape objects.
- Existing native/compiler tests plus the initial lifetime matrix: 26 methods
  pass.
- Expanded lifetime gate: five methods pass, including thirteen retained-value
  cases, the original caller-map fixture, mutation during a callee's loop,
  backward conditional branches, 20,000 self-tail restarts and non-self-tail
  root-frame teardown.
- Every lifetime program executes in NanoVM and generated C with ASan/UBSan.
  Generated C checks a peak of at most 16 map/string owners and zero owners
  after entry cleanup. Leak detection is disabled because other native owner
  families remain separate work.
- `make test TEST_TIMEOUT=3600` completes successfully with the acknowledgement
  below. Its comprehensive source run reports 225 passed and 0 failed; its
  additional compiler, runtime, example, documentation, Forth and failure-path
  gates also complete successfully.

`make test-one-ir-compiler` now includes this lifetime gate. PR #303's other
commits and ancestry still require reconciliation; this repair alone does not
complete the release. MAC: `task_d3310bef8bd541ba9e1e267ee213eb9e`.

For 5.0.1 I also verify scalar floats at local and operand-stack safepoints and
non-self tail-call frame teardown. Float and integer-array storage are never
interpreted as pointers. I track the patch release as
`task_a94efdfee3486a0814f93336cf5c052c`.

I ran a clean three-stage bootstrap before the full release suite. The release
documentation acknowledgement is: “5.0.1 changes internal native map
reclamation only; syntax, CLI, README guidance, and presentation claims are
unchanged.” I keep that acknowledgement out of the historical negative-control
tests so they continue to prove that stale release prose fails closed.
