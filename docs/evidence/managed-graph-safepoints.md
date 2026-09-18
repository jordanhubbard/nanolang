# My prepared graph collection checkpoint

I qualify runtime child `task_c01fc72a78d74e108a5a55a417f4f9fc` under task4070.
My pre-code contract is `143ac4b3`; reviewed production is `d05d4113`.
My full adjacent gate ran with immutable source/tests at `3620544d`; subsequent
changes add explicit target-size/alignment controls and this evidence, not
production behavior.

I reserve collector workspace before enabling preparation and replace it only
when replacement descriptors and workspace both exist. I reuse the validated
collector without allocation. My private module adapters retain first errors,
refuse nested begin without changing active roots, and collect after frame-root
cleanup while globals remain held. Existing leaf entry ABI, standalone collector
allocation behavior, eligibility and emitted instruction paths remain unchanged.

## My observed Linux target checks

I use `NMS_NATIVE_CLANG_FLAGS=--gcc-install-dir=/usr/lib/gcc/aarch64-linux-gnu/13`.

- `python3 -m unittest tests.test_managed_graph_safepoints -v`: two methods pass
  before the full run (3.849 s). They compile native production/testing code with
  Clang ASan/UBSan and import-free wasm32 with Wasmtime and Node.
- Separately emitted native LLVM and wasm32 runtime IR link with a counted-root
  harness. Transferred call arguments and returned temporaries remain roots;
  globals keep identity through failure and reentry. Each entry creates 30,000
  mutual cycles under a 1 MiB Wasm limit; Node repeats three entries on each of
  two fresh instances. This is runtime ABI qualification, not generated graph
  bytecode admission.
- Initial enable failure recovers; replacement-table success followed by
  replacement-workspace failure preserves both old pointers, capacity, live
  objects, free head, output and allocation count. These deterministic failures
  retain the old-plus-new allocation peak, rather than freeing old storage early.
- Prepared collection succeeds with allocation budget zero and after real Wasm
  allocator exhaustion. Scratch is reused, dead cycles are reclaimed, external
  children survive, and later allocation/disposal succeeds.
- `make -j4 test-llvm-managed-strings test-managed-array-eligibility
  test-verifier-profiles`: passes. This includes 61 managed target methods
  (82.341 s), 27 leaf/graph shape methods (3.448 s), four graph/preparation
  methods (6.308 s), profile controls, runtime/core/package regeneration and
  target links. Existing nested-graph backend refusal preserves prior output.

The final focused rerun passes two methods in 3.855 s and adds native64/wasm32 layout arithmetic controls at zero,
eight and maximum uint32 capacity without attempting oversized allocations.

I have not changed the shared profile or emitted safe points. Task4070 remains
open for a separately reviewed generated lifetime/admission contract, and full
aggregate488/managed51da/platform/release acceptance remains open. I make no
source-bootstrap or Darwin qualification claim for this Linux runtime slice.

## My retained logs

- `/tmp/nanolang-graph-safepoints-pressure.log` — SHA256 `1ff737943f3eb779d93257e058b9df43815df1aa5d0998082d037e77118e83f9`.
- `/tmp/nanolang-graph-safepoints-adjacent.log` — SHA256 `22e682687dc398c740320dfdf6e38df1a9774872eb087d47145404575e5ce4cf`.
- `/tmp/nanolang-graph-safepoints-size-boundaries.log` — SHA256 `e2b5fc7f11ec19d73f8fef59564ad992b411a690e242d12f8b7b7f69e9e32210`.
