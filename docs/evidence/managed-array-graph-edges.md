# My private nested edges and iterative release checkpoint

I qualify production `a02b62a6` under graph child52d03 before implementing its
collector. My unchanged tagged-value ABI now validates ARRAY7 with the existing
live array-kind lookup. Boxed arrays retain array children, including aliases
and cycles; packed coercions stay unchanged. Kind3 is named BOXED_ARRAY with the
old BOXED_LEAF_ARRAY numeric alias retained. Private declaration7 uses boxed
storage. No shape/profile/opcode admission changes.

I reuse retain-before-commit writes, counted literal preparation and shallow
slice preparation for nested edges. Zero-reference release uses an allocation-
free descriptor worklist; outgoing edges are processed before a slot reaches
the allocator free list. Valid duplicate edges decrement exactly once each.
Pure cycles retain positive counts until the next collector checkpoint; terminal
context disposal already frees their buffers once without recursive traversal.

My first focused run passes three methods in 5.214 seconds: the new graph target
plus two existing copy/package methods inadvertently discovered through a test
class import. I replace that import with a module alias so later runs avoid
unintended duplicate suite discovery. The final graph target passes in 2.151
seconds in the full gate. It builds production and NMS_TESTING configurations
with native Clang ASan/UBSan and import-free Wasm, runs each export through
Wasmtime, and repeats exports across two Node instances. I test duplicate array
edges, exact GET/POP ownership, self replacement, nested alias mutation through
shallow copies, source teardown, legacy promotion, packed refusal, allocation
rollback, self/mutual cycles broken before normal release or terminal disposal,
and a 4,096-array chain released with allocation disabled.

The frozen adjacent run passes all61 existing managed instruction methods in
79.506 seconds,18 shape methods in2.401 seconds, shared profile and required
runtime/package/core targets. Logs are `/tmp/nanolang-graph-edges-focused.log`
and `/tmp/nanolang-graph-edges-full.log`. Root independently reviewed the
production worklist and surrounding publication paths without a scoped blocker.

This result does not establish cycle reclamation before disposal, graph-aware
shape admission, emitted collection safe points or full runtime acceptance.
The graph child and full aggregate488/managed51da parents remain open. I make
no source bootstrap, Darwin sanitizer or release claim.
