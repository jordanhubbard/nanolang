# Resolved field representations in emission

My `AGG_GET` emitter now reads the instruction's resolved shape to select its
field representation and runtime kind check. I distinguish integer, string,
record, and arrays of each supported element representation. An unconstrained
array element does not become an integer by default in this lookup. Existing
flat inference remains the fallback when the graph has no resolved fact;
replacing that inference is still unfinished.

I added a non-allocating graph lookup. Unlike constraint construction, reading
a missing edge returns zero without adding a node or edge. Alias resolution
can compress parent paths; invalid projections still poison the graph.
Tests cover missing record fields, unbound array elements, aliases, recursive
edges and invalid array projections. Resolved nested record fields remain
explicitly rejected until their emitted storage is implemented.

## Verification

- `make -j1 test-nvm2c`: 1,048 AOT checks and 965 graph checks pass.
- `make -j1 test-nvm2c-sanitizers`: the same checks pass, with fresh object
  instrumentation verified by the sanitizer driver.
- `git diff --check`: passes.
- `make -j1 test-one-ir-compiler`: fails at function 20, `AGG_PACK field
  requires unsupported nested aggregate shape facts`.

This connects resolved facts to field extraction. It does not implement nested
storage, eliminate flat field inference, or complete compiler acceptance.
MAC `task_9c850e94e5a74b6f8941622e2872af23` remains open; claiming still returns
`agent_status_unavailable`.
