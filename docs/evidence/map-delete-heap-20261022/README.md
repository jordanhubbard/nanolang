# Heap-only exact map removal checkpoint

I implement the reviewed tombstone representation and direct checked private/ordinary DELETE adapters. No proof, opcode whitelist, recipe, invocation dispatch, native target, publisher or public selector changes are included. Selected DELETE still refuses.

First gate:2methods2.938s,4361 expanded direct assertions and10318 old private assertions. Expanded gate:2methods2.453s,4441 and10318. Final gate:2methods2.774s,4445 and10318. All use explicit Clang O1 AddressSanitizer/UndefinedBehaviorSanitizer with nonrecovering diagnostics on four compiled providers and direct harnesses. No test failures occurred in these gates.

Final controls cover a valid full live table's missing DELETE, wraparound collisions with counted-NUL keys, all-tombstone missing lookup, survivor reachability, independently allocated equal-content query at maximum references, same physical key, buffered ordinary and private lifetimes, exact measured full-topology/observation work plus one-below refusal/recovery, corruption restoration, allocation-free DELETE, sparse tombstone reuse and failed growth after committed deletion under transient/persistent denial followed by retry. Old private gate retains32 construction prefixes. Destructor balance is checked by the independent tracked allocator, not only object count.

Eight strict Clang/GCC16 cases pass (default/array-only/combined heap plus combined harness). Heap size remains6312 in the combined private configuration; old legacy growth allocated/freed remains7126/7126. The zero-tombstone full validator body and work schedule are byte-identical to d5bc83d23 (hash retained). No Analysis/VM/heap fields, proof allocations, budget constants or caps change. I did not rebuild/rerun the six proof-budget harness in this heap-only gate; prior exact six-row evidence remains in map-paired-publication-20261022, and this component changes no proof code.

Actual command:

```sh
RECORD_ARRAY_VM_CC=/opt/homebrew/opt/llvm/bin/clang RECORD_ARRAY_VM_CFLAGS='-fsanitize=address,undefined -fno-sanitize-recover=all' python3 -m unittest -f -v tests.test_record_map_ordinary_heap tests.test_record_map_heap
```

The existing required test-record-map-ordinary-heap target exercises the expanded direct harness; test-record-map-heap remains its old private neighbor. No new frontend/bootstrap is required for this checkpoint. Raw command/provider selection artifacts remain at paths in final.log; copied selection/log evidence and final source hashes are here. No full VM execution or current-head platform acceptance is claimed.
