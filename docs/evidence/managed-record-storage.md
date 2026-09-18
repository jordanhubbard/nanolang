# My private managed record storage checkpoint

I qualify checkpoint2 of task_2dcceeb7ef38459093baaf52cf642239 at integrated
source/test pin `24459f661a99e002b0156e7591f0854a215bd3d0` on main d3f2085c (through736).
My production files are identical to independently reviewed4a5f7093.
Descriptor733 supplies descriptions with UNKNOWN authority; this private API
adds no source/profile permission and no nominal opcode lowering.

I bind immutable borrowed descriptors once before dynamic objects. I preserve
separate global layout and per-kind record identities, fixed ordered boxed
fields, exact scalar payload bits, retained GET and retain-before-replace SET.
Records and arrays share actual child owners, iterative zero-reference release
and checked standalone/prepared cycle collection. Binding and construction
failures preserve outputs and prior runtime state. No new wire section or
translation unit is introduced.

My focused native ASan/UBSan and import-free Wasmtime/Node controls pass with
production and NMS_TESTING builds. I also compile runtime LLVM IR separately
and link it with a header-only caller on both targets. I test same-shaped
separate definitions, empty records, NUL strings, exact signaling-NaN payload
bits without arithmetic, shared field replacement, retained child lifetime,
array/record receiver distinction and unchanged failure outputs. A 4000-record
chain releases iteratively. Repeated 1000-cycle controls cover record self edges,
record-array cycles, duplicate edges, an external temporary root and exact
single removal of dead-to-live edges. Wasm runs within a4MiB memory limit across
repeated calls and fresh instances; this qualifies private runtime ownership,
not generated nominal source execution.

Checked reference-count saturation controls preserve partial-retain rollback,
GET output and SET contents. Allocation budgets0/1/2 fail field buffer/table/
prepared workspace respectively, preserving original table/workspace/owners;
budget3 succeeds. Recovery succeeds after each failure. Nonempty or previously
allocated tables refuse descriptor rebinding. Dedicated slot identity survives
table growth and is reset during reclamation.

At frozen17d2f700 I passed the full affected command:

```
NMS_NATIVE_CLANG_FLAGS=--gcc-install-dir=/usr/lib/gcc/aarch64-linux-gnu/13 \
  make -j6 test-llvm-managed-strings test-managed-record-plan test-verifier-profiles
```

That includes69 generated managed methods in99.980s, private runtime/package/
array/graph/record controls, and unchanged shared-profile/descriptor controls.
After a clean restack over734/736, the same production plus an explicit test ABI
assertion passed the two record target methods in3.642s and the shared profile
gate. I did not repeat the unaffected full corpus or claim a compiler bootstrap.
All runs passed; [artifact hashes](managed-record-storage-artifacts.json) retain
source, package, tools and logs.

Ordinary heap-bearing authority15f, field-origin/provenance analysis, nominal
lowering, union/tuple/map/callable transport and full aggregate488/managed51da
remain open. I do not infer affine/resource storage permission from field shape.
These Linux results do not establish Darwin or release acceptance.
