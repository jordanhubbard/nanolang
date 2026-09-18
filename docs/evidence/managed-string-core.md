# My non-admitting managed-string core

I implement child `task_bfe3bb8672c04bcea56dede3f531aee7` of the open managed
runtime task `task_51da49b39230468784da3481b893563b`. My source/test checkpoint
is `e22e276a`, restacked without source changes onto main `5193858f` (PR #625).
My parent reviewed the production core independently and found no scoped
blocker. This checkpoint changes no verifier profile, opcode, translator driver
or executable eligibility.

`managed_strings.c/h` supplies context-local literal/dynamic handles, owned
immutable byte allocations, transactional descriptor growth, retain/release,
explicit status packing and entry/disposal guards. Native allocation pins a
64-bit ABI and calls its declared malloc/free backend. My freestanding wasm32
backend owns an aligned split/coalescing free list after the linker heap base;
it reuses storage before requesting pages and checks widened size arithmetic.
All free-block headers reside in allocator-owned memory. Wasm remains import-free.

My tests distinguish the implementation from a bump-only arena:

- An additional retained alias survives descriptor-table relocation. Bytes
  include embedded NUL; an empty allocation has valid nonnull storage.
- Deterministic byte-allocation and later table-growth failures preserve old
  descriptors, aliases, counts and the caller's unpublished output handle.
- Two adjacent freed 24,000-byte blocks provide a 40,000-byte first-fit allocation at
  the first block's address, while a neighboring live string retains its bytes.
  A spare tail block cannot satisfy that address assertion accidentally.
- Two thousand allocate/release cycles of 60,000-byte strings leave no live objects
  and do not increase the established Wasm page count. Terminal disposal frees
  the descriptor allocation too. Separate contexts retain their own aliases.
- A real one-megabyte Wasm maximum forces memory.grow failure. Existing strings
  remain readable, release reclaims storage, and subsequent allocation succeeds.
- Busy entry/disposal returns preserve state. Packed successful negative entry
  results, failed status results, idempotent disposal and disposed entry are
  explicit checks. These guards do not implement application frame cleanup.

`make test-managed-string-core` passes three methods on Linux ARM64 in 1.229s
with GCC and 1.273s with Clang. Both compile/run native C with ASan/UBSan and
leak detection, plus no-test-hook production builds. The Clang command explicitly
selects installed GCC 13 support using
`CC='clang --gcc-install-dir=/usr/lib/gcc/aarch64-linux-gnu/13'`; automatic Clang
selection reported an installation warning before compilation on this host.
I retain `/tmp/nanolang-managed-core-integrated-gcc.log` and
`/tmp/nanolang-managed-core-integrated-clang.log`.

Each run also builds production and test wasm32 products without libc or host
imports. Node runs four runtime groups for three rounds on each of two fresh
instances; Wasmtime runs each group independently. Production smoke runs
without NMS_TESTING, and generated production LLVM IR has no test hooks and
passes LLVM verification. This tests actual memory instructions/linker heap
placement, not only a native approximation of the Wasm allocator.

My Make target joins the existing Wasm acceptance target. I did not rerun a
compiler bootstrap for an unlinked standalone core, and I claim no Darwin or
application-level managed-string acceptance. Function/global root ownership,
cleanup propagation before target traps, emitter/runtime packaging, actual
managed string operations and portable conversion semantics remain under the
parent contract. The separate VM substring prerequisite is still open. My
full LLVM/Wasm coverage and publication hold remain unchanged.
