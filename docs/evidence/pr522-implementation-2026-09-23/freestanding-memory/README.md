# My freestanding memory boundary

Linux LLVM 18 fails the original Wasm startup link on two memset references.
The preceding linux-generated-integration checkpoint retains that baseline.
My first correction replaced the validator scratch-array initialization with
volatile byte writes. It passed startup, then program 62 failed on compiler-created
memcpy and memset references from other runtime code. I discarded that partial
source change. Its logs remain here; its Darwin LLVM run was stopped and is
not qualified. Its completed generated-C result is not final-source evidence.

I now provide bytewise memcpy and memset in the shared Wasm runtime, keeping
native libc behavior unchanged. Volatile accesses prevent recursive lowering.
The exact helper names are reserved by the shared runtime predicate. My
current LLVM entry API already restricts entry identifiers to main or nano_.
The generated native source header is regenerated and its --check passes.

All three runtime-package methods and all three managed-string-core methods
pass on each of Darwin and Linux. The new O0/O2 memory-ABI fixture checks exact
bytes, boundary preservation, source preservation, return pointers, negative
and truncated fill values, zero-length operations, and no Wasm imports in
Node and Wasmtime. The owning package/core suites retain native ASan/UBSan
checks; I enable leak checks on both hosts. Package lifecycle checks also use
ASAN_OPTIONS=detect_leaks=1:detect_stack_use_after_return=1. Core tests retain
their own detect_leaks=1:abort_on_error=1 selection.

I independently rebuild and link the unchanged formerly failing program 62
with the corrected runtime, then execute it in both Node and Wasmtime. The
retained driver and command reports preserve the exact test and product hashes.

Darwin uses Homebrew LLVM, Node and Wasmtime. Linux uses the retained Ubuntu
24.04 ARM64 container, LLVM 18.1.3, Node 18.19.1 and Wasmtime 49.0.0. Full
corrected generated-C and LLVM/Wasm corpus gates are running separately, with
all original assertions and 240-second child deadlines unchanged. Their
completion remains required by task_efde11810990406ba36311702a0746d6 and the
aggregate parent; this checkpoint does not close #522.
