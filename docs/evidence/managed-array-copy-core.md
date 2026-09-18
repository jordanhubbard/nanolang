# My private literal and slice preparation checkpoint

I qualify production efbd6ea4 against merged VM prerequisite715. My additional
scalar-pointer ABI contract was committed first at4e6ca749. This checkpoint
adds no instruction, verifier-profile, shape-transfer or consuming LLVM
lowering. Childb702 and aggregate488/managed51da remain open.

My existing NEW still prepares capacity8. A private helper now accepts checked
initial capacity, and literal/slice preparation uses exactly max8/count. Literal
payload/tag vectors are borrowed and uint16-count bounded before input access;
I validate the finite element matrix before allocation. Prepared arrays retain
string children and remain private until complete. Slice creates a fresh
VM-policy handle, preserves declared storage and packed bits, and reacquires
source/destination slots after possible descriptor relocation. Empty private
packed buffers avoid null-pointer arithmetic. Scalar module wrappers borrow
all inputs and retain exact uint32 endpoint fallback/wrapping; later lowering
must consume its stack roots explicitly.

Failure leaves source arrays, input owners and public output unchanged. A
partially prepared array releases its edges and data. The runtime may retain
increased descriptor-table capacity for reuse after a later retain failure;
terminal disposal releases that internal capacity. I do not claim identical
allocation counts or heap pages to NanoVM.

My new two-method target gate passes native ASan/UBSan in production and testing
builds, import-free Node and Wasmtime, and separately generated native/wasm32
LLVM package ABI calls. Repeated/fresh Wasm instances cover reuse and disposal.
Controls include seven declaration kinds, counts0/1/7/8/9/16/17, source order,
all three portable cross-tag coercions, exact signed-zero/min-subnormal/max-finite
and NaN payload bits, duplicate string edges, descriptor growth during slicing,
copy mutation independence, retained child survival, legacy string-array copies,
clipped/empty/reversed ranges, native/Wasm scalar output ABI and endpoint wraps.

Deterministic allocation limits test buffer/table rollback and output sentinels.
A private test sets the existing checked reference counter near its limit,
then verifies rollback after one retained edge and refusal of the next; it
restores the fixture count before teardown. Actual bounded Wasm memory pressure
checks unchanged source contents, repeated failure, full release and reuse.
These corrected ordinary controls exercise defensive checks, not an unchecked
failure demonstration.

On efbd6ea4, the full adjacent gate also passes all52 existing managed target
methods in75.564 seconds, shape12, shared profile API, runtime/package tests and
string/array/boxed/packed core controls. Log: `/tmp/nanolang-array-copy-full.log`.
The initial and expanded focused logs are `/tmp/nanolang-array-copy-focused.log`
and `/tmp/nanolang-array-copy-expanded.log` (expanded2 methods3.069 seconds).
I used `NMS_NATIVE_CLANG_FLAGS=--gcc-install-dir=/usr/lib/gcc/aarch64-linux-gnu/13`.
No source-bootstrap, Darwin sanitizer or full release claim follows from this
runtime-only checkpoint.
