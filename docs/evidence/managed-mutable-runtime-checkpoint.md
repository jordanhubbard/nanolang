# My mutable-array runtime checkpoint

I recorded adapter task `task_41b195abe9734b0aa818335d6d49a7a1` and contract
`8bcf0359` before implementation. This checkpoint is frozen at `4a429e44` on
base `c9e2a16f`. The task remains open for emitted ownership, shared eligibility
integration, link closure and actual opcode admission.

I add VM-policy capacity 8 construction and checked doubling to my private
runtime. Policy and declared kind survive descriptor relocation and reset on
teardown. Legacy private empty-buffer factories retain their policy. My new
split variant shares the byte loop but constructs boxed tagged children before
publication, so replacing a prepared VM-policy element does not allocate.
Legacy private string-only promotion still allocates; I do not extend the
no-allocation claim to it.

My module adapters use scalar payload/tag arguments and explicit scalar output
pointers. GET retains a child, POP transfers it, and failed outputs remain
unpublished. Append/set borrow operands here. Emitted helpers must still consume
those operands and transfer the receiver correctly; this checkpoint supplies
none of that lowering. Bounds status 7 is appended without renumbering 0..6 and
survives both status clamps. The old read-only split/get/length route remains.

The first two focused methods passed in 2.847 seconds. Expanded methods passed
in 2.857 seconds (`/tmp/nanolang-mutable-runtime-expanded.log`), covering:

- Exact kind/policy through descriptor relocation; capacities 8/16/32/64;
  checked growth thresholds through the pure capacity helper without huge allocation.
- Prepared boxed split bytes including NUL/high bytes, same-child replacement
  under zero allocation budget, retained GET and last-child POP lifetime.
- Descriptor/buffer constructor failure, every boxed split child/table/growth
  allocation stage, output preservation and retained source aliases.
- Failed append growth preserving an alias's old buffer/length/content;
  one hundred reclaim/reuse rounds and actual one-MiB Wasm pressure.
- Bounds status, first-error preservation, later entry reset, independent
  instances and terminal disposal.

I compile the complete runtime fixture with native Clang ASan/UBSan and build
import-free wasm32 in production/testing variants. Node repeats core/failure/
pressure groups across two fresh instances; Wasmtime runs each export. A second
method appends literal LLVM calls to the exact target-specific packaged runtime,
verifies that IR, and executes native and Wasm scalar-output ABI calls. It does
not infer a shared C aggregate-return ABI.

I statically audited the later verifier link consumers: Makefile.gnu,
examples/Makefile, src/nanovirt/wrapper_gen.c, modules/nanoisa/module.json and
modules/forth_see/module.json. The verifier-cleanup test includes verifier.c and
links the shared objects. Other search hits are documentation/schema assertions.
This checkpoint adds no analysis dependency; its eventual admission connection
must update and exercise those consumers before publication.

Independent production review found no scoped blocker. This checkpoint changes
no verifier profile, LLVM instruction lowering or public mutable-array admission.
Parents488/51da and task41 remain open, as do nested/cyclic/host coverage,
Darwin sanitizer7ba and evaluator791a. I claim no new source-bootstrap evidence.

My frozen affected gate passed (`/tmp/nanolang-mutable-runtime-full.log`):

| Gate | Methods | Seconds |
| --- | ---: | ---: |
| Mutable runtime checkpoint | 2 | 3.213 |
| Runtime package | 2 | 2.704 |
| String/boxed/packed core | 3 | 5.705 |
| Existing profiles | 1 | 0.401 |
| Existing string core | 3 | 2.871 |
| Managed operations | 45 | 61.245 |

I used `NMS_NATIVE_CLANG_FLAGS=--gcc-install-dir=/usr/lib/gcc/aarch64-linux-gnu/13`
with `make -j4 test-llvm-managed-strings test-verifier-profiles`.

I restacked onto main `66508fe4` with runtime/header/tests byte-identical to
reviewed `4a429e44`. Integrated head `0158eb61` passed both runtime methods in
2.799 seconds and the existing profile method in0.378 seconds
(`/tmp/nanolang-mutable-runtime-integrated.log`). Upstream range/native-label
changes remain included. This evidence commit changes no tested source.
