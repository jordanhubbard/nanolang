# My checked VM array creation boundary

I implement taskaeff9da2749041fea3aa15033f56fc80 before managed array opcode
admission. Static inspection at main5cc34548 shows OP_ARR_NEW publishing the
result of vm_array_new without checking allocation failure. My heap constructor
already returns NULL and releases partially allocated storage when either the
array descriptor or element buffer allocation fails.

I check that result before value publication and return VM_ERR_MEMORY through
my existing terminal error path. Successful construction retains the existing
array tag, element tag, empty length and capacity behavior. I do not change
array element coercion, shape admission, indexing, or managed LLVM/Wasm profiles.

I first add the guard, then test fresh ordinary array creation with deterministic
allocator refusal at the descriptor and backing-buffer boundaries. I require
an unchanged output, empty invocation stack/frames, unchanged heap counters,
and successful subsequent creation/release in the same VM. I cover boxed string
and unboxed integer storage under the focused sanitizer gate and full VM suite.
I do not execute old failed artifacts or reproduce an unchecked failure.
