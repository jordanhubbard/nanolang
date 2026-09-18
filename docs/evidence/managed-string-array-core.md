# My shared string-array runtime foundation

I qualify task83a671 at frozen source `b75a27c313767c940e76a08c6180110528186bc7`,
based on main `25ca7ed8`. My contract is
[shared-table string arrays](../NANOISA_MANAGED_STRING_ARRAY_CORE.md).

I add explicit string/array descriptor kinds and transactional shared-table
publication. My string-only arrays retain each child edge, preserve mutation
identity across aliases and table relocation, return retained children, and
reclaim their buffers and children. Whole-context disposal frees every live
buffer once without recursively releasing the same child twice.

On Linux ARM64, I passed this frozen command:

```sh
NMS_NATIVE_CLANG_FLAGS=--gcc-install-dir=/usr/lib/gcc/aarch64-linux-gnu/13 \
make test-managed-string-core test-managed-runtime-package \
     test-llvm-literal-strings test-llvm-managed-strings test-verifier-profiles
```

I observed these completed gates in `/tmp/nanolang-string-array-full.log`:

| Gate | Result |
| --- | --- |
| New string-array core | 2 methods, 1.901s |
| Existing string core | 3 methods, 2.497s |
| Generated runtime package | 2 methods, 1.850s |
| Scalar globals | 11 methods, 8.262s |
| Literal strings | 9 methods, 32.772s |
| Managed strings | 39 methods, 43.876s |
| Shared verifier profiles | 1 method, 0.342s |

My new core gate compiles actual native LLVM IR, verifies it with `opt`, and
executes production and testing builds under Clang ASan/UBSan. It separately
compiles and verifies wasm32 IR, links without imports, and executes with Node
(two fresh instances, three rounds each) and Wasmtime. I test alias mutation,
duplicate children, retained gets surviving array release, missing unsigned
indices, descriptor/buffer growth, byte-exact NUL/high-byte children, allocation
rollback, valid wrong-kind refusals, independent contexts and terminal disposal.
The real 1MiB Wasm memory bound forces a finite append allocation refusal while
preserving length and child owners. One hundred release/recreate rounds retain
stable memory-page usage and zero live allocator objects after disposal.
Production builds omit testing hooks. Existing package checks regenerate both
ABIs and verify source/header/generator hashes.

I do not admit array opcodes, array frame/global/call ownership, or STR_SPLIT
in this change. I do not claim general cycles, nested arrays, arbitrary element
shapes or nominal fields. Parent488 and parent51da remain open, as do the
separate Darwin managed sanitizer7ba and historical evaluator791a incidents.
I changed no compiler source and claim no fresh compiler bootstrap here.

During static next-step inspection I recorded taskaeff9 for OP_ARR_NEW's
unchecked allocation publication. I did not execute a failing input or alter
the VM in this foundation. That defensive prerequisite precedes array opcode
parity work.

I restacked onto main `5cc34548` after PR685. The prior published head
`ba104b35` and integrated implementation `2a736e40` have identical managed
core/header, array fixtures/harness and Makefile bytes. The inherited delta is
PR685's source-borrow loop-exit work and its documentation; it does not change
this runtime or its generated package inputs. My completed frozen gates remain
applicable; `git diff --check` passes without an additional runtime rebuild.
