# My unsigned-byte scalar contract

I track this work under task_ad1c498801b34aa38d82f58a588f1d5a.

My ISA declares TAG_U8 as an unsigned byte and PUSH_U8 carries an unsigned
one-byte immediate. My VM already preserves that tag, compares equal bytes by
value, converts them to int without sign extension, and treats zero as false.
My source-language arithmetic table does not introduce a separate byte
arithmetic overload. I do not add one here.

I interpret same-U8 ordering as unsigned numeric ordering over 0 through 255.
CAST_FLOAT produces the exact binary64 representation of that value, matching
the existing numeric CAST_INT conversion. The current VM omissions (default
zero float and default equal ordering) do not implement that numeric contract.
I repair those two omissions before backend admission.

I then admit tagged byte values within the closed scalar backend profile:
constants, locals, direct call parameters/results, supported joins, TYPE_CHECK,
CAST_INT/FLOAT/BOOL and explicit CAST_INT followed by typed I64 comparisons.
My VM and C backend also implement same-U8 generic comparisons; LLVM/Wasm
continue to refuse generic comparison opcodes until task01d or a separate
static operand-proof prerequisite establishes their complete admitted contract. Typed I64/F64/BOOL operations keep
their exact tag checks. I do not add a CAST_U8 instruction, byte arithmetic,
heap or host byte transport, or a byte executable-entry ABI.

Mixed-tag arithmetic and comparison policy remains task_01d3e7ca2b6a47b0bef9504f2fd04006.
I preserve current VM mixed-tag observations and explicitly refuse profiles I
cannot match; I do not infer a numeric promotion rule from tag ordering.

My acceptance requires ordinary boundary values (0, 1, 127, 128, 254, 255),
all same-byte order relations, exact conversion/result tags, call/branch
transport, and previous-output preservation for unsupported profiles. I retain
separate VM, native C, LLVM and Wasm evidence and do not claim full backend
coverage from this scalar slice.

My display control exposed a separate CAST_STRING(U8) empty-string fallback
in the VM (`/tmp/nanolang-u8-common.log`, `/tmp/nanolang-u8-native-sanitized.log`).
I record task_d08968be827a4d26942123825229ac8e rather than treating that fallback
as an intended numeric formatting rule. This slice preserves it and tests
ordinary numeric PRINTLN separately.

## My measured acceptance

My source base is `765ec87bb02baf390ab8e331363338e687d3b09a`.
I recorded the contract at `79b44419`, implemented scalar transport at
`40dc6221`, and retained the separate string boundary plus checked-call controls
at `3d0bab7a`.

- I pass 273,392 VM checks and 32 value methods, including every byte-to-float
  conversion and all 65,536 same-byte ordered pairs. Evidence:
  `/tmp/nanolang-u8-vm-r1.log`.
- I pass 2,414 native translator checks, 1,092 shape checks, 1,051 owned-runtime
  checks and 32 owned allocation checks. Evidence:
  `/tmp/nanolang-u8-native-owned.log`.
- I pass all 45 shared scalar methods in 73.751 seconds, including seven new
  U8 methods. Ordinary positives execute the same module in VM, C, LLVM
  interpreter, optimized LLVM, linked LLVM, Wasmtime and import-free Node.
  Evidence: `/tmp/nanolang-u8-common-final.log`.
- My seven U8 methods pass in 21.990 seconds with generated native C
  instrumented by ASan/UBSan and leak detection, and in 17.946 seconds with
  strict Clang. Evidence: `/tmp/nanolang-u8-native-sanitized-final.log` and
  `/tmp/nanolang-u8-clang.log`. These runs do not instrument every VM or LLVM
  component and do not establish Darwin execution.

I check byte identity across calls, local stores, branch joins and implicit
returns. A declared-byte argument retains its actual tag for an explicit
TYPE_CHECK entry guard; parameter declarations alone are not a new implicit
VM call check. Wrong return tags fail in VM/LLVM/Wasm and cause a checked C
shape refusal that preserves previous output. A byte cannot satisfy a typed
I64 operation without an explicit numeric conversion.

I retain earlier failed development logs: two initial test expectations
(disallowed literal typed operands are caught by assembly, and the diagnostic
says “do not support”), missing C tagged-runtime inclusion corrected before
publication, and the separate U8 string conversion observation. I reran the
corrected focused gates and final shared gate; I did not rerun a full compiler
build or change frozen acceptance products.
