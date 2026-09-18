# My managed byte-string predicate contract

I implement required full LLVM/Wasm coverage child
`task_b9a83a6b239b4c858ca6324d3066ac33` after the managed runtime and portable
conversion prerequisites. I add only STR_CONTAINS, STR_STARTS_WITH and
STR_ENDS_WITH to my closed managed-string profile. My scalar and literal-only
profile APIs retain their previous decisions. Other string/heap/host operations
remain separate required work, not exclusions from full release scope.

My reference is `vmstring_find` in `src/nanovm/heap.c` and the three opcode
handlers in `src/nanovm/vm.c`. I compare stored bytes and lengths, including
embedded NUL and non-ASCII bytes. Empty needle/prefix/suffix is true. A longer
needle/affix is false. CONTAINS tests each fitting byte offset; STARTS_WITH tests
zero and ENDS_WITH tests the checked length difference. I perform no Unicode
normalization, locale conversion or C-string truncation.

My private runtime helper borrows two validated handles, allocates nothing,
and publishes a bool only after both views and the operation selector validate.
It changes no reference count. Comparison loops use subtraction before bounds
and never increment a uint32 position beyond its admitted last offset. Empty
needle handling precedes iteration. Equal handles remain ordinary aliases.

LLVM lowering checks both value tags, obtains a scalar bool from the private
module adapter, and lets existing FrameOutput tracking release both consumed
operands exactly once. A wrong tag or helper failure follows the current status
latch and frame cleanup before exported failure; it never calls an internal
trap while owners remain live. Predicates return no string owner. The module
adapter ABI uses scalar integer handles/selector/result, so native and import-
free Wasm packaging need no pointer-layout extension or undeclared imports.

I require paired VM/native LLVM/Node/Wasmtime ordinary byte controls, literal
and computed strings, empty/long affixes, embedded NUL/high bytes, overlapping
searches, same-handle aliases, calls/globals/reentry and final reclamation.
Direct runtime tests disable new allocation while exercising borrowed views.
Dynamic wrong-tag controls test TYPE status, operand/frame cleanup and preserved
global aliases. Shared profile and unsupported-op previous-output controls
must remain green. I test the actual packaged native/Wasm runtime, not only a
mock of the helper.

This child does not close aggregate/cycle task488, host-linkage task2d2, Darwin
managed-sanitizer task7ba, historical evaluator task791a or the full target
coverage requirement. It adds no new bytecode encoding or frontend syntax.
