# I own my checker-created metadata

I continue task_e69bf88f5cbb49bd933b5ad7e25ae3e6 under00c47 after canonical
5485b5e9d (union882). Before code I record the historical595-byte/16-allocation
report at /tmp/nanolang-metadata-sanitizer.log. Its208-byte import tracker is
already freed;387 is historical subtraction, not a current measurement.
Function names/parameter containers/names, one shallow callback TypeInfo wrapper,
and empty array headers/data have concrete allocation stacks. I never replay the
old binary. The current temporary harness was subsequently narrowed and omits
the parsed case; it cannot stand in for that original acceptance.

## My bounded ownership change

I add an environment-owned allocation registry for explicitly checker-created
storage. It owns individual malloc-compatible allocations until environment
teardown, independently of Function/Symbol slot replacement or truncation.
I register allocations immediately; registry-node allocation failure exits using
my existing environment allocation policy. This is not an allocation-failure
recovery claim. The registry does not infer ownership from type tags, names,
non-NULL pointers or mutable symbol fields. It is not serialized metadata.

Both checker entry paths register their allocated function names, copied parameter
containers/names/nominal strings, result nominal names and module names. AST bodies,
parameter/signature/TypeInfo subgraphs and manual/builtin registrations retain
existing borrowing. Shallow synthetic callback TypeInfo owns only its wrapper,
never its AST fn_sig. Empty checker array placeholders retain exact current
creation/element inference semantics; their header and data allocations enter the
registry individually. I do not free arbitrary runtime array aliases.

I drain the registry once after normal environment metadata cleanup. Registry
entries contain allocation identity, not pointers into movable symbol arrays.
Truncation and redefinition cannot lose ownership or double-register a borrowed
pointer. Existing interpreter value and source admission semantics stay intact.

## My review and acceptance order

1. I commit this contract and roadmap before production.
2. I send the complete source and focused fixture checkpoint for independent
   review before any build or execution.
3. My fresh focused harness explicitly invokes unchanged
   test_function_metadata_lifetime from tests/test_module_metadata.c. I also
   exercise both checker APIs, AST-first/environment-first lifetimes, borrowed
   manual signatures, repeated names, truncated slots and array inference.
4. I rebuild current providers with ordinary flags and separately with O0
   ASan/UBSan/LSan, detect_leaks=1 and no suppressions. I retain first terminals,
   source/tool inventories and exact instrumentation. Existing complete module
   metadata and environment-scoping targets remain required adjacency.
5. I seal and publish through review. The child and parent remain open until
   actual qualified merge and explicit criterion reconciliation. This does not
   claim all compiler allocations, runtime arrays or callback ABI are repaired.

## My source checkpoint

I add one opaque list pointer to Environment and one explicit transfer helper.
Every entry holds one malloc-compatible allocation and its next pointer; entries
are freed after existing symbol, nominal, namespace and module cleanup. No entry
consults AST or mutable symbol metadata during destruction. Callers register only
fresh allocations and never register a block twice. NULL follows existing
allocation behavior; allocation-failure recovery is not added.

I preserve the array placeholder header/data and callback wrapper bytes. Both
checker paths register exactly the fields described above; builtin registration,
interpreter value destruction, Function ABI and metadata serialization do not
change. The focused target includes the original parsed lifecycle function
verbatim, then eight checker-path/destruction-order/slot-replacement combinations
and stack-owned borrowed signatures plus a caller-owned runtime array control.
The parsed body checks array element inference through at. No gates have run.
