# Owned WebSocket catalog query

I implement the exact immutable query for my [WebSocket service
contract](../../NSI_WEBSOCKET_CATALOG.md) under issue #990. Four methods and seven
nominal types describe owner acquisition, exclusive borrowing, consuming close,
counted message strings and explicit timeout parameters. Ordered network and DNS
capability declarations remain separate. Descriptors grant no authority.

I extend the shared catalog matcher from one capability pair to an exact ordered
capability list. Existing File and TCP documents retain their original single
capability and unchanged exact matching behavior.

## Evidence on Darwin

Clang, GCC 16 and LLVM ASan/UBSan/leak builds each pass 1431 instrumented and 983
ordinary linked checks. I mutate every method/parameter/type/member descriptor
field, capabilities (including duplicate and reordered rows), counts and null
arrays; invalid plans allocate nothing and preserve the prior output. Allocation
failure preserves output. Queries remain valid after freeing the source document,
retain exact owner outcomes, rights and timeout domains, and reject invalid/null
indices. Integer substitutions for URL/message strings are refused.

My Clang adjacency run also passes the unchanged File query checks (1319
instrumented, 905 linked) and TCP checks (1647 instrumented, 1136 linked). I retain
the initial harness failure: I referenced a nonexistent legacy WebSocket schema.
I corrected that incompatible-contract control to use the existing legacy net
schema, preserving the other checks. This was a test-fixture error, not a catalog
implementation failure.

`clang-and-adjacent.log` and `gcc-and-sanitizers.log` retain exact commands/results;
`verify.sh` repeats the latter builds. `hashes.json` identifies current source,
fixture and final GCC/sanitizer binaries.

## Remaining boundary

These are descriptor/query tests, not verified service execution. I have not
registered this catalog for source admission. String-bearing nominal metadata,
owned WebSocket values, exact runtime status semantics, per-instance grants,
VM/native dispatch, paired source lowering and installed Linux/Darwin execution
remain required. Legacy integer wrapper tests do not close these obligations.
