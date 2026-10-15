# My exact empty array constructor declarations

I add EFFECTS revision 3: the revision-2 element table, a count and ordered
(function, function-relative byte offset, element index) triples, then the existing
operation signatures. I retain up to 32,768 constructor sites. Each site must be
an instruction-boundary empty ARR_NEW or ARR_LITERAL with the matching tag;
function/code bounds, unique ordering and exact element indexes are checked.
I validate every binding, including unreachable sites and other functions, before
publishing an operation, type table or selected constructor query.

My affine analysis uses explicit bindings for empty arrays with otherwise ambiguous
nominal or nested types. Missing bindings preserve unambiguous-tag construction.
Both empty opcodes, distinct nominal result refusal, mismatched pushed elements,
nested arrays and resource-array handler/resumption transfers are checked. Failure
preserves instruction facts. These raw fixtures still refuse VM verification and
native emission: producer integration and matched runtime cleanup remain required.

My contract tests cover every truncated prefix, malformed counts and indexes,
operand-byte and non-constructor offsets, duplicate/out-of-order sites, mismatched
tags, nonempty literals, malformed prefixes, code bounds, an unrelated bad operation
row, zero sites, the exact capacity and one beyond it. I check unchanged query
outputs and binary/canonical-assembly preservation. The first mutation run used
another valid string index as an allegedly invalid key; I retain that failure and
correct the fixture to an out-of-range key. No product assertion was removed.

`make -f Makefile.gnu test-affine-bytecode test-ownership-contracts test-affine-state`
passes 2,240 ordinary and 3,249 allocation-fault bytecode checks, the ownership
contract harness and 563/634 state checks. The fault sweep now also covers the new
empty-constructor resource-effect fixture, with unchanged facts and recovery after
every injected analyzer allocation failure. ASan/UBSan/leak checks pass all 3,249
bytecode and 2,437 contract assertions. The analyzer, state, ownership reader,
verifier and harness are instrumented; other linked objects retain their existing
instrumentation. I retain exact sanitizer build commands and current input hashes.

Rebuilt VM/native effect cleanup tests pass all 26 fixtures, and the five shared
C-source effect methods pass. These are regressions, not collection runtime or a
fresh self-hosted compiler qualification. Destructive resource-element operations,
array-bearing resource payloads, both source producers and matched VM/native
collection execution remain open under #976.
