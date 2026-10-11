# Scalar array fields in records

I propagate scalar array writes through shared handles in my C AOT shape graph,
using the existing directed write mechanism for record arrays. The exact
destination element constrains aliases; I do not equate optional source values
with destination storage. My retained self-hosted List<int> record/union
reproducer previously failed at AGG_PACK field 0 despite later integer writes.

I also retain concrete constructor evidence for arrays that remain empty.
After graph inference, an unresolved ARR_NEW TAG_INT constructor supplies its
integer default. I do not infer integer storage from an unresolved parameter.
Consumers have already had their chance to determine the constructor shape.

My six bytecode methods cover the original source module, forwarded scalar
writes, mutation through record aliases, an empty integer-array field,
incompatible writes and unresolved parameters. I execute positive modules in
NanoVM and sanitized C AOT. My initial attempted string-specialization fixture
used ARR_NEW TAG_INT and failed in NanoVM before translation; it was not a valid
parity fixture. I retain that failed test-development log, replace it with
the unresolved-parameter control, and keep explicit string/float/bool controls.

The first broad gate selected Apple Clang through Makefile CC despite an
environment override, and its tests failed because that sanitizer does not
support detect_leaks. I retain the failure and rerun with command-line
CC=/opt/homebrew/opt/llvm/bin/clang and NANO_NATIVE_TEST_CC set to the same path.

The scalar-write correction passes 2,443 structured C tests and the gate's
prerequisites. The final constructor-evidence correction has its own gate log.

I retranslate the pinned compiler module from the preceding unchanged-source
audit, link its native compiler, and run it against the 16-method nominal-order
suite, the empty-list source control, and 16 payload/resource/layout methods.
All eight stages pass; module and changed-source hashes remain unchanged.
This tests the changed translator against an existing compiler module, not a
new frontend bootstrap or exact-candidate Linux/Darwin release qualification.
The product-manifest.json records the pin, commands, results and timings.

My final broad gate passes its prerequisites, all six new alias methods and
2,443 structured C tests. The pinned-compiler validation takes 29.185 seconds
across its eight recorded stages; all source/module hashes remain unchanged.
