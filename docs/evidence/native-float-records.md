# My float record transport

I store binary64 field bits in the existing 64-bit aggregate cells. Packing
copies the representation with `memcpy`; projection copies it back into a
double. Nested record copies and record arrays retain those cells. Tagged
scalar fields retain the float tag and bits. I add float to the existing
exact nominal scalar-field agreement, without accepting conflicting layouts.
My canonical emitter now accepts scalar float fields in supported records;
float arrays remain outside that emitter subset.

My source fixtures run through both the C-seed and canonical NanoISA emitters,
then verification, VM execution, translation, strict C compilation and native
execution. They cover nested copies, replacement of a mutable record binding,
field projection across calls, mixed scalar fields, record arrays, and float
globals packed into records. A separate assembled module carries signed zero,
NaN and infinities through nested aggregate cells and a record array. I check
runtime tags, signed-zero/infinity output and NaN comparison behavior.

The first fixture incorrectly used dotted `set` syntax. Both parsers refused
it; I retain `/tmp/nanolang-float-records-tests.log`. The corrected fixture
uses supported whole-binding replacement and retains assertions that the
original nested value is unchanged. Three methods pass with GCC sanitizers
and Clang sanitizers, with Linux leak detection enabled. The first Clang
invocation stopped on its host GCC-installation selection warning before
compiling generated code. Selecting the installed GCC 13 toolchain explicitly
retains `-Werror` and all sanitizer checks; the passing log is
`/tmp/nanolang-float-records-clang-gcc13.log`.

This is scalar value transport, not reference-field or ownership execution
admission. My reference and full-release gates remain separate.

At integrated code `802d63e7`, my final core gate passes 2,418 native checks,
1,092 shape checks, 86 comparisons and 85 paired methods (110.190 seconds).
The final focused gate passes five float-record/global-array methods in
1.966 seconds; three float-record methods also pass Clang sanitizers in
1.489 seconds. Logs are `/tmp/nanolang-float-records-final3-gates.log`,
`/tmp/nanolang-float-records-final-focused.log` and
`/tmp/nanolang-float-records-final-clang.log`.

I replaced the old float-projection refusal with positive execution through
nested records and global storage in both function orders. Extending that
fixture to a nested record-array field projection exposed an independent
optional/record shape refusal, retained in
`/tmp/nanolang-float-records-integrated-gates.log` under
`task_65d164a9fb204ff7872002f53e708bdf`. My current record-array extension
checks its length after global reload; existing dedicated global-array cases
check ordinary field reads, aliases and collection. I do not claim the
unimplemented deeper shape path from those passing cases.
