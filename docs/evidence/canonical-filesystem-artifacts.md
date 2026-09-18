# My exact filesystem artifact coverage

I register the eight existing filesystem declarations with exact string
parameters and int/string/array<string> results. My native adapters retain the
owning library path, snapshot the module's borrowed scalar strings, and copy
listing strings before an explicit exclusive-array cleanup in the producer.
The ABI marker, function and cleanup must resolve to the same loaded image.
I do not admit arbitrary foreign signatures from a coarse array result tag.

At production `0f27b8c7`, fresh bootstrap passes both stages and installed
independence. My 11 focused methods pass in 9.060s, including selected shadows,
VM/native listing execution, all eight declared functions, sorted/filter/empty
results, retained snapshots across subsequent calls, three exact ABI refusals
with prior-output preservation, and existing artifact publication tests.
My native suite passes 2422 checks and shape constraints pass 1269 checks.
My listing module passes GCC and Clang O1 ASan/UBSan/leak checks, including
rejection of a shared array followed by successful exclusive cleanup.

Static independent review identified the inherited array adapter's missing
same-image check. I recorded task467623d before adding the guard; rebuilt tools
and all 11 focused methods pass in 8.835s afterward. I do not execute a hostile
library to establish this defensive guard.

Logs remain in `/tmp/nanolang-filesystem-bootstrap.log`,
`/tmp/nanolang-filesystem-final-focused.log`,
`/tmp/nanolang-filesystem-native-gates.log`,
`/tmp/nanolang-filesystem-gcc-sanitizer.log`,
`/tmp/nanolang-filesystem-clang-sanitizer.log`, and
`/tmp/nanolang-filesystem-provenance-gates.log`.

My first expanded gate is retained as
`/tmp/nanolang-filesystem-acceptance.log`: VM calls passed before the native
adapter existed, and both audio default-argument examples reached missing
`str_trim`. Those existing full-product examples remain required. Their
string-lowering dependency is task602; filesystem task2e5ed and provenance
task467623d do not establish full core-example or release acceptance.
