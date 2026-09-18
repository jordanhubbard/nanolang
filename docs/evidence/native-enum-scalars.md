# My native enum scalar evidence

I implement task29fc after contract `a9d3e27c` on main `e2318f47`.
Production `46f56279` adds raw ENUM_VAL as boxed tag9 with an ordinal and
NULL pointer. My root tracer visits only supported heap tags, so enum values
never become owner handles. My classifier preserves enum provenance through
locals, globals, branches, direct calls and returns; I check declared enum
return tags. ADD/SUB/MUL/DIV convert copied enum operands to integers before
existing numeric promotion. MOD/NEG keep exact rejection.

I retain VM compatibility for enum/int equality and ordering, distinct enum/float
tags, same-enum ordering's existing zero result, nonzero ordinal truthiness,
CAST_INT ordinal, CAST_FLOAT zero and CAST_STRING empty fallback. I print
`enum(ordinal)`. I make no new nominal identity or heap-layout claim.

Six focused GCC methods pass in 8.034 seconds, including generated
ASan/UBSan/LSan execution. Twenty Clang methods (six enum, seven numeric-union,
seven tagged arithmetic) pass in 10.605 seconds with the same sanitizers.
One added enum-zero/wrapping boundary method passes GCC in 0.212 seconds and
Clang in 0.285 seconds. I test both arithmetic operand orders and resulting
tags, zero and maximum ordinal, locals/globals/calls/non-self tail return,
branch joins, comparison/cast/display behavior, wrong enum return tags,
MOD/NEG invariant termination without sanitizer diagnostics and invalid
module publication preserving previous output.

I retain the initial new-code failures: the scalar truthiness guard lacked tag9,
and the new enum tail result was absent from the return-assignment whitelist.
I corrected those enum routes and kept the controls. Three initial negative
controls expected a different diagnostic phrase; I corrected them to the actual
invariant prefix and still require SIGABRT with no sanitizer finding.
Logs are `/tmp/nanolang-native-enum-focused.log`, `final-gcc.log`, `gcc.log`,
`clang.log`, and `boundaries-{gcc,clang}.log` under that common prefix.

My static audit records two separate existing dependencies: typed integer enum
coercions (taska77ca) and U8 non-self tail result loss (task9eb8). Neither is
silently repaired by this child. LLVM/Wasm enum task3762 remains dependent;
its existing profile refusal remains in place. Full66a6 and release scope stay
open. I do not execute retained compiler startup artifacts.

My full native gate passes 2,422 checks with zero failures and 1,139 shape
checks on unchanged production46f56279. I retain its complete log at
`/tmp/nanolang-native-enum-full.log`.
