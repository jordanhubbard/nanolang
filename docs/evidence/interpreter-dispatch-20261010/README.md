# Interpreter dispatch and live record receivers

I continue the compiler shadow timeout from bcdfbf1f4. I retain three temporary
instrumented compiler profiles of the same parser-import harness. Each profile
uses monotonic timing and subtracts completed nested instrumented time. Clock
cost and inlining changes affect these measurements; I use them for attribution,
not a production performance guarantee. My retained patches apply to bcdfbf1f4.

The env profile measures function lookup, symbol lookup, variable definition,
assignment and record construction, in that order. The eval profile measures
call dispatch, expression evaluation, statements, foreign calls and scoped
blocks. The fields profile measures source-visible lookup, qualified imported
literal lookup, definition, assignment and record construction. Exact counts
and totals are in profiles.json; raw logs retain every shadow boundary.

My field profile finds 1,454,218 source-visible lookups consuming 1.010974 of
1.668936 shadow seconds. Live record reads performed these static checks twice.
I now recognize an actual unlocated runtime record binding before static
resolution. I still evaluate the receiver, preserving tracing and fresh values;
static enum and imported-literal paths retain their existing checks. I do not
cache field values, change string ownership or broaden lexical visibility.

My call profile attributes 0.386345 of 1.729146 seconds to named call dispatch,
which contained 261 string comparisons over 258 exact names. I classify each
name once with an immutable sorted table, then preserve the existing numeric
branch order. Unknown names retain user/FFI dispatch. Argument evaluation and
builtin aliases remain in their original positions. Tests check table ordering,
exact spellings, prefix misses and unknown qualified names.

My full evaluator and environment checks pass, including the existing shadow
unit, a live record parameter shadowing an enum, retained string field results,
record replacement and the classifier controls. Environment checks pass 60
assertions and ten lexical-scope methods. Binary64 bit and arithmetic evaluator
checks pass. Twenty existing capture, mutable-record, scope, global and enum
methods pass; a new native/import-alias receiver method passes separately.
Seventeen global/enum/callback-array methods pass with nine compiler components
instrumented by ASan/UBSan and leak detection disabled. Linked components outside
that set remain uninstrumented.

My unchanged compiler-product test finishes in 198.038 seconds with failure
at native linking. The prior shadow blocker is corrected: all 1,107 selected
shadows complete with no assertions in 5.402708 summed seconds, inside the
unchanged ten-second whole-suite deadline. The driver advances to native C
compilation and linking, which reports 606 duplicate symbols. My input
inventories before/after match (`product-manifest.json`). I retain the original
explicit stage deadlines and original 180-second helper default.

The linker reports 478 duplicate definitions between nanoisa and file_product,
79 between compiler_support and file_source_inputs, 36 between nanoisa and
file_source_inputs, and 13 between file_product and the separately compiled
UTF-8 runtime. These modules bundle overlapping C support sources into native
objects. I preserve the complete terminal failure (`product-seed.stderr`) and
track native support linkage as the next implementation gap. I do not claim
this complete compiler-product route or final release qualification passed.
