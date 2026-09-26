# My borrowed record-name lifetime correction

Hosted run `35997348910` at compiler-source `d56d15ff6` fails both unchanged
record projection accessors on Linux x64, Linux ARM64 and coverage. Its
units-02 worker establishes heap-use-after-free in `emit_formatted`, after
recursive operand checking frees the record name's AST type information.
I reproduce invalid generated C on Linux ARM64; Darwin's ordinary allocator
happens to pass the same baseline, which does not establish memory safety.

I emit a scoped typedef for the exact nominal element before descending into
ordered operands. The final getter/setter uses that typedef after recursion.
I preserve source-order single evaluation and all original test assertions.
I do not retain a borrowed type-name pointer across operand checking.

The original two-method projection suite passes after rebuilding the Linux
C seed. On Darwin, the projection and seven setter methods pass ordinarily
and with generated-product ASan/UBSan/leak/use-after-return instrumentation.
The combined 15-method adjacency run retains four failures in two separate
declared-array-push native-stage methods: ambiguous indirect scalar results
and unresolved stored callbacks. Their C-seed subcases pass. I track those
remaining failures as a separate required contract, not as a passing suite.

I also build a fresh complete C seed into private object/binary directories
with Homebrew Clang ASan/UBSan and run the unchanged projection suite through
that compiler with leak/UAR checks and instrumented generated products. Both
methods pass in 5.775 seconds. The first private invocation failed because
its relocated compiler inferred missing runtime-source paths; I retain that
setup terminal, add explicit source/module/stdlib symlinks to the actual
checkout, and run into a fresh log without weakening checks.

Hosted Darwin separately reaches three native-module-linking refusals for
`fixture_value`; that raw log is retained here for the distinct tracked task.
This correction does not claim to resolve callback binding or module linkage.
Final hosted qualification remains required. Prior fixed-point evidence stays
pinned to its recorded compiler-source revision.
