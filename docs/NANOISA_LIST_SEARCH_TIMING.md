# My bounded substring-search timing diagnostic

I keep the original compiler source, complete shadow selection, ten-second
child deadline and120-second outer bound. My production pin is46b2faad1.
The old b33 observer measured generate_global_constants at2.204 seconds
inclusively; it does not establish which nested operation consumed that time.
The shadow generates two complete C products and performs five positional
searches through transp_str_index_of. That helper scans byte offsets with
str_substring; existing builtin str_index_of support is a separate producer
audit, not permission to replace the helper before measurement.

I reuse the reviewed5d762bdba per-shadow observer unchanged in its boundaries.
I add private cumulative counters to its bounded numeric output, selected only
for an actual user function whose registered name equals transp_str_index_of.
Both evaluator invocation implementations are instrumented after resolution
and arity/foreign handling, before parameter copies through return cleanup.
Argument-expression work and earlier dispatch lookup are excluded. Timings
are inclusive of nested evaluation and are not additive critical-path data.

I collect monotonic wall and process CPU nanoseconds, started/completed/active
counts, clock failures and overflow indication. Checked additions and time
conversion prevent wrap. I preserve errno around observations. The existing
4096-record bound and four write attempts stay unchanged. No source strings,
argument contents or pointers are emitted. Disabled observation contributes
no clock calls. Early process termination leaves a final unreported interval;
only deltas between completed markers support attribution. No marker proves
that all remaining work completed.

I first retain this source checkpoint for independent review, then rebuild
the observer-owning TUs against exact frozen46b provider maps and execute one
full-graph diagnostic. Observer overhead remains explicitly attributed. I do
not rerun old memory-faulting revisions or relax acceptance. Post-terminal
load snapshots cannot establish concurrent workload or infrastructure cause.
