# My separate managed graph provenance query

I implement child078055bd at production checkpoint `477b5e1d` after private
storage/collector PR724. My new graph report wraps existing array-origin facts
and adds child-origin sets plus explicit unknown flags. The existing leaf report
layout, API and verifier consumer remain unchanged; no runtime/profile/emitter
admission changes here.

Boxed child summaries weakly join complete tags/origins/unknown through writes,
literals, GET/POP, shallow slices, CFG/local joins, direct calls/recursive
summaries and globals/reentry. Fresh slice outer origins preserve their copied
inner identities. Packed storage retains its finite coercion matrix. Every
origin/tag/unknown bit only grows, including repeated allocations at a site;
the bounded finite domains therefore converge without guessing favorable joins.
The graph is an overapproximation, not a runtime root/reachability certificate.

My27 focused methods pass in3.437 seconds:18 existing leaf methods and9 graph
methods. Ordinary and Clang ASan/UBSan probes compare reports and statuses.
The graph probe repeats successful queries for exact report stability and
compares leaf reports/diagnostics, profile results and canonical module text
before and after graph analysis. Failed allocation leaves caller output pointers
untouched. I test every private allocation stage, authoritative-origin refusal,
origin caps, nested GET/POP mutation, shallow outer identity, duplicate edges,
self/mutual cycles, mixed packed/boxed alternatives, recursive/repeated sites,
global reentry and loop copies. Accepted ordinary fixtures execute on NanoVM;
both managed backends still refuse nested programs without replacing old output.

My first run omitted the conservative initial-VOID path in one global fixture's
expected child tags. I preserve `/tmp/nanolang-graph-origins-focused.log` and
correct that expectation without narrowing analysis. A subsequent initializer/
loop control exposed a real transfer issue in the new graph mode: slice output
received an ARRAY tag before its global source acquired any array origin, which
poisoned later joins as unknown. I preserve
`/tmp/nanolang-graph-origins-corrected.log` and keep that graph transfer at bottom
when no successful array receiver is possible. Later fixed-point visits can add
real origins. Genuine unknown or ARRAY-without-origin paths remain unresolved;
leaf-mode behavior is unchanged. The corrected gate is
`/tmp/nanolang-graph-origins-convergence.log`.

Generated owner tracking, allocation-pressure/repeated-entry collection safe
points and final nested admission remain child4070da262. Full aggregate488,
managed51da, nominal/map/callable graphs, platform and release requirements stay
open. This query is neither lifetime proof nor executable acceptance.

My frozen adjacent gate passes all61 managed instruction methods in81.164
seconds and shared-profile/runtime/package/core targets, including the prior
explicit graph collector controls. I retain
`/tmp/nanolang-graph-origins-adjacent.log`. Root independently reviewed the
complete production/header delta and surrounding transfers/final checks without
a scoped blocker. I make no new source-bootstrap or Darwin acceptance claim.

I restack onto main `a0a8fbcf` with independent reconstruction PR726. My analysis
production/header and probe/tests remain byte-identical to the frozen reviewed
checkpoint; the Makefile only gains main's separate reconstruction target.
I retain the completed gates without repeating unaffected tests.
