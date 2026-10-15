# C source selected-union checkpoint

I register concrete resource-bearing union layouts in dependency order, copy
their type arguments for stable lifetime, retain transitive ownership, and emit
explicit constructor, local, call and result transfers. Selected unguarded
matches consume one union token. Resource payloads require explicit destructive
patterns; ordinary and empty selected arms discharge their union shell while
retaining checked ordinary payload fields. I preserve source field evaluation
order and constructor declaration order. Block-valued matches retain exact
returned union identity and outer-owner joins. Guarded resource matches retain
the existing source refusal.

I reuse all 36 original generic/nongeneric selected-owner acceptance tests
without changing their sources or assertions. Accepted NanoVirt output runs
through NanoVM and strict sanitized native C; source refusals preserve prior
output. The final Make gate exits 2: 35 methods pass and one remains failed.
`test_scrutinee_call_evaluates_once` requires a mutable global counter. My
owned source profile still refuses that declaration. I retain this required
case and add the entire target to test-units; I do not mark source parity done.
Checked global initialization/types through helper reads/writes are next,
followed by the self-hosted source route and complete installed qualification.

The first build finds an unused wrapper after replacing scalar-only payload
queries; I remove it. Initial source runs expose missing nongeneric type-name
lookup and variant brace-literal normalization, then block-valued match return
lowering. The fourth and final Make runs leave only the global-counter failure.
All logs, source fixtures and generated text are retained per run. Binary
modules/executables remain under the original /private/tmp paths and are
represented by SHA256 inventories here, not committed binary artifacts.

Adjacent gates exit zero: all 90 NanoVirt checks and all three contract-buffer
allocation refusals with same-process recovery. Successful source cases use
ASan/UBSan and leak detection in native execution. This does not claim that the
C compiler process itself was leak-qualified, or qualify the self-hosted route.
