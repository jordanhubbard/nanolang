# My representation fact joins

I port the complete classifier fact-merge block under #1142, after my
storage-kind, carrier-mask and C ABI type selections. These are value and
owned-vector operations. I grant no module, graph, payload, ownership or
executable admission through a merged kind.

My policies retain exact field merging; present scalar/optional parameter
widening; checked record/callable and primitive-container call boundaries;
finite record-result unions; exact record-array globals; conflicting global
field refusal; sticky map payload-hint removal; and exact global-map pairs.
All physical uint8 kinds remain distinct, including unknown2 and tagged8.

My state retains changed, failed and the first error independently. A prior
failure does not change an individual helper's later return or mutation: the
C helpers can still accept or merge later facts while preserving that failure.
Callers must reject failed analysis before admission. Vector merges preserve
partial changes before the first refusing field and stop there. I copy caller
input vectors before changing an owned result; physical C pointer identity,
function-scoped allocation/free and allocator-fault equivalence are not
claims of this value API.

I reproduce diagnostics through logical locations for effect, parameter,
parameter-field, result-field and array-result cells. Those descriptors name
the same positions as the C contiguous fact arena; they are not C addresses
or table/index admission. Global diagnostics retain slot/field and source
function/offset identity. Checked entries reject out-of-domain kinds, policies
or locations instead of truncating them or performing invalid C pointer reads.

I preserve context widths from the C module builder: at least one record/local
slot, record width at most65535 and local width at most66559 (1024 locals plus
65535 captures). Source function/offset are uint32 values. Logical ordinals
are nonnegative addressable integers; callers own their table bounds.

My permanent gate compares actual unchanged C with ordinary/sanitized
references and all five NanoLang routes. Every kind pair per policy, full
map-pair predicate branch domains, every location category, initial/sticky
state, sequential/vector/partial-failure boundaries, detached copies and
malformed/prior-output neighbors supplement independent expectations. I keep
600/60/default10 and immutable source/tool/qualified-Stage2/index/six-host
guards. My six-method component gate passes across seven routes; my
[evidence and independent reader](evidence/c-repairs-linux-20261126/representation-facts-20261127/README.md)
retain all source inputs, reports, products and the separate early prototype.
This qualifies the component only. Whole module
representation/emission and all later qualification/release gates remain.
