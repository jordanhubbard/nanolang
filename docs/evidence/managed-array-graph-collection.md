# My explicit managed graph collection checkpoint

I implement the reviewed graph contract after private edge checkpoint7a92e69f.
Production `ef665a95` adds `nms_collect` without changing any opcode, profile or
shape transfer. I retain native/Wasm tagged-value and numeric status ABIs.

My collector first allocates checked trial-count, mark and queue scratch. It
validates descriptor kinds/storage accounting and every boxed child before
subtracting internal edges from trial counts. Positive trial counts identify
external owners. Iterative marking queues each live slot once. Only after that
complete preflight do I remove dead-to-live edges and free the unmarked set;
I never recursively release edges within the dead set. A live owner remains
valid, and collection frees all scratch before returning.

My first collector target passes in2.440 seconds with production and testing
builds under native Clang ASan/UBSan and import-free Node/Wasmtime. It checks
rooted versus unreachable self/mutual cycles, duplicate edges, shared surviving
strings, retained API roots,4,096-node cyclic and acyclic graphs,300 repeated
collection/reuse rounds, independent contexts, terminal disposal and every
scratch-allocation failure. Failure controls retain exact objects/bytes/child
references/free-list head and allow later successful collection. The earlier
iterative-release controls disable allocation during chain teardown.

I add a fresh ordinary VM nested-identity fixture: shallow outer copies share
nested children, self-edges remain identity-preserving, and alias mutation is
visible. Both managed backends still refuse that module without replacing old
outputs. This qualifies private graph storage and explicit collection only.
Current emitted leaf programs cannot create these graphs. Origin/child
provenance, generated root handling and collection safe points under pressure
and repeated entry remain required before nested opcode admission.

I retain `/tmp/nanolang-graph-collector-focused.log` and the frozen full gate in
`/tmp/nanolang-graph-collector-full.log`. Full aggregate488/managed51da, nominal/
map/callable graphs, platform acceptance and release requirements remain open.
No new source-bootstrap or Darwin acceptance is claimed.

My frozen complete gate passes all61 managed instruction methods in80.798
seconds,18 shape methods in2.435 seconds, graph-core plus ordinary VM/refusal
controls in2.631 seconds, and shared-profile/runtime/package/core targets. The
production/testing graph builds run independently of instruction admission.
