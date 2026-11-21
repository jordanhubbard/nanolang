# My nominal service metadata evidence

I implement descriptive service plans at `d147c1d72` and unified queries at
`1ffd6ed553ff0d3820bf806993369e559faa0ac6`, under #1142 and draft PR #1144.
My NanoLang implementation calls no C planner or catalog getter at runtime.
I retain the C implementation as an independent test oracle. This checkpoint
changes no installed translator, executable selection or host authority.

## My contract

I describe File, TCP and WebSocket v2 plans and mixed/repeated v3 plans.
I check exact import names/signatures, per-kind layout counts, nominal names,
prior-only member edges, catalog-local instance identity and ownership-v1
flags/descriptors before returning rows. I preserve unknown layouts and their
actual wire kinds, distinguishing tuple1 from enum3. Invalid plans return no
partial rows.

I compare complete layout, type, source and import queries, including missing
indices. Unified private queries preserve `instance*9+type` and
`instance*5+method` identities, catalog counts, missing slots and TCP endpoint
classification. Original C oracle getters check unchanged output on failure.
These logical identities do not admit code flow or create host handles.

My catalog generator reads canonical C views at development time and emits
immutable NanoLang data. `make check-nvm2c-service-catalog` checks regeneration;
the differential driver reconstructs the complete catalog from pure getters.
I retain both catalog reports and the generator source.

## My qualification

I compare 7,907 synthetic carriers and eight actual producer carriers per host:
7,915 complete reports across four profiles and five routes on Linux ARM64
and Darwin ARM64. The synthetic corpus includes 7,139 explicit status checks.
The routes are C-seed native and C/NanoLang producers through verified NanoVM
and ASan/UBSan-instrumented generated C. The actual four C nominal planners,
raw codecs, unified views, cursor and immutable catalogs are also instrumented
in a separate full-corpus differential method. The linked AOT runtime and
ordinary extraction providers retain their separate scope.

I retain every truncation of the selected service/layout/ownership packets,
seeded mutations, all256 import result/parameter tags, malformed names,
nominal cross-instance edges, unknown tuple/enum rows, ownership versions,
flags/local tags/modes, unsigned indices, import and callback precedence,
interleaved/reversed layouts/imports, all64 instances and65,536 layouts.
Complete C reports compare values and queries, not only acceptance statuses.

I independently regenerate identical decompressed corpora on both hosts and
match523 source hashes against each other and the qualification revision.
Compressed bytes differ between host gzip encoders; I compare decompressed
content and record each archive identity. I verify source/module/carrier
hashes inside both actual-product archives. Each contains eight modules and
32 exact files: File/TCP/WebSocket/mixed sources through both real producers.
Selected shadows run during production. These metadata fixtures do not run
network lifecycles, and metadata success does not establish those lifecycles.

Original File, TCP, WebSocket and multi-instance metadata/allocation controls
pass on both hosts. I retain their complete commands/results, mandatory shadow
inventory (38 named functions) and identifier checks. All original assertions
remain intact. Allocation-denial tests describe the C providers, not NanoLang
heap failure behavior. I reuse the genuine development NanoLang compiler
image from the paired ownership receipt; its hash is checked against that
receipt. I configure the genuine Linux assembler capture helper explicitly
for this image outside bin. No stage alias or installed compiler is invented.

## My retained failures

I retain the original incorrect record/variant enum shadow comparison,
strict misleading-indentation C refusal, missing unified-view provider link,
tuple/enum reader/fixture mismatch and ordinary extractor refusal of a private
WebSocket product. I correct the enum/guards/link inputs and select the existing
exact WebSocket metadata transport adapter. I weaken no warning, assertion,
shadow selection or execution refusal.

My first final Linux attempt was interrupted with exit143 when its execution
runtime was replaced. Its sanitizer method passed, but the suite had no
terminal result. I retain it separately and require a new complete run at the
unchanged source pin. Earlier base runs do not replace the unified-query gate.
The resumed Linux run passes both methods in393.270 seconds with exit0;
Darwin passes both in115.684 seconds. These are the acceptance logs. Their full
terminal results and source/tool identities are retained in my input archive.

## My boundary

Normal immutable value carriers model valid C storage. Original C tests retain
pointer/null/overlap and allocation controls beyond that model. Instrumented
generated fixtures link an ordinary AOT runtime; I claim neither a fully
instrumented runtime nor complete evaluator LeakSanitizer qualification.

I complete only the nominal metadata-plan roadmap row. Legacy File v1 metadata,
whole-module service/private profile admission, instruction verification,
representation analysis, NanoLang C emission, canonical integration,
installed/no-C-seed products, raw fixed points and the rest of #976 remain
required. My release, issue and draft PR remain open.

`manifest.json` binds the archived inputs, exact sources, full reports,
producer products, commands, original failures and terminal evidence by SHA256.
