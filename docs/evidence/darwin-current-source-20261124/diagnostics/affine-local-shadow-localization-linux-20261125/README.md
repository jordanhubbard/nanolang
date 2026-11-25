# My local shadow projection diagnosis and collector controls

I retain the complete failure under #1259 and the separately published
[five-method terminal](../affine-foundation-shadow-failure-linux-20261125/README.md).
The original traced command exits1 in5.901s under600/default10, publishes no
module and retains449371530 stderr bytes. `local_info` returns Query through
`no_value`; the caller projects its nested Slot and passes that record to
BOOL_NOT instead of reading Query.valid. I do not modify a C product file.

My three-file reduction defines Decoy with valid at field1, then Query with
valid at field0. An imported direct-call `.valid` projection fails in the
C-producer NanoVirt route. An explicit `query.Query` local binding passes.
Both source variants pass C-seed native compile/run and genuine original7b9
NanoLang Stage2 compile/NanoVM execution: eight successful control commands.
The static C source retains the qualified call-result inference and first
matching field fallback in `src/nanovirt/codegen.c`. That identifies a repair
area, without proving that every affected source form has one cause.

My isolated two-binding candidate passes the original local driver's imported
shadows. Source `7b75651234c3235cd5a0ddd82d0a02c45fb3c8a8` then gives all three
external local_info results explicit `state.Query` bindings. Every original
assertion remains, and the transfer function, driver and10409/1001 corpora are
unchanged. The three-binding candidate passes the original default NanoVirt
shadow build with source/tool guards. This clarifies my new fixture; it does
not repair imported direct-call field projection or close #1259. Full
foundation/source/generated-sanitizer/paired qualification remains pending.

I also retain offline observation work completed before that gate failed.
A snapshot of15230 commands occupies27368902 bytes. Five full-history JSON
serializations take0.058–0.090s each before filesystem writes. My candidate
journal saves only changed tail records and explicit metadata updates, seals
prior rows, records every command start/PID/completion, and atomically publishes
the original terminal JSON once. It requires a save for each appended command.
Provider/setup metadata save sites must explicitly request metadata capture;
terminal publication always captures it. It has not replaced any collector.

I preserve my first offline control failure: its snapshot test appended all
historical rows without per-command saves, so ordered replay refused the missing
prefix. The corrected controls require those saves and explicitly test their
omission. All nine controls pass in2.718s under60, including interrupted and
truncated journals, late history mutation, metadata updates, terminal failure,
atomic publication failure and exact retained-snapshot replay. My independent
reader reconstructs all15230 command rows byte-for-byte. Five observed tail
saves take0.000068–0.009630s. These are different observation operations; I claim
no product/runtime or full-gate speedup from these measurements.

I independently read367 records with:

```sh
python3 docs/evidence/darwin-current-source-20261124/diagnostics/affine-local-shadow-localization-linux-20261125/read_archive.py
```

My manifest SHA256 is
`6101cb0c7d2fa544e8a63a15d814d9e977dfd88a23ae6f27af34b20abdc2b319`.
The reader binds the selected old source to exact Git blobs where its bytes
already belong to the prior failure archive; I correct its draft lookup of
those files as new standalone external records. I preserve the initial failed
control, full trace, minimal source/products and both typed candidates.

No compiler repair, collector replacement, full five-method foundation,
source/generated-sanitizer/paired, CFG/profile, pure C-emission/cutover or release
acceptance follows from this diagnostic receipt. The queued global/array gate
remains unexecuted. A corrected full foundation still needs fresh admission.
