# My aggregate-transfer C expectation audit

I retain pure source checkpoint `46b19b1cc88a6fa866199e9676dd4693f27c2c84`
and corrected fixture checkpoint `5810558180c3d9b34c807d3b1e146043c7ecb5cf`
under #1142. I implement STRUCT_GET97, MATCH_TAG107, AGG_PACK251, AGG_GET252
and AGG_TAG254 in NanoLang. This receipt qualifies my diagnostic expectations
against unchanged C; I still require execution of the new NanoLang source.

All1158 actual C commands pass in4.028 aggregate command seconds. My1001
original frame/flow/FIFO reports retain SHA256
`f5f4555c26ea20dbc13237f4a9843c42e98b16f7fc6f4e9f9ea3de7b3d3dcbd0`.
My15222 new cases compare complete results, exact messages, stack values and
exclusions, globals/handler context and mutable authority snapshots. Complete
aggregate reports have SHA256
`f26b2400f0ec919b5fcf5c15a869f14db43bfd2a6394842ac5b6040a9777a468`.

I preserve each consumer's physical tag/observation/owned guards, graph mode,
copyable record constructor layouts/counts/fields, graph-independent union
ordinal/variant construction with scalar/STRING and copyable/resource record
payloads, layout-based match scans without consuming/refining, proven-arm
projection, shared/exclusive holds and observation STRING access. I keep
original256-slot/field, unsigned operand and malformed-input bounds. My C
state diagnostic compiles unchanged state source in place of its ordinary
object; the embedded bytecode step replaces its ordinary object. All other
bootstrap object, production source and C tool hashes match before/after.

My first C audit passed the original1001 reports but rejected5305 new inputs
before transfer: a shared scalar-union fixture's empty arm started at offset0
after a nonempty Some arm. I preserve that complete run and correct only the
empty arm offset to the preceding field count. All15222 expectations remain
unchanged. Exactly5305 ordinary inputs and seven derived malformed inputs
change; their remaining bytes are tied to the preserved source correction.

Strict C99 syntax, NanoLang typed AST, mandatory shadow presence and diff checks
pass. The source-review receipt binds its earlier source checkpoint; selected
NanoLang/C source bytes remain identical at the corrected fixture checkpoint.
The typed-AST compiler matches the preserved original bootstrap product/image
hashes. I do not claim executed shadows from that check.

I independently verify38061 archive records with:

```sh
python3 docs/evidence/darwin-current-source-20261124/diagnostics/affine-aggregates-c-audit-linux-20261125/read_archive.py
```

My manifest SHA256 is
`bdaf2ee0fd1b5a1ab215ea3bc1a9897c1f4b54aa69ec345f0c33ae3d925d430f`.
My original8-case/600-second and new32-case-or8KiB/60-second batches remain.

Executed NanoLang shadows, source/VM/native/generated-sanitizer and paired-host
qualification remain pending, in order after the admitted foundation and
prepared global/array gate. I grant no branch convergence, CFG/profile,
executable, AGG_SET, pure C-emission/cutover or whole-release acceptance.
