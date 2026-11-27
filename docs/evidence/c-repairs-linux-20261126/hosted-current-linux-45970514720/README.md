# My current hosted Linux original gates

I retain both completed Linux jobs from
[run45970514720](https://github.com/jordanhubbard/nanolang/actions/runs/45970514720)
at product source6e09c07a9f1651294e4669282cdc17ee14126775. The selected compiler is
Ubuntu GCC15.2.0-16ubuntu1, with the existing verified capture assembler selected
by the original action. Both workers pass the workflow's hosted-worker and
storage-headroom requirements before building fresh tools.

My exact GitHub ZIP archives contain3164 bootstrap and178 cache records. Their
SHA-256 values match the API's archive digests:

| Artifact | ID | SHA-256 |
| --- | --- | --- |
| Bootstrap |15186352989|633e13d9a7fbadefbf3158adceda306d02feef5c31b10f1451d3587f1efd84f8|
| Cache |15186001791|8856656d5041341444e6d0f15f0fb13a67e7e7d5f2aea5c1751098d86921ae71|

I seal these exact ZIP bytes, API snapshots and independent checkers in11
records/99 chunks,191147524 raw bytes. Manifest SHA-256:
`0c1334f9a021e0988856a57affbd89fd12ca59fa63eb730fe6c07de6edf86653`.
My reader reconstructs and checks every ZIP member without executing a compiler
or product, then verifies both component boundaries below. Its private altered-
manifest control refuses before archive reconstruction or product execution.

## My native component

All17 original bootstrap steps pass with their exact retained log hashes.
Stage1/Stage2 raw modules are identical1064512 bytes, SHA-256
`376f17aca1f2e424045ebe3e6ad922edf035305d0999c5f191e57f1e53cb3d61`.
Raw equality is checked within this hosted job. The installed image matches
Stage2 and the original no-C-seed probe passes. Native36 passes32.953s and
emitter12 passes257.399s, with original300-second setup commands and
1800/30/default10 workflow limits unchanged.

The reader verifies all1377 Git source inputs against the pinned revision,
the exact receipt-bound generated modules/index.json, all five compiler
products, all17 step logs and six actual host libraries. The recorded source
tree matches Git; copied selected tools and actual compiler/runtime images
match both before/after guards. The old hosted ee5 archive's missing-index gap
remains preserved separately; this artifact retains its own actual index.

## My cache and regression components

The complete original cache target reports284 methods with34 existing platform
skips (9/24/1 across its affected groups). All three assembly-worker methods and
all six expanded array/pointer regression methods pass. My reader checks the
original log groups/counts, tool/image guards and exact source tree. It does not
re-execute tests or claim that every removed per-case temporary product survives
in the uploaded artifact; the ZIP bytes retain exactly what the worker uploaded.

```sh
python3 docs/evidence/c-repairs-linux-20261126/hosted-current-linux-45970514720/read_hosted_current.py \
  --archive docs/evidence/c-repairs-linux-20261126/hosted-current-linux-45970514720/archive \
  --repo . --workspace /path/to/mounted/volume
```

These receipts qualify the named hosted Linux components. Current Darwin and
the current-Stage2 private/public/cache/solver/callable rerun remain active at
this checkpoint. The two frozen structured-C CAST_STRING assertions still need
the explicit reconciliation decision. Required exact-candidate merge CI,
complete pure-translator composition, paired platforms, installed cutover and
all69 release requirements remain open. I merge, tag and publish no release
from these component receipts.
