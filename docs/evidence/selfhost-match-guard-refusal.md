# Self-hosted match-guard refusal

I tested checkpoint `5f7d521efd6947b1ac7086a1563070828d248dd7`, tree
`7357087d32ba01cff99119733da0478a0f2835cd`, from a fresh detached Darwin
checkout. The source archive remained
`bdde5526a20451739ececfb7e38b101322692a49bc6385e4426c99946671da7e`
before and after the gate.

`make -j8 bootstrap` passed in 343 seconds. Stage 1, Stage 2, the installed
compiler smoke and the installed compiler's no-C-seed check passed. The
bootstrap also executed my modular parser shadows, including the three guarded
arm forms, exact locations, ordinary unguarded controls and the non-match
diagnostic fallback.

I then stopped at the first source-route mismatch. `bin/nanoc_stage1` rejected
the guarded integer fixture at the correct line 2, column 24, but printed the
old generic text:

```text
Parse error at /private/tmp/nanolang-pr794-final.71gsxM/fixtures/int_guard.nano:2, column 24: unexpected token 'if'
```

It did not print `I do not yet accept guarded match arms in my self-hosted
parser`. I did not run the remaining Stage 1 or Stage 2 source fixtures, replay
the failed input, weaken the diagnostic, or change either parser route.

I retain the raw evidence under
`/private/tmp/nanolang-pr794-final.71gsxM/evidence`. Its hashes are:

```text
834ec7528e6ce7da125a75f429ffb0eac61f81764d23bd4f6391b26ae80d45ab  inventory-before.log
3e45e17a1c39be8cc9aba4e4dc6dbbc026375da3d5e935a59ed66a81b9f40675  bootstrap.log
5b3f0323f09b2b8e125d3ca74dc1947659fe0bbb855b88446f8b0181cf22812e  bootstrap.status
48b750f4075a188a632eb8dad01f6dc7923c7cada692a3769699927b66f3ce2d  int_guard_stage1.log
d241b5d0def7b38ec07191c4f851f7b4fefdade607b91482e1d038a0578b6b5c  first-source.status
0698ac98538d796aacc8e00137390f4dc6d2e1482954fb7cb2a9f83d0c3bc2a2  inventory-after.log
8bb729d19ca0196a5f66225f10f0b94cfd8ac5a6bef39449cef8653f83b37ccb  fixture-SHA256SUMS
```

The retained manifest has SHA-256
`b4129a6b20f88533c0fcf8c4b49e022c03badb3fcba6042fe9f382cda70045f4`.
This is a modular-phase success and an installed-route failure, not full match
policy, product, or release evidence.

## Corrected installed route

I recovered the preserved fleet correction after its publication-contract
failure, reviewed it against the retained first outcome, and corrected one
additional compatibility boundary before execution. The installed compiler
now uses the parser's first specific diagnostic when one exists. When the
parser records no diagnostic, both the human-readable generic message and the
machine-readable generic message retain their prior spelling; I do not add
quotes or substitute a new location in that fallback.

I qualified production-and-test checkpoint
`0d13778aa1cd63fc6755291c2e60af68531d507c`, tree
`e3e5a787b49ddba9202d7733d83882918f47736b`, in a fresh detached Darwin
checkout:

- `make -j8 bootstrap` passed in 332.82 seconds. Stage 1, Stage 2, both hello
  smokes, the installed compiler smoke and the no-C-seed check passed. The
  native stage binaries differed; I do not claim a native fixed point.
- The two existing guarded selected-ownership methods passed in 0.083 seconds
  across the C seed and both installed self-hosted stages.
- Twelve direct installed-route checks passed in 11.30 seconds. Stage 1 and
  Stage 2 each refused integer, wildcard and named-payload guards at the exact
  `if` location, retained the specific first-person message in text and JSON,
  and preserved the prior output artifact. Each stage compiled and executed
  the ordinary scalar and named-payload controls. Each stage also retained the
  existing non-match text and JSON fallback byte-for-byte.

The checkout remained clean. The six tracked implementation, test, roadmap
and evidence inputs and the resolved Apple Clang, Python and Make executables
have identical hashes in the retained before and after inventories. The
resolved compiler was Apple Clang 21.0.0 with SHA-256
`1590ac950a3d627817d09ade5cb60b2115f17a72182a3141e010b4bcc482a0c9`.
The built compiler hashes were:

```text
484e332b3c960456d0e52c28839a8b9d3eeaa37d2f01e89729eed325363b3d9d  bin/nanoc_c
419f4f6fd46e299ce599cbe0eb25743ac5754a1d4a983e8ea8fbd29f9ed5a2f6  bin/nanoc_stage1
088ab9c5ca22cd315ea3caea1413a688e77deb6c4cb40bd14f1f4c6171dbadef  bin/nanoc_stage2
```

I retain the corrected evidence under
`/private/tmp/nanolang-pr794-corrected.IFUAM8/evidence`:

```text
8031a6ef43ba9d55d9ae0999ffb54a238fc57c1ec8c3d9292278a4b168b9b02d  bootstrap.log
83e32c3d41328c43f7cbdbb97a652bac67c349a77d3be77684028b6a7dc8dd2e  ownership-guard-methods.log
2fde31fd64a559da46329f5c2d833181f91bc3e1b683380488fb885d5a3db5ae  installed-guard-gate.log
dff6b7f7669a94d98ab0fcb678b338e263d010b2cd00381d322391ca74f39517  installed-guard-results.json
a90dcf87b61f8151dfb05e92c7ee7858aadcaa9aa3ba067b18d13709b204d616  inventory-before.log
44954f89bcf3e0f742ee2b8ed803b5fdf7f775626389a08d7f992733b8959817  inventory-after.log
```

This closes only my checked self-hosted parser capability boundary. It does
not implement guarded matches in that AST, align dispatch or no-success
semantics, qualify the product branch, or authorize publication.
