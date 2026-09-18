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
