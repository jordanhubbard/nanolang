# My ordinary-authority ledger correction

PR #995 automatically closed #1001 on 2026-11-02. The issue's full contract still
requires complete ordinary-record publication through required source paths,
including generic/imported/forward declarations and preserved module identity.
Its last retained comments explicitly leave later native/public/source work open.

I verify remote main remains `82c0f4f8097bfe41f4fc64f8f887865a43d85fa4`. My
[current source guards](current-publication-guards.json) retain exact full-file
hashes and line-numbered snippets from main, the `435dae34c` integration head and
reviewed source-float checkpoint `ebaefe24b`. The C publisher returns before
publication when `cg->enum_count` or `cg->union_count` is nonzero. Its declaration
review independently refuses enum/union declarations. The self-hosted publisher
also returns empty when the parsed program contains enum/union declarations.
These are retained guards, not missing diagnostics inferred from a job timeout.

I preserve every real bounded publication, private proof, runtime and native
qualification checkpoint. Their tests do not establish the full source contract
whose guards remain. I reopen #1001 and keep its implementation/acceptance row
unchecked. This audit changes the ledger and evidence only, not source admission,
runtime behavior, verifier assertions, native receipts or the release scope.
