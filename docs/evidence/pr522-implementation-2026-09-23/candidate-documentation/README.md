# Candidate documentation checkpoint

I correct my 5.1 candidate status, bootstrap instructions, default output,
reference compiler options, local inference and shadow-test claims.
My code remains pinned to `e70c0de4666139a01a77edd0e6dfa7fb9abe6f26` for
this documentation-only check.

I extract the getting-started hello program unchanged. Native and bytecode
compilation succeed, and both products print exactly `Hello, World!\n`.
`hello-validation.json` records the commands and exits. All six guide editions
render and validate; documentation freshness passes.

My repository-wide local link check fails on 383 targets omitted by this sparse
checkout. Every reported target exists in the pinned Git tree, as recorded in
`links-sparse-audit.json`; no target remains unaccounted for. This is not a
claim that the filesystem checker passed. The full-checkout hosted link job at
the code pin passed before these edits.

Final hosted acceptance, native and VM fixed points, and release presentation
reconciliation remain open. I have not tagged or released 5.1.
