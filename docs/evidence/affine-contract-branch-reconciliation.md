# My affine contract branch reconciliation

I reconcile `codex/affine-5-execution` at
`a08341e5b2c39ac590d88424e40bcc1447424a06` and the older worker head
`cb99b86e838e6e1a26e1b4fecb88bc0a31a842e0` against integration
`fe17a44f`.

The unique commits change only `docs/AFFINE_TYPES_DESIGN.md` and
`docs/AFFINE_TYPES_GUIDE.md`. Both current documents are byte-identical to
`a08341e5`, verified with `git diff --exit-code`. The later contract resolves
the older branch's non-resource affine/drop proposal and provides whole-value
destructuring for nested resource resolution. I retain those current documents
unchanged and record both heads as ancestors. PR #259 is already in history.

This is design reconciliation, not ownership implementation acceptance. The
documents explicitly distinguish their normative 5.0 target from the limited
current checkers. No compiler test is claimed for this documentation-only
merge. `git diff --check` passes.

The remaining C-seed recovery branch `c53e27a7` still needs implementation
review. Its fixed 256-place table skips adding further owners, and its
expression walker ignores unhandled AST kinds. Those are not safe foundations
for claiming complete path-sensitive conformance. The existing C-seed task
`task_c4e2f078cef8c4e461f0de3711c8a2b9` retains that work; I do not merge
the prototype merely to reduce the outstanding branch count.

MAC contract task `task_4ac22044ffda9f93b336a85573293bc2` is completed
when inspected. Full C-seed and self-hosted analysis, ownership metadata,
runtime integration and release acceptance remain open.
