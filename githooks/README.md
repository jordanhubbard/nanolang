# Git hooks

Version-controlled git hooks for this repo. They keep the CodeGraph index
(`.codegraph/`) in sync with the working tree after commits, merges,
checkouts, and rebases.

Git only runs hooks from this directory if `core.hooksPath` points here.
One-time setup per clone:

```bash
git config core.hooksPath githooks
```

The hooks are no-ops when the `codegraph` CLI is not installed.
