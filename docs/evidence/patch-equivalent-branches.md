# My patch-equivalent branch heads

At integration `1c7105a1`, a refreshed local/remote inventory found 21 heads
outside integration ancestry. I record all names, full hashes and patch
classification counts in `branch-inventory-1c7105a1.json` beside this document.
The earlier conversational count of 22 was incorrect.

For these six heads, `git cherry HEAD <head>` reports no `+` commits: all
commits not already ancestral have patch equivalents in integration history.

| Branch | Head | Equivalent commits |
| --- | --- | ---: |
| `codex/affine-5-scope` | `24c64b41` | 1 |
| `fix/gpu-kernel-artifact-location` | `e82088a2` | 1 |
| `opencode/integrate-pr-136` | `010790ab` | 1 |
| `opencode/integrate-pr-137` | `23569d21` | 1 |
| `origin/feat/nvm-container-hardening-tests` | `f1bcac0a` | 2 |
| `origin/fix/forth-ide-terminal` | `a630375c` | 1 |

I retain these heads in merge ancestry without replacing current source.
Patch equivalence establishes historical inclusion, not present feature
correctness, full platform coverage or release readiness. Existing changes to
ownership documentation, GPU build paths, call verification, linked calls,
container/wire checks and terminal rendering remain subject to their gates.

`make test-verifier test-cop-protocol` passes on the current integration tree.
The protocol suite passes all 35 tests, including little-endian layouts and
hostile lengths. The log is `/tmp/nanolang-equivalent-branches-gates.log`.
This is not a fresh graphical, GPU or ownership acceptance run.

I do not delete any local branch or modify the original user worktree. Two
old affine worktree registrations point to missing directories; their branch
commits remain available and I do not treat that as permission to erase them.

The remaining 15 heads require content review. In particular, the local
affine checker prototype uses fixed 256-place storage and a recursion cutoff;
I will not assume it fulfills the path-sensitive ownership contract merely
because it contains tests. The 75-commit frontend branch also needs substantive
reconciliation without weakening newer verifier checks.
