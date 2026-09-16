# PR #273 reconciliation

I reviewed the complete one-commit, four-file change in
[PR #273](https://github.com/jordanhubbard/nanolang/pull/273) on 2026-09-15.
Its head is `d66362a10e3bb43087b36db577d7a00cb4f1d269`.

My current module builder already quotes source, include, dependency, object
and shared-library paths through bounded command builders. It also captures
dependencies and source snapshots, retains compiler-phase argument handling,
and publishes complete cache generations. I retain that production code and
its existing `shell_path.h`; the incoming parallel helper API and replacement
compile/link construction would duplicate quoting and lose newer behavior.

I retain the incoming path-word regression using the actual production helper.
Its shell runs inside a private `mkdtemp` directory; checking for an injected
sentinel does not remove an unrelated file from the working directory. The
test checks exact argument bytes and rejects a command-buffer overflow.

I also require the existing manifest-backed literal-path regression from
`test-module-metadata`. It compiles single and multiple ordinary sources plus
private shared sources using include/cache/source paths containing spaces,
quotes, shell substitutions, backslashes and newlines. It checks execution,
absence of an injected file, conservative cache records and changed-header
behavior. This is stronger evidence than shell word comparison alone.

The standalone quoting test passes; the manifest regression passes all five
fixture variants (3.830 seconds). Local log: `/tmp/nanolang-pr273-paths.log`.
These tests do not establish a complete manifest sandbox or release readiness.

`make test-module-metadata` also passes after rebuilding compiler stages and
running bootstrap smoke checks. It runs the new quoting test, all five manifest
path variants and the existing metadata tests. Log:
`/tmp/nanolang-pr273-metadata.log`.

Historical checks report success except skipped Pages deployment. The sole
review comment reports Copilot quota exhaustion, not approval. I merge under
the user's authorization, preserve original head ancestry, and leave the PR
open until integration reaches main. Original task:
`task_8c10a946dfa14d92b491fd80f4635187`.
