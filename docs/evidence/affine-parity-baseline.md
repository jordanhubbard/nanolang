# My affine frontend baseline

At integration `ef5b014b`, the existing `test_affine_selfhost.sh` checks only
positive acceptance on the C seed. It checks two negative cases on Stage1
and Stage2, accepting any compiler failure as rejection. That does not
establish frontend ownership parity.

I replace the shell's predictable temporary directory and recursive cleanup
with a Python `TemporaryDirectory` per compiler/case. The gate now checks all
nine combinations of three compilers and three existing fixtures. Positive
output must execute with result 42. Negative output must have a positive
failure status, an ownership-related diagnostic, and unchanged prior bytes.
Crashes and unrelated shadow failures are not ownership diagnostics.

`bash tests/test_affine_selfhost.sh` exits 1 after 9.682 seconds:

- All three positive executions pass.
- Both bootstrap stages reject both negative fixtures with the expected
  diagnostic category and preserve existing output.
- C-seed use-after-move reaches shadow execution and fails an assertion.
  It does not produce an ownership diagnostic.
- C-seed unresolved ownership compiles successfully.

Log: `/tmp/nanolang-affine-parity-baseline.log` on this host. I deliberately
leave the strengthened gate failing until the checker is repaired; I do not
mark ownership acceptance complete. This corpus is only an initial baseline,
not the full normative branch/loop/borrow/aggregate matrix.

The old C-seed task `task_c4e2f078cef8c4e461f0de3711c8a2b9` is cancelled
despite unfinished roadmap work. New task
`task_91ae827be4154eaa8f22698aeecc8cf1` tracks the corrected gate and checker
implementation. The `c53e27a7` recovery prototype remains unmerged pending
review of silent owner-capacity and unhandled-expression gaps.
