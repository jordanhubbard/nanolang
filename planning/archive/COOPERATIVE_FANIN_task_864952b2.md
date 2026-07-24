# Cooperative Fan-In Integration Record — task_864952b2f88e43199182c6f0c60a5f4c

This note records the cooperative fan-in that integrated ten independently
executed child tasks into canonical `main`. It exists so the integration is a
concrete, verifiable repository change with a stable audit trail.

## Canonical base

- Integration branch based on canonical `main` at `aa458c71bbb581fd32b69a272d1c09ddbc29352c`.

## Integrated children (non-squashed commit ancestors)

Each child's exact `head_sha` was fetched and verified as a true ancestor of
canonical `main` (`git merge-base --is-ancestor <head_sha> main`), preserving
individual commit ancestry rather than squashing:

| Task | Head SHA | Title |
| --- | --- | --- |
| task_528dd8d5d59248d5ab8989075253bd10 | 84967dfcdaeb25e3644b2da933558ff2bb2ca2eb | Fix match arm binding scope in typechecker |
| task_65deef8a6c054da98429d561d0b5cd73 | f1c68b4466164028d169d12c4ad8495c3f415f1b | Complete TupleTypeRegistry integration in transpiler |
| task_a0c5e79236fb4c6c8608dda1f2cc0adb | b84a35148298bdbab6aa042512ed43d398e2d888 | Fix array element-type propagation in typechecker |
| task_455d7b6df84f4f248f627c00d87d1972 | 0844541271cb5c7dea2761c252c1a97f0e2ed133 | Close the one Admitted case in formal/Equivalence.v |
| task_4c8c112823354b36b272ea904783a69f | 1612182c0ceb4d25f0bfc6b39e3f6e97974abecc | Add missing nl_control_* core test coverage |
| task_6aaf7dd86a9f4d0587e18613812c4b55 | 7dd9c7903603d37541d0f0531ff5735cc8d54703 | Update proof-status docs now that Equivalence.v is Admitted-free |
| task_d380798f4600461cb9f36acd14cc58a5 | 038732721afdd158fa36ca0fe8bff42be9f7a110 | Finish tuple-return typedef integration in transpiler.c |
| task_4b8c4a3f6b614bc1a6e8526f26bdeb6a | f213d971088d6d31184360b1682e24f56487d8d7 | Re-enable module metadata embedding disabled by a bus error |
| task_a50788f084a24056a23b3a6e03abb27f | 07c28500a1717d4a31362a4556466c56df265415 | Align README backend claims with WASM/LLVM reality |
| task_4193cbb12d14433491ed5f4c5c81d73f | c195ef79b5402fdb97b96a825630d7cdc8aee199 | Reconcile stale planning/status docs against implemented reality |

## Verification of the combined tree

- `make build` — 3-stage bootstrap completed (Stage 1 C reference compiler,
  Stage 2 self-hosted parser/typecheck/transpiler, Stage 3 bootstrap validated).
- `make test-quick` — 12 passed, 0 failed, 0 skipped.

All ten required child outputs are present as non-squashed ancestors; no child
was missing or unintegrable.
