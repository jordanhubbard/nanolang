# PR #284 reconciliation

I reviewed both incoming commits and the complete six-file diff of
[PR #284](https://github.com/jordanhubbard/nanolang/pull/284) on 2026-09-15.
The head is `66cb4dc59c027bb2bc1ec6fb165080cedc199e93`; its test commit is
`be1bf32231aabb5ba16e332bbcb2b7f21163c181`.

I retain the existing `file_compare_identity` and `outputs_preserve_sources`
implementation. It checks root and loaded dependency identities before parsing,
diagnostic writes or shadows. Separate destination checks also protect artifact
and diagnostic collisions. I do not introduce a second identity API or duplicate
checks. These remain stable-filesystem checks, not race-free publication.

I retain the incoming Make target and shell entry point, routing them to three
existing Python regression methods. Their private temporary directories avoid
the incoming PID-based path and recursive cleanup of a predictable directory.
The tests require identity-specific diagnostics, preserve source and prior output
bytes, and exercise root/dependency aliases through relative paths, symlinks and
hard links for native and C-source targets (24 combinations), a diagnostic alias
before a parse error, and a symlink-loop lookup failure. Arbitrary compiler
failure is not a passing identity check. The target selects rebuilt Stage2;
the shell entry point retains the `NANOC_SELFHOST` compiler override.

`make test-selfhost-path-aliases` passes on Darwin after rebuilding both
compiler stages and running bootstrap smoke checks. All three methods pass
in 0.854 seconds. The shell entry point also passes from `/tmp` with an
absolute Stage1 override (0.309 seconds). The bootstrap/Stage2 log is
`/tmp/nanolang-pr284-gates.log`. These focused checks are not a full release gate.

Historical PR checks report success except cancelled Pages build/deploy jobs.
The sole review comment reports Copilot quota exhaustion, not approval.
I reconcile under the user's merge authorization and preserve the original
head as a merge parent. I leave the PR open until integration reaches main.
Original MAC task: `task_507117859a0642f4882b7015296d8af6`.
It remains stopped and unowned; MAC refuses a claim from that state.
