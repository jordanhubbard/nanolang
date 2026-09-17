# My PR #287 reconciliation

PR #287 has head `a58dd71d2197a49b434cf69742faac141d907957`. Its title names
paired compiler arguments, but the commit replaces 267 files relative to its
parent. I do not apply that old snapshot over my newer release work.

Its complete tree is byte-identical to integration ancestor `3fb98917`:
both trees are `a9d5caa7f62df19e375cdd3d8e65e1be9a17da93`, and
`git diff --quiet a58dd71d 3fb98917` succeeds. The original paired-operand fix,
`d2082006`, is also an integration ancestor. This establishes that the whole
snapshot is represented in my history, not merely the change suggested by
the PR title.

I retain the current source and add the PR head as an actual merge parent.
My subsequent callback contracts, capture deadlines, source evidence and
compiler changes remain intact. The merge is reconciliation, not a new
implementation of argument handling.

## Current verification

I rebuilt `obj/test_module_generation_probe` and ran these methods from
`tests.test_source_snapshots.SourceSnapshots` on macOS:

- `test_paired_fragment_normalization`
- `test_forwarded_include_operands_are_not_rebased`
- `test_split_assembler_search_restored_inputs`
- `test_split_assembler_search_order_phases_and_recovery`

They check split/joined argument identity, operand ownership, restored-input
capture and actual cache reuse. The phase/recovery method covers common,
platform and package flags, integrated/external assembly and both cache roots;
it also checks changed inputs, rejected missing inputs, preservation of the
last valid generation and successful recovery.

All four methods pass in 121.173 seconds. The command log is
`/tmp/nanolang-pr287-focused.log` on this host. These are
focused tests, not a fresh Linux matrix or full release acceptance. PR #287's
historical GitHub checks include failures and no review approval; I do not
represent them as green or merge this directly into main.

MAC task: `task_8500a5ea65584cc3a5cfb02a7812b16b`. The PR remains open until
the integration branch is landed on main or explicitly superseded there.
