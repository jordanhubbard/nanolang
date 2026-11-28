# My 5.1 release-readiness inventory

I review release scope at integration source
`4a1d4e24eeeb134addfc362411ef3bec844e743e`, with main observed at
`82c0f4f8097bfe41f4fc64f8f887865a43d85fa4`. My query exports span
2026-11-26T18:31–18:44Z. They are dated observations rather than an atomic
transaction or final-candidate qualification.

| Captured state | Count |
| --- | ---: |
| Original headline entries plus two additions |67 + 2 =69|
| Open issues |134|
| Release-scoped open issues |132|
| Existing issues outside this gate |2|
| Release-scoped open PRs |22|
| Draft PRs |13|
| APPROVED review records across all22 PRs |0|
| PR heads already ancestral to integration |18, including integration itself|
| PR heads ancestral to main |0|

My release-readiness gate **fails**. I merge, close or move no item in this
inventory. A stack ancestor alone does not establish its acceptance criteria
or an approved independent merge. Every scoped item remains open.

I inspect titles, bodies, labels, milestones and linked release work. The full
issue/PR body exports are retained. My review records identify the exact scope
passage or release parent for every item, with acceptance_verified=false.
Label/milestone-only selection would omit most work: only #1039 carries the
release milestone in this snapshot. Linked closed #1255 and merged #995 retain
their source context; their closure does not close related requirements.

I preserve two existing exceptions. #1089 explicitly says its additional legacy
AST-to-C returned-capture extension is outside the5.1 release gate. #1195 is the
separate fleet worktree migration. I do not alter their scopes, close either
issue, resume migration, or introduce another exclusion. All69 headline entries
and every original optional/deferred decision remain byte-identical.

I retain paginated review-state exports for all22 PRs. Their only review state
is COMMENTED; no APPROVED record is present. I retain review IDs, authors,
submitted times and reviewed commit IDs. Review comment bodies are outside this
state inventory. The reviewDecision field is blank for all PRs, so I inspect the
actual review records before reporting the absence of approvals.

`pr-bases.json` records GitHub's reported base commit. `live-refs.txt` records
actual branch refs separately. For #1176, #1122 and #1052 those values differ;
I do not treat the reported base commit as the current branch ref. The ancestry
observations compare each exact PR head with both observed main and integration.

Verify all exported bytes, scope paths, partitions, review records and counts:

```sh
python3 docs/evidence/release-readiness-20261126/read_audit.py
```

Also replay all44 read-only Git ancestry observations and the exact69-entry
source contract against a checkout containing the recorded commits:

```sh
python3 docs/evidence/release-readiness-20261126/read_audit.py --repo .
```

The offline mode reports git_ancestry_replayed=false. Neither mode executes a
compiler/runtime, queries GitHub, changes repository settings, or establishes
code/platform acceptance. Exact native CI, paired qualification, pure translator
cutover, release prose, tag and publication remain required under #976.
