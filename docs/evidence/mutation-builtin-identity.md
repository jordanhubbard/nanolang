# I qualify lexical mutation builtin typing

I qualify the bounded checker prerequisite
`task_13d1c53953674d279a849306c9eee26d` on Linux ARM64. My guard changes are
confined to `src_nano/typecheck.nano`: bound `array_push` and `array_set` names
continue through existing ordinary symbol checking, and only an unbound push
uses builtin result inference. Qualified lookup uses the exact resolved name.
I preserve C's existing reserved `array_set` declaration policy.

## I retain each terminal

| Frozen checkpoint | Boundary | Result |
| --- | --- | --- |
| `66a4686e1` | First bootstrap | Status2,20.035s: two new shadow expressions had an excess delimiter; no source acceptance claim |
| `82ce8b29d` | Corrected fresh bootstrap | PASS269.511s; Stage1/Stage2 and mandatory shadows; source and eight actual host tools unchanged |
| `54eab7f3c` | First focused suite | Three methods pass; unbound method has three parse failures in97.039s because two `array<float>=` spellings form GE tokens |
| `95a1f6224` | Corrected unbound method | PASS8.031s with all three producers; alias, push, set, length and element assertions unchanged |
| `95a1f6224` | Previously unrun adjacency | Seven methods PASS8.276s: five canonical prefix-conversion and two scalar-reduce binding/refusal methods |
| `0e4d2436d` | Added nested scope method | PASS0.039s; two source cases through three unchanged checker drivers |

The first focused suite is not globally passing. Its three passing methods
establish36 full-source checker cases,16 typed refusal/output-preservation
subcases and C's reserved-set policy. The later nested method adds six checker
cases, for42 total. Declared, local, formal, initializer-before-binding, nested
restoration, wrong result and wrong arity cases are parsed and checked by three
independently compiled checker drivers. Those drivers never execute the tested
bound-call bodies. This checker change does not claim to repair a separate
emitter builtin-dispatch path, general noncallable handling or all qualified
argument checks.

I preserved rejected outputs and executed no refused artifact. I did not repeat
an unchanged bootstrap or the36 passing checker cases after the fixture-only
space correction. I retained the three checker-driver binaries and checked their
hashes before/after the nested controls. All five bootstrap outputs still match
at the final seal; compiler production is identical since82ce. The focused setup
may populate module caches, so I preserve input inventories before/after setup
and each phase without asserting whole-object-map equality.

## I retain reviewable evidence

My [manifest](mutation-builtin-identity/manifest.json) hashes73 reports, with
38 retained fixture artifacts,2253 final source inputs and eight final bin tools.
Each gate has exact commands, bounds, environment and before/after source/tool
maps. The corrected bootstrap and all later source maps are internally equal;
I verified every final source hash against the current tree. Actual GCC/G++/
cc1/assembler/linker/make/Python/git paths and hashes are retained. I selected
`ASAN_OPTIONS=detect_leaks=1:halt_on_error=1` for the focused/adjacent runner.
The unchanged conversion harness removes its successful temporary artifacts;
I retain its command results without claiming to archive those removed files.

The external archive is `/tmp/nanolang-mutation-identity-qualification.tar.gz`,
SHA256 `95022101d036e413608859c85720120cec38add96927eede11d862d810b0f920`.
The source/tool tree is `/home/jkh/Src/nanolang-mutation-builtin-corrected`.
The archive contains the five gate evidence directories and five retained
fixture directories. The manifest records their exact paths and hashes.

My first packaging attempt asserted an unjustified minimum of41 retained files;
it stopped before publishing an archive/manifest. I retained that runner,
terminal and partial reports, recorded the defect before correction, then
published only after verifying the five directories and exact38-file inventory.
No execution was repeated for packaging.

## I keep the remaining boundaries explicit

MAC task state includes an independent failed auto-worker attempt. Both local
and parent description updates are blocked by the existing credential scope;
I preserve the pending description and ROADMAP evidence without restarting a
worker, changing shared tunnels or claiming completion. Actual-merge ledger
reconciliation remains pending.

This is Linux checker qualification, not Darwin source acceptance or owner-ARRAY
admission. Tasks18731/bba622, parent430220, mixed ownership4be, product and release
remain open. Owner-ARRAY source implementation still depends on qualified public
runtime activation and a separately reviewed lowering checkpoint.
