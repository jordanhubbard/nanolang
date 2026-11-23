# I qualify STRING references without aggregate ownership

I publish source `2892d40b020bed97dae6522f89ba44f3a0d0cb93` on
`feat/5.1-service-module-integration` / PR1168. My preceding
[STRING receipt](../carrier-string-scopes-linux-20261123/README.md) remains
hash-bound. This archive retains 2037 records, 1417 exact Git inputs and 15 final
provider ELF images, including the native FFI fixture images. Fresh normal and
ASan/UBSan providers use the original strict GCC flags and serial Make commands.

I add an explicit `env_carrier_enable_strings` mode. STRING references and nested
scope marks can be qualified while `env_carrier_enabled` remains false: aggregate
adoption stays off, legacy aggregate cleanup stays on and enabling collection
fails. Upgrading to full-private mode works only between frames. Idempotent
calls and NULL refusal preserve their contracts. Failed fresh full-mode
construction restores ordinary mode. Ordinary environment creation, public
Value/container/FFI layouts and CLI activation remain unchanged.

| Actual gate | Normal | ASan/UBSan |
|---|---|---|
| New STRING-only/full mode boundaries and focused source controls | PASS | PASS, LSan on |
| Original full-private STRING scopes and active aggregate roots | PASS | PASS, LSan on |
| Original evaluator, ordinary mode, all 134 cases | PASS | PASS, LSan off |
| Original evaluator, STRING-only mode, all 134 cases | PASS | PASS, LSan off |
| Original evaluator, full-private tracking, all 134 cases | PASS | PASS, LSan off |

The original sibling-mutation and duplicate-loop-return source files are exact
copies of the preceding retained witnesses. The sibling witness balances
16/16/0/0 registered/released/exported/live in both new STRING-only and existing
full-private modes. The admitted synchronous async fixture checks sibling
mutation, 130 repeated STRING-returning calls and independent C caller/result
ownership, balancing 1339/1338/1/0 in each mode. Caller pointer/content/refcount
remain intact; all focused positive fixtures require zero GC objects/bytes after
context teardown. Full-private duplicate-return balances 6/6/0/0.

I retain two expected negatives. The current ordinary sibling witness still
reports ASan use-after-free. The exact duplicate-return source in STRING-only
mode completes both original shadows and balances STRING references, then LSan
reports 59 bytes/four allocations rooted in unbound static-array construction.
Its separate semantic replay disables LSan explicitly; it does not replace the
failed leak check. Full-private mode is clean. General #1116 aggregate lifetime,
#1213 native-byte acceptance and #1217 ordinary safety remain open.

My first queued-call diagnostic used the reserved name `handle`; I retain its
parser refusal. Renaming only that local still refuses `spawn` and
`scheduler_run` at typecheck. The roadmap already records these unregistered
source APIs. I do not bypass admission with synthetic declarations or claim an
admitted queued-task failure. The final async fixture uses the actual supported
synchronous async-call route.

I archive final provider images after adding the explicit C caller check, and
retain the preceding passing Make wrappers with their earlier test-source
snapshot. The earlier passing unit image was replaced by the final unit build;
I make no claim to retaining that intermediate executable. All failed setup
outputs and the retained baseline executable remain available.

Run `python3 docs/evidence/string-reference-mode-linux-20261123/read_archive.py`.
I verify raw/compressed hashes, Git bytes, the declared source boundary,
unchanged original evaluator and source witnesses, additive Make edits,
mode/reference counters, all 134 observed cases in each full run, retained
refusals/failures, source identity at qualification and ARM64 ELF products.
The five guarded tools and nvm2c source remain exact. All native jobs are
terminal; no Linux/Darwin reservation remains. Puck source/merge/capture/CI
ownership, frozen roots/audits/stashes and NAS/TCC/migration pauses remain intact.
No default activation, paired bootstrap or whole-release acceptance is claimed.
