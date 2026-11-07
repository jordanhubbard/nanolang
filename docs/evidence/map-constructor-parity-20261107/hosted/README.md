# My hosted scalar map checkpoints

I retain two complete hosted runs with their original source boundaries.

| Checkpoint | Revision | Platform/compiler | Methods | Result |
| --- | --- | --- | --- | --- |
| Exact runtime storage | `a3c3f76b4` | Linux GCC | 4 | Pass, zero skips |
| Exact runtime storage | `a3c3f76b4` | Linux Clang | 4 | Pass, zero skips |
| Exact runtime storage | `a3c3f76b4` | macOS 15 native compiler | 4 | Pass, zero skips |
| VM/C AOT plus unchanged storage/lifetime corpus | `7d07751b4` | Linux GCC | 36 | Pass, zero skips |
| VM/C AOT plus unchanged storage/lifetime corpus | `7d07751b4` | Linux Clang | 36 | Pass, zero skips |

My [runtime jobs](runtime-jobs.json) belong to
[run 42418393564](https://github.com/jordanhubbard/nanolang/actions/runs/42418393564).
My [AOT jobs](aot-jobs.json) belong to
[run 42419312675](https://github.com/jordanhubbard/nanolang/actions/runs/42419312675).
My [summary](summary.json) records exact archive hashes/CRCs, original method
counts/timings, source identity and skip boundaries. Each directory retains
complete compressed build/test/provider logs where supplied, compiler version,
worker records and the original artifact member inventory. Full original ZIPs
remain under `~/.worktrees/nanolang/audits/2026-11-07/map-runtime-a3c3/` and
`~/.worktrees/nanolang/audits/2026-11-07/map-aot-7d077/`.

I compare each runtime input hash with its exact committed file. I compare
each AOT artifact's complete source-tree record byte-for-byte with Git at its
recorded revision; three fresh assembler/VM/translator providers remain
unchanged through the test phase. The AOT runs include all four scalar map
pairs, empty snapshots/growth, optional writes, detached snapshots across
collection, forward map parameters/returned arrays, signed extremes and
prior-output refusals. They preserve the selected original map key storage,
lifetime, byte-debt, collection-debt and optional-array-read modules.

Runtime storage checks include the complete original string-map runtime fixture,
all four exact key/value pairs, copied inputs/results, aliases, growth, extrema,
zero successful-teardown allocations, invalid-input boundaries and all 136
measured allocation-denial positions. The AOT checkpoint adds the checked key
kind helper and its controls. I do not treat the earlier runtime revision as
Darwin qualification for that later source change.

Generated snapshot allocation-denial sweeps, the complete original source
constructor matrix through fresh stages, full intended JUnit, latest Darwin
AOT/integration, source custody, GitGuardian and native lifecycle/receipt
requirements remain open. These are executed runtime and bounded native AOT
checkpoints, not complete map or release acceptance.
