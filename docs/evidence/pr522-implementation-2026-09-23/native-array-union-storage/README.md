# My array/union transport baseline

I reopened draft #522 at my creator’s explicit request on 2026-09-24. My
baseline is pushed compiler checkpoint `8237cd980e6dd4495f8312ce1b708d64728310be`.
This directory records the next failing cases, not a repair or release gate.

`array-envelope.nano` stores Box/record, Text/string and Empty variants inside
ordinary records in one array. Its assertions pass through my C seed and both
canonical VM stages. Both canonical native stages refuse translation during
`ARR_LITERAL`: conflicting aggregate shape kinds `optional/record`. I retain
the complete results in `envelope-baseline.log`.

`array-choice.nano` uses direct `array<Choice>` instead. My C seed rejects its
annotated elements as struct versus union, and both canonical routes reject the
local type. This is a separate frontend admission barrier, not evidence about
native storage. I retain those diagnostics in `baseline.log` and track them
under `task_d44b2db373d38a942db3e8c4567b8044`.

My native classifier currently equates each record-array literal producer with
one element shape (`OP_ARR_LITERAL` in `src/nanoisa/nvm2c.c`). It also omits
constructor maps at record-array reads and transport. These are repair sites;
the observed diagnostic alone does not establish a complete root cause.
Array aliases and mutations must share representation obligations without
sharing tag-guard authority. An isolated literal change is insufficient to
qualify calls, globals, writes, unknown producers or invalid projections.
The native work remains under `task_065a1a2c9858b8968fd3851135b48c47`.

I reproduce both comparisons with:

```sh
NANO_CC=/opt/homebrew/opt/llvm/bin/clang python3 docs/evidence/pr522-implementation-2026-09-23/native-array-union-storage/run_comparison.py
```

The runner deliberately exits nonzero while these defects remain. It preserves
the existing 120-second per-command limit, shadow execution and runtime
assertions. These baseline runs are ordinary builds, not fresh sanitizer
qualification. `baseline-sha256.txt` records the compiler source and executable
identities. The earlier nested-record sanitizer results remain scoped to their
own retained fixtures.
