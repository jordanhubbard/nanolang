# My array/union transport baseline

I reopened draft #522 at my creator’s explicit request on 2026-09-24. My
baseline is pushed compiler checkpoint `8237cd980e6dd4495f8312ce1b708d64728310be`.
I retain that failing baseline below and the subsequent record-array repair afterward.
Neither is complete release qualification.

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

The runner deliberately exits nonzero while the direct-array defect remains. It preserves
the existing 120-second per-command limit, shadow execution and runtime
assertions. These baseline runs are ordinary builds, not fresh sanitizer
qualification. `baseline-sha256.txt` records the compiler source and executable
identities. The earlier nested-record sanitizer results remain scoped to their
own retained fixtures.

## My record-array repair

I convert record-array literal and write producers into element storage rather
than equating their distinct nested shapes. I carry element constructor maps
through local/global loads and stores, direct calls, returns, tail calls and
stack joins. Array aliases join possible layouts in both directions because
either alias can receive a write. Reads retain representation metadata and a
fresh value witness; they do not inherit a selected constructor guard.

My permanent source regressions are
`AggregateFormatting.test_union_record_array_literal` and
`AggregateFormatting.test_union_record_array_aliases`. The second starts with
only a boxed record payload, introduces a string payload through a mutating
function, observes the update through the global/returned aliases, and checks
that a previously saved record still has its original payload. It also appends
a record and replaces the first element with the empty variant.

My raw indexed-union tests add array/local/global/stack-join/tail-call transport
in both producer orders, plus unguarded, wrong-tag, wrong-field, bypass and
unknown-array-caller refusals. The existing prior-output assertions remain.

Direct source `array<Choice>` admission is still blocked in the frontends;
`run_comparison.py` therefore still fails its direct-array method. Its
record-array method is the repaired control. Constructor metadata for deeper
container combinations and complete parent acceptance remain open.

My broader generated-product comparison encountered a 120-second C-seed
shadow-driver setup timeout after 54 methods. I retain it in
`source-comparison-timeout.log`; it does not establish an infrastructure cause.
My final focused instrumented translator run passes 13 methods, including the
strengthened mutation-only payload introduction and 38 routed raw translator
invocations. The nested untracked-array write refusal preserves prior output.

My final ordinary and fresh private full sanitizer gates each pass 2,657
translator assertions and 1,942 shape checks. The private gate verifies ASan
and UBSan object symbols and retains `detect_leaks=1`, stack-use-after-return
checking and halt-on-error. Logs are `repair-ordinary.log`,
`repair-full-sanitizers.log` and `repair-focused-sanitizers.log`;
`repair-commands.txt` records their invocations. Source hashes are in
`repair-source-sha256.txt`. The compiler-stage executables are the retained
baseline stages; this C translator change does not rebuild or requalify a
bootstrap fixed point. These are Darwin checks, not complete hosted/platform
or release acceptance.

With the concurrent gates complete, I reran only
`tests.test_source_record_unions.SourceRecordUnionShadows` with the same compiler,
product instrumentation, assertions and 120-second per-command limit. The
method passes all twelve retained source/shadow fixtures in 94.634 seconds
(`repair-isolated-shadows.log`). A passing isolated run does not establish the
cause of the earlier timeout; I keep that incident on my roadmap. The original
broad run was not a clean 55-method pass.
