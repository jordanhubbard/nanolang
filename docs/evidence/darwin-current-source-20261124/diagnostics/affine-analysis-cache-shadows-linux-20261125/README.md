# My private analysis-cache ordinary shadows

I preserve source `0e9aa6e58dbc46ede67e1631731c17bf949467da`, typed AST, shadow presence, diff and ordinary C-producer imported-shadow checks at original600/default10. All pass with unchanged selected source/tools. These primitives initialize eight isolated function slots, preserve declaration/global guard order, reuse cached success/failure, reject in-progress recursion, reset fresh allowed masks and bypass ordinary memoization for caller-bound analysis. The future pure walk must derive all results and preserve per-invocation module identity.

I retain both earlier imported-shadow failures: the first declaration query returns bool instead of int; after an explicit typed Declaration binding, the same projection pattern in affine_facts.prepare gives bool to I64_NE. I give that query the same explicit binding. Every cache/query assertion and all surrounding query branches remain unchanged. A reduced three-file direct projection feeds the wrong type to I64_EQ while its explicit typed control passes. My first two reduction attempts refuse both controls; I preserve them without an independent causal claim. The product C compiler defect #1259 remains open.

I also retain ordinary global/array imported-shadow readiness from a separate exact7b75651234 snapshot, with all NanoLang source/tool guards. This does not run or replace their queued full qualification gate.

I add an ordinary-shadow unit target in source `3043c9ef248a156efe215761a4a5da579c9b621d`. Its actual Make invocation passes with the original immutable C producer,600-second wrapper/default10 shadows, a private OBJ_DIR and all source/tool guards. The reader verifies the additive Make suffix and source identity; I do not rebuild the compiler or run a full source gate in this control.

I verify this receipt with:

```sh
python3 docs/evidence/darwin-current-source-20261124/diagnostics/affine-analysis-cache-shadows-linux-20261125/read_archive.py
```

My manifest is `38d9168f866dfc6f39c97e20b522c77191ee2c9b867ab92f7fe369caef0cf456`; the reader verifies750 content records, exact source Git bindings and both typed-only changes. C differential, pure callee summary derivation/convergence, full source/generated-sanitizer/paired, CFG/private/executable/emission/cutover and5.1 release acceptance remain required. All earlier live/queued source roots and scripts remain unchanged.
