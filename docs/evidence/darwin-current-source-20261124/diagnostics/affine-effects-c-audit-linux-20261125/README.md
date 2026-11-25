# My effect-step C expectation audit

I retain source `c3e6017fb5235bac66e25aa1e8afb82b6f26c138` under #1142.
My pure NanoLang helpers implement HANDLER_PUSH165's existing no-op step,
HANDLER_POP166, PERFORM167, operation-key equality and current-function escape
recording. This receipt qualifies diagnostic expectations against unchanged C;
I still require execution of the new NanoLang source.

All878 actual C commands pass in3.202 aggregate command seconds. My1001 original
frame/flow/FIFO reports retain SHA256
`f5f4555c26ea20dbc13237f4a9843c42e98b16f7fc6f4e9f9ea3de7b3d3dcbd0`.
My8128 new reports retain SHA256
`0521f0169aa763ebe53c45311cb89ff343c42589d22deea3b3cac3f5a257cba9`.
I compare complete decisions/messages, stack values and exclusions, globals,
handler floor/activation/order, mutable authority and escape vectors.

My cases cover raw key bytes/aliases/NUL suffixes and absent indices, caught
versus escaping operations, duplicate/capacity/eight-function boundaries,
inherited handler floors, all physical argument/prefix tags and authority
flags, canonical retained signatures, original256-parameter/stack limits,
VOID/scalar/STRING/copyable/resource record/union/structurally equal nested-array
arguments and results, local owner/region/borrow obligations, error order,
full-stack rollback and malformed carriers. Seeded bounded escape vectors test
private defensive behavior; I do not admit them as valid module summaries.
The direct C key/escape helpers retain their nonnull-module precondition.

I reuse the unchanged embedded C step and state diagnostic. Their source
replaces only the ordinary bytecode/state objects; every remaining original
bootstrap object, selected source, production C and C tool hash matches
before/after. No product C file or old test/recipe body changes. Strict C99
syntax, complete-driver typed AST, shadow presence and diff checks pass. I
retain the first library-only typed-AST invocation's missing-main diagnostic;
the recorded successful AST command checks the complete driver and executes no
NanoLang shadow.

I independently read11848 records with:

```sh
python3 docs/evidence/darwin-current-source-20261124/diagnostics/affine-effects-c-audit-linux-20261125/read_archive.py
```

My manifest SHA256 is
`92d5bf4eb1a30e099d149074979983ae235c0de2fdf68d235ea8f44dae14d66c`.
Original8-case/600-second and new32-case-or8KiB/60-second batches remain.

Executed NanoLang shadows, full source/VM/native/generated-sanitizer and paired
qualification remain pending. I preserve the live corrected foundation and
unexecuted global/array order; no second heavy gate starts. Handler CFG edges,
activation/resumption/call summaries, global-effect proof, executable/profile,
pure C-emission/cutover and whole-release acceptance remain required.
