# My public C literal normalization evidence

I tested reviewed decoder checkpointbae77456 and dependency/harness28c3f0d9 after
canonical PR755/752 integration63eb6069. My [manifest](public-c-literal-normalization.json)
records nineteen integrated source/harness/tool hashes verified unchanged, the
initial eighteen-hash freeze and ten retained logs. My [contract](../PUBLIC_C_LITERAL_NORMALIZATION_CONTRACT.md)
precedes implementation under cf8714a41ada4272b26da1db3b6450c2.

I share the previous nl_unescape_string body verbatim through an internal header
and retain its exported wrapper/ABI. Public C decodes an owned temporary before
emitting portable fixed-width byte escapes, then frees it. Allocation refusal
preserves the first error and existing path/stream output. Quote, backslash and
question-mark protection prevents C literal interpretation from changing bytes.
Explicit header dependencies cover the decoder and shared formatter.

| Integrated gate | Result |
| --- | --- |
| Literal GCC / Clang | 2 / 2 methods, 1.080 / 1.399 seconds |
| Adjacent equality/format/scalar GCC / Clang | 16 / 16 methods, 6.498 / 8.268 seconds |
| Existing public C backend | 7 programs pass, no skips |

Fresh source independently observes exact bytes for all existing escapes, unknown
escape spellings, UTF-8, control-byte/digit adjacency, trigraph-like text, empty
strings, distinct equal spellings and the first decoded zero. Public C executes
C99/C11 O0/O2 with ASan/UBSan/default Linux leak checks. Interpreter, verified VM
and corrected native C produce the same expected bytes. Isolated decoder/API
controls retain trailing backslash behavior and bytes after decoded NUL in the
owned decoder buffer, exercise every explicit staging allocation failure, check
zero live owned allocations, preserve path/stream output and recover in-process.

I preserve the initial full-method failure at native output: the public C,
interpreter and VM observations passed, while native C trigraph preprocessing
changed ordinary question-mark text. Separate merged PR755 repairs that boundary
and strict shared-provider references. I rebuilt current corrected tools and used
fresh source/module/output paths for final parity, without replaying the earlier
emitted binary. Integration changes outside my decoder are canonical PR752/755;
my public emitter/header/harness source stayed identical. No selfhost bootstrap
or Darwin qualification is claimed for this C-layer repair.

I match the current source decode+strlen contract. The visible string ends at its
first decoded zero; this does not add arbitrary length-bearing VM byte strings.
Full C portability6ade, concat ownership, remaining GNU conversions/blocks and
release remain open. Native children35aad/f172 are reconciled from canonical PR755;
this literal child awaits its own canonical merge.
