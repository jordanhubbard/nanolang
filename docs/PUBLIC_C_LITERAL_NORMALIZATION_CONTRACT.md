# My public C literal normalization contract

I execute task_cf8714a41ada4272b26da1db3b6450c2 on main44ad5f69 after PR753.
My earlier static audit, not a replayed failing program, establishes that parser
AST_STRING retains raw escape spelling. Canonical NanoVirt calls nl_unescape_string
and stores strlen(decoded); public C currently re-escapes the raw spelling.

I share the existing nl_unescape_string algorithm through a small internal header
implementation, keeping its existing exported wrapper in eval.c. Both canonical
callers and the public C emitter therefore consume one decoder without changing
symbol ABI, module link manifests or the meaning of an existing escape. Newline,
tab, carriage return, zero, slash and quote escapes preserve their current meaning;
unknown escapes and a trailing backslash retain their current literal bytes.
I do not silently adopt C hexadecimal/octal/Unicode escape rules.

My public C emitter decodes each literal into an owned temporary, emits its visible
NUL-terminated prefix as a C string token, and frees that temporary on all paths.
I explicitly match the current source routes' strlen boundary: a decoded zero ends
the visible string even if later decoded bytes exist. This is not arbitrary VM
length-bearing byte-string transport or a new embedded-NUL language contract.
The source-wide length-bearing representation question remains outside this repair.

I emit control/high bytes with fixed-width octal C escapes and protect quote,
backslash and question-mark/trigraph spellings. Adjacent digit characters cannot
extend an escape. I preserve UTF-8 bytes without relying on a C compiler's source
encoding conversion. Allocation failure produces a first retained diagnostic before
path/FILE publication; later valid same-process emission must recover. Both dry
planning and final emission use the same normalization and free owned temporaries.
A handcrafted absent AST string retains the previous empty-string behavior.

Before production I record this roadmap/MAC contract; before fresh execution I send
a production checkpoint. I freeze complete source/harness/tool identities and run
GCC/Clang C99/C11 O0/O2 ASan/UBSan/leak controls. New ordinary source covers each
existing escape, unknown sequences, quote/backslash, UTF-8 bytes, trigraph-like text,
control-byte/digit adjacency, first decoded zero and distinct equal spellings.
Independent expected bytes plus interpreter/verified VM/native observations check
source parity; direct decoder/API controls cover trailing slash and deterministic
allocation failure, previous output and recovery. Existing equality, formatting,
scalar and backend gates remain acceptance. No historical failed artifact runs.

I preserve full C portability6ade, concat ownership and remaining GNU conversions;
this source normalization does not close those required scopes or full release.
