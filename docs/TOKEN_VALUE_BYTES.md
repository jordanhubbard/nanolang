# My token byte-count prerequisite

I implement only the first reviewed prerequisite of
[NANOISA_FILE_PAIRED_SOURCE.md](NANOISA_FILE_PAIRED_SOURCE.md), under existing
8bbc. I add no service AST, execution or lowering.

My token values currently retain raw escape text. I keep those bytes unchanged.
`value_bytes` is the complete decoded byte count for literal STRING tokens,
including escaped NUL and its suffix; other tokens carry their raw value length
(NULL/empty is0). Unknown escapes retain both bytes, recognized n/t/r/0/quote/
apostrophe/backslash escapes occupy one byte, and a final backslash occupies
one byte. I do not implement Unicode escapes: UTF-8 bytes count individually,
and an unknown backslash-u stays literal. I share the C escape classification
with the actual existing decoder; Nano independently expresses the same rule.
Grammar body STRING tokens remain raw: their existing preceding GRAMMAR token
and raw-body constructor distinguish them from literals. This count is not a
claim that decoded Nano strings retain NUL, nor permission to use value_bytes
as an extent of the raw token string. Future consumers must decode and reject
NUL/extent mismatch explicitly. Unterminated literals still fail in the lexer.

I append the field to the primary schema and regenerate both C and Nano types.
I specify int64_t through a primary-schema C field override, mirrored by both
schema generators; Nano int is64-bit. This avoids narrowing the count to the
older C token position fields. A supplied bridge count is copied unchanged.
I update all explicit constructors, including the historical integrated Nano
compiler, parser EOF values and both token-list bridges/getters. Struct-by-value
ABI changes: old compiled compiler objects/modules must not mix with new lists
or generated types. These are internal compiler types, not a versioned wire
format or public File API. I require a clean C-seed/runtime/module build followed
by fresh Stage1/Stage2; I do not reuse old compiler executables against the new
runtime closure. Existing token/AST enum values and serialized formats remain
unchanged. Installed compiler/runtime providers must be upgraded as one build;
this checkpoint makes no binary compatibility promise for separately compiled
consumers of internal generated compiler headers.

Before gates I will submit complete production and paired nonexecuting fixtures:
exact scalar/raw/literal counts, every decoder escape, trailing slash, escaped
NUL followed by text, UTF-8, f-string static/expression copies, raw grammar
bodies, EOF, list grow/insert/set/remove/copy and getter preservation. Ordinary
parsing remains adjacent. C decoder output is compared with explicit byte spans,
not strlen. Paired Nano output observes counts/raw source only, not invented
binary string support. Schema regeneration, full constructor inventory, clean
bootstrap and all selected new helper shadows are required review/gate inputs.

## My source and fixture checkpoint inventory

My primary schema uses `c_field_types.value_bytes = int64_t`; both generators
honor that explicit C override. I regenerate compiler_schema.h and compiler_ast.nano;
no token/parse enum or ownership wire value changes. Active lexer construction,
raw grammar construction, parser EOF values and the historical integrated
compiler's explicit declarations/constructors carry the field. Both list
implementations copy complete structs and use owning sizeof, so their code
needs no field-by-field reconstruction. token_helpers exposes the retained
64-bit field with a header matching its actual List_LexerToken implementation.
The inactive lexer bridge now uses the actual lexer_main List_LexerToken return
and by-value getter, deep-copies text and preserves counts verbatim. Its NULL,
empty, array-allocation and both string-copy failure prefixes are fixture cases;
it is not activated as a compiler route.

My fixture deliberately passes a count above32 bits through grow/insert/set/
remove/pop/getter and bridge copies: copying metadata must not infer new facts
from the string. Actual lexers are independently checked against fixed expected
counts. C decoder tests compare full byte spans, including NUL suffixes and all
255 nonzero possible escape bytes; paired Nano helper shadows cover every
recognized escape, unknown u/x/q, trailing slash, escaped slash and UTF-8.
Grammar raw-body and f-string construction/copies have separate checks.

`tests/test_token_value_bytes.py` rebuilds five C providers per configuration,
runs the direct decoder/list/bridge fixture and unchanged f-string neighbor,
and compares full observable rows through the three explicitly fresh compiler
paths. It compiles the actual Nano schema generator and requires all selected
shadows (including the new override check); it does not run its writing main
against the immutable source tree. Exact selected-name multisets come from the
complete retained import graph, not sample matches. The inherited file-backed
runner preserves first terminal, process-group cleanup and empty LSAN_OPTIONS.
Sanitizers cover the named C providers/fixture, not the entire compiler.

Future gates require clean Linux/puck C-seed/Stage1/Stage2 builds and actual
ordinary parser/module/wrapper adjacency after this internal ABI change, plus
Linux GCC/Clang ordinary/sanitizer C checks and puck Apple/Homebrew ordinary and
Homebrew sanitizer C checks. Existing compiled module caches are not reused.
I have only regenerated source text and inspected diffs; no new build, shadow,
fixture, bootstrap or service execution has run. Complete checkpoint review
precedes those gates.
