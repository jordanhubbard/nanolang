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
