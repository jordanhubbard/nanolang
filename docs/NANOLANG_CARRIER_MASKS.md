# My scalar-carrier facts

I port the three adjacent nvm2c.c scalar-producer/carrier predicates under
#1142 after my storage-kind component. These are internal representation facts,
not evidence of a checked payload or executable admission.

I preserve scalar_kind_tags for every physical uint8 kind. The exact mappings
are storage int0 to mask2, string1 to32, bool9 to16, float11 to8 and function13
to2048. Every other byte maps to the0x8000 unknown marker. These are canonical
tag masks, not my private storage IDs or shape kinds.

I preserve boxed_carrier_tags over all65536 masks: the existing marked
variant-payload predicate, plus exactly int/u8 and int/float pairs with optional
VOID. Boxed storage does not establish any payload kind beyond those inputs.
My optional predicate requires VOID, exactly one payload bit and no bits above
TAG_BOOL. I preserve that existing bound: string, enum, function and other
optional masks do not satisfy it. A later consumer owns payload checking.

My NanoLang entries reject values outside their physical byte/word domains,
rather than truncating them. I reuse the pure storage-kind payload predicate.
An observer compiles the actual unchanged three C helpers and their two
variant-mask dependencies, with the real ISA constants, from source bytes.
No C helper or FFI implements my NanoLang library.

My permanent gate compares all65792 physical inputs through ordinary and
sanitized actual C and five NanoLang routes, retain independent exact sets,
unknown/malformed/prior-output boundaries and imported shadows, and preserve
600/60/default10. My three permanent methods pass12.280s on Linux ARM64 with all38 commands
and the current qualified Stage2 producer. I retain [my checked receipt](evidence/c-repairs-linux-20261126/carrier-masks-20261127/README.md),
source/tool/producer/receipt/index/six-host guards, exact packet/report bytes
and independent replay of460810 report lines without product execution.
Complete module representation/emission, paired platforms, installed cutover
and all69 release requirements remain separate.
