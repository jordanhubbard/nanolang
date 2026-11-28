# My C ABI type selections

I port aggregate_value_tag, aggregate_kind_for_tag, c_result_type, result_is_i64
and c_local_type under #1142 after my storage-kind and scalar-carrier facts.
These select internal C spellings. I grant no checked module, payload, profile,
ownership or executable admission through a spelling.

My result entry accepts physical uint8 result count, tag and inferred array
kind, with checked integer-domain guards. I preserve the reference's order:
count zero or VOID returns void before the multiple-result refusal. A single
INT/BOOL returns int64_t; FLOAT returns double; STRING returns const char *.
U8/ENUM/OPAQUE/BSTRING/FUNCTION/CLOSURE use nmap_value, HASHMAP uses nmap_t,
and STRUCT/UNION/TUPLE use nrec_t. ARRAY uses its inferred storage kind:
word arrays use narr_t, string arrays nsarr_t, record arrays nrarr_t, and
other kinds refuse. Other result tags refuse. The separate i64 fact requires
count one and INT/BOOL; it does not follow merely from a chosen C type.

My local entry preserves all256 storage-byte decisions, including the default
int64_t for unknown and unrecognized bytes. My aggregate entry reports both
the exact STRUCT/UNION/TUPLE predicate and the reference's default TUPLE kind
for all other bytes. Neither fallback admits an unknown type or aggregate.

The reference observer compiles the five actual unchanged helpers and their
word-array dependency, with the real function-entry and ISA declarations. I
provide only the minimal array-results owner field used by these value queries;
I do not substitute a new C algorithm or call C from my NanoLang library.

My permanent matrix covers every256x256 result-count/tag pair with one fixed
array-kind and every256x256 result-count/array-kind pair at ARRAY, deduplicating
their256-row intersection. All256 local kinds and256 aggregate tags bring the
unique input count to131328. This covers each selector's complete branch input
domains; I do not call it an exhaustive256-cubed Cartesian result matrix.
Independent expected strings and ignored-array-context controls supplement
ordinary/sanitized actual C and all five NanoLang routes. Original600/60/
default10, malformed/prior-output boundaries and immutable guards remain.
My four permanent methods pass23.349s/all45 commands on Linux ARM64,
including6144 ignored-array-context controls and imported shadows with the
current qualified Stage2 producer. I retain [my checked receipt](evidence/c-repairs-linux-20261126/c-abi-types-20261127/README.md),
all selected source/tool/producer/receipt/index/six-host guards, the initial
missing-header observer refusal and the exact unchanged-header correction.
The reader checks4686 unchanged Git sources and replays962724 report lines
and all four method assertions without products. I still compose complete
module representation/declarations/body/runtime/profile/ownership/host
emission and qualify paired platforms, installed cutover and all69 release
requirements.
