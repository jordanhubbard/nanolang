# Advisory module metadata transport

I retain ordered v2 METADATA key/value pairs through my in-memory module and
canonical assembly. This prerequisite does not implement local-name producers,
frontend proof facts, compute profiles, or high-level reconstruction.

I use existing string-pool indices and lengths, including empty strings and
embedded zero bytes. Duplicate keys remain ordered entries. I give the last
`nano.source_file` entry precedence for the legacy source-file view. My append
API updates that view; serialization refuses a contradictory independently
modified view. With no explicit source entry, I preserve the existing legacy
source-file behavior by synthesizing one pair. I reuse an existing key string.

I strip all source-file entries when explicitly stripping debug information,
while retaining unrelated advisory entries.

I preserve unknown keys without interpreting them. Advisory entries never grant
ownership authority, pure-call eligibility, capabilities, or execution support.
My typed ownership and passive verifiers remain authoritative.

I reject metadata-bearing v1 serialization instead of dropping entries. Ordinary
v1 modules retain their encoding. My canonical `.metadata` directive uses two
string-pool indices so every byte is carried by the existing escaped `.string`
representation. It appears outside functions, after its strings are declared.

Before acceptance I check v2 conversion and text roundtrips, source precedence,
duplicate order, exact bytes, existing ownership/passive facts, lifetime and
allocation cleanup, module-copy/link consumers, ordinary VM/native behavior,
and the genuine canonical compiler seed build. I keep every broader Phase20
acceptance row open.

## Shadow execution markers

Inside a function, `.shadow "target"` emits one `NOP` and retains the target
under the advisory key `nanolang.shadow.offset.<absolute-bytecode-offset>`.
I require a nonempty target without control bytes. My ordinary and owned
self-hosted shadow emitters place this marker immediately before each selected
shadow. With `NANO_SHADOW_TRACE` present, NanoVM reports
`I am testing shadow <target>` to stderr when execution reaches that NOP.
A failed shadow therefore does not report later unreachable shadows. With the
environment variable absent, the marker has ordinary NOP behavior.

These names are diagnostics, not verified source identity or permission to
execute. They neither grant capabilities nor change ownership admission.
Raw-bytecode consumers may author advisory metadata themselves. A bytecode
rewriter that relocates instructions must relocate or discard these offset
keys; copying them across a changed code layout cannot preserve trace meaning.
Native translation preserves execution semantics but does not currently emit
this optional diagnostic; my self-hosted compiler executes shadows in NanoVM.

## Declared function purity

I retain each emitted function's frontend `is_pure` flag under `nano.purity.v1`.
Its eight-byte value contains a little-endian u32 function-table index, one
byte (`0` for no pure declaration, `1` for a pure declaration), and three zero
reserved bytes. A synthetic function with no declaration carries zero when it
passes through the ordinary source emitter. Functions produced through other
paths may have no record. Zero does not assert that a function has effects.
Extern imports have no function-body entry; this convention does not encode
import annotations. Unconstrained generic pure inputs still fail the existing
purity check; admitted ordinary specializations retain zero.

My assembler accepts `.function_purity 0` or `.function_purity 1` inside a
function, once. Canonical assembly retains raw `.metadata` entries instead.
My lookup API returns `-1` for absent metadata, an invalid function index, a
malformed v1 record anywhere in the table, or duplicate records for the queried
function. Unknown versions remain uninterpreted. My append API refuses malformed
v1 tables and an existing record for the target function; allocation failure
publishes no partial metadata entry, though unused strings may remain.

These records describe frontend declarations. They do not certify inferred or
verified purity, permit calls, or replace the ownership/passive verifiers.
Stripping, forging or duplicating them cannot change executable behavior.
A linker or rewriter that changes function-table indices must remap or discard
these records. Exact module transport preserves them; general fact-aware linking
and reconstruction remain separate requirements.

## Generic specialization bindings

I retain one `nano.generic.v1` entry per bound type variable of an emitted
specialization. Its value starts with five little-endian u32 fields: the
function-table index, then byte lengths for the original declaration name,
source path, variable name and resolved type spelling. Those four UTF-8 strings
follow inline, without terminators or padding. The source path may be empty
when the producer has no filename; the other fields must be nonempty. No field
may contain NUL. The codec treats the counted strings as descriptive bytes; it
does not certify UTF-8 or parse the type spelling.

My C producer retains its checked `TypeInfo` spelling, including internal
nominal owner names. My selfhosted producer retains its resolved, whitespace-free
type spelling and recovers the original declaration name before module renaming.
These frontend spellings can differ. They preserve the frontend's identity;
they do not replace canonical layout identities or recover original type aliases.
Calls and contextual function values sharing one specialization share its
binding records. Ordinary functions have no generic binding record.

Inside a function, `.generic_binding "declaration" "path" "variable" "type"`
constructs an entry. Canonical assembly transports the raw `.metadata` bytes.
My lookup borrows counted fields and leaves its output unchanged on failure.
Unknown versions remain uninterpreted. A malformed v1 record makes lookup fail;
duplicate variables or inconsistent declaration origins for the queried function
make its records ambiguous. Missing records do not imply a complete binding set.
Appending refuses an ambiguous target binding, inconsistent origin or malformed
v1 record. Allocation failure publishes no partial entry; unused pooled strings
may remain available for retry.

I do not use these strings to authorize execution, substitute executable types,
choose layouts or transfer ownership. Stripping and forging them leave execution
unchanged. A function-table rewriter must remap indices or discard the records.
Because fields are inline bytes, string-pool relocation needs no hidden index
rewrite. Complete generic signatures, fact-aware linking and source reconstruction
remain separate obligations.

## Conservative frontend effect summaries

I retain `nano.effects.v1` independently of declared `nano.purity.v1`. Its
8-byte payload is a little-endian `u32` function index, one mask byte, and three
zero reserved bytes. The mask uses I/O `1`, mutable read `2`, mutation `4`,
unknown `8`, unsafe `16`, and resource `32`. Zero means my frontend analysis
closed an empty summary; a nonzero bit records a possible effect or an analysis
boundary, not proof that an effect occurs. An absent record is not a zero mask.

My lookup returns `-1` for absent, malformed, or duplicate records. I reject
out-of-range indices, reserved bits/bytes, and wrong payload lengths when
interpreting this key. I transport unknown versions without interpreting them.
My assembler accepts `.function_effects 0` through `.function_effects 63` inside
a function and canonical assembly retains the raw metadata representation.
Rewriters must remap function indices or discard these records.

My first producer slice reuses the purity checker's reachable-body analysis for
ordinary functions, including direct calls and recursion. I bound advisory work
at 64 source bodies and 4096 visited AST nodes; larger modules or exhausted
budgets retain unknown. My self-hosted producer also bounds local scratch slots
at 4096 and retains unknown for global identifiers without a symbol context.
My declared-pure checking remains unchanged and does not use this advisory
budget. These conservative summaries need not be equal outside the tested
ordinary subset: imported/global context, specialization, nested declarations,
affine uses, and match exhaustiveness remain separate completion work in #998.

I do not consume these records as verifier, native lowering, optimization,
handler, ownership, or runtime authority. Stripping, duplicating, changing, or
forging them must not change accepted execution or generated native code.
`test-function-effects-metadata` checks actual emitted summaries, canonical
round trips, VM/native execution, and those negative controls. Set
`NANOLANG_EFFECTS_COMPILERS` to colon-separated fresh producer paths to exercise
both the C seed and self-hosted stages.

## My checked match-site coverage

I retain `nano.match.v1` only as an advisory checker result. My payload is
little-endian: function index u32, line u32, column u32, coverage u8,
coordinate provenance u8, two zero reserved bytes, source-path byte count
u32, then the counted source-path bytes without a terminator or string-pool
reference. A rewriter must remap function indices or discard these records.

Coverage 0 is UNKNOWN, 1 is checked total integer coverage, and 2 is checked
total named-union coverage. I initialize the parsed field to UNKNOWN and
retain success only from my existing coverage checker. My known scrutinee
domain is not a coverage proof. Unchecked case lowering retains UNKNOWN even when its
separate defensive coverage/backstop checks succeed. My C return lowering may
invoke the real expression checker before emitting a match, and I retain that
checked result. Active generic instantiations remain UNKNOWN: I do not reuse a
shared template fact as a demonstrated per-instance check. I do not serialize a
claim that every function, arm body, guard type, or resource path is valid.

Coordinate provenance 1 means original C-frontend source coordinates.
Provenance 2 means positions in my self-hosted flattened parser input; the
path identifies the owning source when available, but the line is **not** an
original imported-file line. Empty paths explicitly mean unknown source.
Lines and columns are positive. Missing records are UNKNOWN, never evidence
that a function has no matches. Other producer families and unretained
specializations may omit facts. I make no collection-completeness claim.

I emit `.match_coverage line column coverage coordinates "source-path"` at
actual match lowering. The assembler stores the corresponding ordinary
metadata entry; canonical assembly transports the bytes through `.metadata`.
I reject duplicate site identities when adding records. Lookup rejects any
malformed known-version record and duplicate identity (function, path,
coordinates, line, column), leaving its output unchanged. Unknown versions
remain uninterpreted ordinary metadata.

I do not feed these facts into verifier, runtime, ownership, or optimization
admission. Stripping, forging, or duplicating them cannot make an invalid
program valid. My checked affine-use requirement and broader frontend-fact
requirement remain open under #998.
