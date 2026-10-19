# My bounded owned string and print reconstruction design

I own this design-only #999 slice on `design/5.1-owned-string-reconstruction`,
from aggregate `1e13109d19f9a6436173b341198edbc69109be73`. My prior loop component
remains immutable on its pushed branch. I propose no source compiler, verifier,
wire schema or runtime change. Implementation and baseline execution await review
and a coordinated slot.

## My existing executable anchor

I retain `examples/language/nl_affine_resource_demo.nano` unchanged. It declares
FileHandle with one INT field, main()->INT, open_file(STRING)->FileHandle and
close_file(FileHandle)->VOID. The existing producer places main at index0 and
retains the other declarations in source order: three ordinary functions. Its
separate selected-shadow module adds the synthetic entry and retains all three
original functions: four functions. Both fit the existing eight-function bound.
I treat the ordinary module and separately emitted original-shadow module as distinct reconstruction inputs, and require their own exact stdout parity. I will measure exact signatures, local counts and constants in the baseline;
source declarations and producer routing are not a fresh executable measurement.

`tests.test_source_borrow_emission.test_owned_string_unchanged_affine_example`
uses `graph_positive` to compile through NanoVirt and both installed stages,
compare canonical dumps, strip names, execute VM/native, and separately execute
all original selected shadows. Ordinary output is exactly
`audit.log\nresource closed\n`; original-shadow output is exactly
`audit.log\nresource closed\nresource closed\naudit.log\nresource closed\n`.
The retained integrated42-method evidence at e56f75bf in
`evidence/source-owned-string-print.md` establishes that historical accepted
source path, not fresh qualification at my base.

My runtime/source authorities are `NANOISA_OWNED_STRING_PRINT.md` and
`NANOISA_SOURCE_OWNED_STRING_PRINT.md`. They already distinguish immutable
STRING parameter/literal values from resource ownership, retain exact acyclic
value calls and PRINT/PRINTLN, reject embedded NUL in this profile, and preserve
normal/assertion-failure resource cleanup. I do not invent STRING fields,
results, new STRING local declarations, reference mixtures or general operations.

## My proposed facts interface

Original `nvm_verify` admission remains mandatory before facts extraction.
Complete layouts, signatures, call graph and existing counts remain checked.
In the existing consuming value graph only, I allow exact STRING/NO_INDEX
mode-zero parameters and their corresponding initialized parameter-local slots.
Other STRING locals, STRING results and resource fields still refuse. STRING
parameters use a separate immutable parameter-expression map, not Function.locals
(the ordinary emitter initializes that map as named mutable locals). Every
STORE_LOCAL targeting STRING, including its parameter slot, refuses so direct
parameter-leaf reuse cannot reorder mutable reads. Borrowed
profiles do not inherit any string or print admission. I preserve exact positional
INT/BOOL/owner/STRING arguments and nominal identities.

The current facts carry PUSH_STR's numeric operand but no executable constant
bytes, and reject STRING parameter/local tags. I propose an additive facts member
only for modules using this new slice: a table keyed by original constant index,
with explicit byte length and hexadecimal bytes for referenced executable string
constants. I use existing `nvm_get_string`/`nvm_get_string_len` accessors after
verification; I do not decode the wire format again or derive length with strlen.
I require every referenced index exactly once, exact length/hex consistency and
no embedded NUL. Advisory names do not become executable literals. Modules outside
this slice serialize exactly as before. This is a proposed private facts interface,
not a wire-schema edit or implementation in this design commit.

I propose an explicit reconstruction-only literal payload cap of1MiB total unique
referenced bytes, not a new runtime or language limit. Unique referenced indexes
cannot exceed the existing4096-instruction module bound (nor the actual verified
constant count). Repeated PUSH_STR uses reuse one table entry, not another byte
copy; equal bytes at distinct original indexes retain distinct identities. Before hex encoding, I sum lengths with checked size arithmetic and refuse
if any addition exceeds the cap; encoded payload is exactly twice that sum and
at most2MiB, plus bounded per-entry index/length JSON overhead. Empty literals
consume a table slot but zero payload bytes. Python independently checks the same
entry count, byte sum and exact hex length before decoding, and rejects duplicate
indexes even when bytes match. No count/length/byte truncation is allowed. The parent approves this
new conservative output-profile refusal, with exact boundary/over-limit atomic
refusal and malformed length/count/duplicate controls;
it does not claim an existing1MiB module limit. Original uint32 lengths come from
the verified module accessor, never a narrowed host strlen result.

## My IR and lifetime boundary

I add immutable string-literal and string-parameter expressions, and explicit
ordered print statements whose newline flag is exact. PUSH_STR pushes one exact
STRING expression; LOAD_LOCAL may copy only its initialized parameter. PRINT or
PRINTLN consumes exactly one STRING and pushes nothing. A value CALL accepts the
exact positional type and evaluates/materializes every actual in original order.
Immutable string copies never enter the live-owner/origin map, consume an owner,
resurrect an alias or substitute for a nominal resource operand. OWN_PACK/UNPACK,
MOVE, STORE, return cleanup and ASSERT retain their existing checks.

Only admitted immutable literals and parameter copies may remain as expression
leaves. Effectful calls, prints and assertions remain explicit statements; a
string forwarding operand is not reevaluated across an intervening owner call or
print. No generic scalar whitelist widening may admit STRING arithmetic,
comparison, conversion, STORE into another local, branch phi or return. Resource
owners remain subject to the complete original affine graph and reconstruction
proofs, including successful terminal cleanup. A failed ASSERT is a trap, not a
successful owner-return path.

## My output representations and byte order

For C I propose immutable static unsigned-byte arrays plus exact-length views,
passed by value. Empty strings have valid zero-length storage. PRINT uses the
exact length and PRINTLN adds exactly one newline, with no allocation/free or
format-string interpretation. Lifetime is static for the whole generated program;
resource cleanup never frees string storage. I preserve the existing output/error
boundary rather than invent durable-I/O semantics. Failure tests must capture
actual bytes written before ASSERT; no arbitrary flush changes are assumed.

For Nano I emit ordinary exact string parameters and literals, and checked print
statements. I must use a byte-preserving literal encoder matching the actual
source parser: recognized escapes for quote/backslash/control characters and
literal UTF-8 where valid, without host locale conversion. Unknown escapes, literal backslash followed by zero,
non-ASCII and control bytes need actual roundtrip tests. I must refuse bytes the
source encoding cannot represent before publication rather than decode with
replacement or silently alter bytes. Invalid UTF-8 acceptance is not assumed.
The proposed hex facts preserve bytes even when an output target refuses them.

All original field/call/assertion effects keep order. C and Nano need not simulate
VM string root counts: static native views and ordinary checked Nano parameters
implement the admitted immutable values, while original/regenerated VM lifetime
and native sanitizers provide independent execution evidence. There is no hidden
interpreter, dispatcher, added runtime authority or new ownership kind.

## My shadows and finite limits

Reconstruction does not recover original shadows from an ordinary executable.
I retain that existing disclosure. The current `OwnedEmit.run` only appends explicit owner consistency shadows for
borrowed profiles; the consuming value-call output delegates ordinary function
emission to `Emit.run` and does not synthesize a helper witness. Existing consuming
roundtrip tests compile that ordinary output directly. I preserve that behavior
rather than import borrowed zero-owner witnesses into this STRING profile. If a
regenerated selected-shadow product is requested, I measure its actual existing
compiler behavior separately and do not claim recovered original shadows or
checker-proved helper coverage. The independently produced original-shadow module
is executable input in its own right; reconstructing it preserves its calls and
prints, not the absent source shadow declarations. No arbitrary empty-string
witness, no removed activation and no weakened assertion is authorized.

The current Nano renderer reserves one of eight functions for selected shadow
execution. The anchor has three ordinary functions and no arithmetic helper need,
so its source cardinality is feasible; actual generated entry/helper/shadow local
counts still require measurement. The128-local,4096-instruction and existing
function/graph limits remain unchanged. All new print/materialization storage must
be counted on the actual emitted body, with ordinary atomic refusal at a bound.
C and every old scalar/owned/borrowed profile must remain byte-identical when no
new string/print constructs occur.

## My staged acceptance and refusals

First, after review/slot release, I retain exact unchanged source/provider hashes,
ordinary and original-shadow module dumps, original verifier/VM output, current
facts/reconstruction refusal and unchanged prior outputs. I then measure all
ordinary reconstructed functions and selected shadows before a broad corpus.
I do not count a preflight bypass as ordinary support.

A later implementation gate requires original VM/native versus reconstructed C
and regenerated Nano VM/native, strict O0/O2 ASan/UBSan, exact captured bytes,
unchanged source/shadows, stripped names and binary/text roundtrips. Controls cover
empty/nonempty and escaped/non-ASCII bytes, repeated/sibling/three-frame forwarding,
mixed positional scalar/string/owner actuals, prints before successful cleanup
and false ASSERT, and no duplicate/reordered output. Constant bytes must survive
canonical transport without dependency on advisory naming.

Original-verifier negative controls cover wrong string argument/print tag,
constant index, result count and ownership identity. Defensive facts separately
cover malformed hex/length/index tables, absent constants, misplaced STRING local
or result, embedded NUL and unsupported source encoding. Refused output preserves
the previous file byte-for-byte. STRING fields, local mutation, return, collections,
operations, borrowed CALL_REF mixtures and unsupported graphs remain explicit
refusals. I retain old output goldens and actual cap boundaries. Fresh platform
qualification and full required #999 remain open after this bounded slice.
