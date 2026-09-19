# I retain required file-service facts before execution

I record `task_6833551e538141a285dc059164c7a2d9` under file boundary6931,
related to d03c and ed702, from canonical `3f91cd7e3` after PR814. This is a
contract-only checkpoint. I propose no source syntax, service opcode, runtime
binding or execution admission. My [File ownership contract](NSI_FILE_OWNERSHIP_BOUNDARY.md)
and qualified private plan remain the semantic source of this finite catalog.

## My existing transport and consumers

`nvm_format_v2.h` defines required feature bits0..8 (known mask0x1ff), sections
0x01..0x0d and imports FFI0, coprocess1, artifact2. The header reader rejects
unknown feature bits; the directory validator rejects unknown section types,
duplicates, reserved flags and ranges. This supports an additive required feature
without changing the v2 format version or ISA instruction version.

`nvm_v2_module.c:required_features/build_plan` derives flags from contents and
emits sections canonically. Ownership/passive payloads already require exact
presence/feature agreement, but its plan currently has13 entries. The new section
must participate in size calculation, directory/CRC and cross-section validation;
merely increasing a known mask would silently erase facts in existing conversion.

`NvmV2Module` has borrowed raw payloads plus owned decoded tables; `NvmModule`
owns retained ownership/layout/passive bytes. `nvm_v2_convert.c` bridges them,
while legacy `nvm_serialize` rejects facts it cannot retain. I must cover free,
copy, both bridge directions and the v1 refusal before admitting the new feature
as readable. The in-memory NvmModule is not itself a promise of v1 wire support.

METADATA is free-form string-pool key/value data. It remains unsuitable for
execution-required owner transitions. `nsi_gen.c:nl_nsi_gen_nanoisa_imports`
currently prints demonstration IMPORT text with return=int and TRAP cap lines;
it does not carry this contract. Source extern/import paths and owned profiles
remain the refusals identified in the File contract. The ordinary verifier,
owned verifier, VM FFI loader, native C emitter, LLVM/Wasm and reconstruction
must not reinterpret a service binding as generic FFI merely because its category
signature uses existing tags.

## My smallest finite wire proposal

I reserve, subject to pre-code review and rechecking canonical assignments:

| Item | Proposed value |
| --- | --- |
| `NVM_V2_FEATURE_SERVICE_BINDINGS` | bit9, 0x00000200 |
| `NVM_V2_SECTION_SERVICE_BINDINGS` | 0x0e |
| `NVM_V2_IMPORT_SERVICE` and in-memory counterpart | kind3 |
| Payload encoding version | 1 |
| Catalog identity | 1: immutable local-file catalog from PR814 |

The section is little-endian, exactly56 bytes in this first family:

| Offset | Width | Meaning |
| --- | ---: | --- |
| 0 | 2 | encoding version1 |
| 2 | 2 | catalog1 |
| 4 | 4 | count5 |
| 8 | 4 | flags0 |
| 12 | 4 | reserved0 |
| 16 + 8*i | 4 | operation ordinal i, exactly0..4 |
| 20 + 8*i | 4 | import index for that operation |

Ordinals are temp, write_byte, rewind, read_byte, close in that order. The catalog
version identifies the entire immutable contract, not just names: exact interface,
resource/Result identities, parameter modes and lifetime, required/acquired rights,
Ok/Error owner postconditions, scalar error fields, checked INT byte domain and
scalar-only public escape restriction. I reuse one authoritative definition of
those facts; transport cannot carry a mutable caller-selected alternative catalog.
Any semantic change needs a new catalog version and independent review. Encoding
version changes are separate from catalog semantics. Unsupported either refuses.

I do not duplicate a generic type/ownership language in these56 bytes. The section
is a required exact catalog reference with import-index bindings. It can describe
this family only; it is not a general NSI transport or closure of ed702. An absent
section means no service catalog, never a default catalog. A zero-count or empty
section is invalid. Exact size and reserved fields reject trailing data or hidden
extensions. Five distinct import indices must bijectively reference the five
imports in this bounded family; other imports/callbacks/module links are refused
for service-bearing modules in this checkpoint. Unrelated old modules keep their
existing rules.

Each referenced import has kindSERVICE, exact module string
`nsi:nanolang/filesystem` and exact full method-ID string from the catalog. I compare
stored string length and bytes, refusing embedded NUL or spelling normalization.
Its SIGNATURES entry has exact positional category tags, no upvalues or hidden
arguments: temp()→UNION; write_byte(STRUCT,INT)→UNION;
rewind(STRUCT)→UNION; read_byte(STRUCT)→UNION; close(STRUCT)→UNION.
These tags are transport categories only. STRUCT is not authorization to pack a
File from fields; UNION is not authorization to copy its selected owner. The
catalog retains the nominal identity and ownership relation. Future executable
admission additionally needs exact runtime layout/opaque-provenance and selected
variant facts; this transport checkpoint does not invent complete ordinary
layouts to bypass those prerequisites. No new ISA value tag/opcode is allocated.

## My atomic private APIs and retained ownership

I propose a standalone payload decoder/validator returning a fixed-size value
containing version/catalog and the five import indices. It checks the entire
payload before publishing. It does not allocate or borrow input memory. The
encoder stages/validates first, leaves output and size unchanged on failure,
and emits canonical bytes. Size-only query returns56 only for a valid value.
Unknown version/catalog, wrong count/ordinal/reserved fields, duplicate indices,
short or long input all refuse. Cross-section validation is a separate checked
query because the raw codec cannot validate imports/signatures/strings.

A private builder takes a successfully validated `NlFilePlan` and five explicit
import indices, plus the destination module for cross-checking. It cannot accept
an arbitrary catalog struct or infer an import by sanitized symbol. It constructs
and validates bytes privately, then attaches one owned exact payload atomically;
allocation failure or an already-present different payload preserves the module.
No document storage is retained. The builder grants transport only, not runtime
host registration or a callable generated binding. Identical reattachment can be
specified as checked no-op; no replacement of live unrelated facts is allowed.

`NvmModule` owns its service payload; freeing the module frees it. Conversion to
NvmV2Module may borrow it under the bridge's existing source-lifetime rule.
Decoded NvmV2Module borrows the input section bytes; conversion to NvmModule
copies them with checked failure cleanup. A failed full bridge does not publish
a partially built module. Serializer/deserializer preserve exact catalog bytes
and import indices. The attach API is transactional even though some historical
whole-module APIs have their own established failure conventions; I do not
silently change unrelated API output rules.

## My required-feature and compatibility rules

I require exact equivalence: service payload present, required feature set, and
the five service imports all appear together. KindSERVICE without the section,
section without its flag, flag without nonempty section, unrelated flags/versions
inside the payload, downgraded FFI kind and wrong signatures/identities refuse.
A module without service facts emits no new feature or section. Existing byte
corpora remain byte-identical, including older required ownership/layout facts.

Old readers reject bit9 before decoding this module. If a caller strips bit9 but
leaves the new section, old directory validation rejects0x0e. If it strips both
but leaves imports, kind3 must be rejected by old import consumers rather than
become generic FFI. This is fail-closed versioning, not authentication against an
adversary rewriting an entire module into a different one. Existing resource
verification and runtime host registration remain necessary.

New container readers may inspect/roundtrip the declared module, but common
executable verification rejects service-bearing modules explicitly until the
later matched lifetime/runtime contract lands. Owned/scalar/managed profile
selectors inherit that refusal. VM foreign entry points also explicitly reject
kindSERVICE without a registered reviewed service dispatch; no fallback to raw
FFI lookup. Native C, LLVM, Wasm conversion, source reconstruction, disassembly/
assembly and linking receive a source audit: each either retains the section
exactly through a non-executing path or refuses before output publication. Initial
link/assembly/reconstruction support may refuse this new feature; silent loss is
never accepted. I do not require full new opcode reconstruction as a side quest.
Legacy v1 serialization must refuse the payload and kind3, without erasing it.

I keep schema/parser/generator-dispatch APIs unchanged. The private plan is the
only NSI input to the private builder. Existing demo generators do not begin
emitting callable service code. Paired source integration remains a later reviewed
phase that must emit exactly this contract and qualify normal shadows.

## My dependency-ordered acceptance

1. I review and implement the fixed raw codec and private non-admitting query,
   with exact56-byte golden bytes, full field/length/endianness/limit mutations
   and output preservation. I test data, not a service or failed binary.
2. I review the combined module retention/bridge and required-feature/import-kind
   changes with common execution refusal before exposing the new known feature.
   No intermediate commit makes it executable or silently droppable. Cover
   header/directory/CRC, feature/section/import mismatch, signatures/identity,
   duplicate/out-of-range imports and allocation-prefix cleanup. Preserve old
   feature bits, ordinary FFI/artifact/coprocess and unrelated module outputs.
3. I connect only the private plan builder, verify exact roundtrip through raw
   codec, full container and both in-memory bridges, input lifetime independence,
   and v1 refusal. Test both import orders via explicit index remapping; no linker
   silently remaps these bytes. Audit all named shipped consumers and test their
   actual checked refusal/output preservation without service invocation.
4. I qualify strict native builds, sanitizer allocation/error controls, existing
   NSI private-plan and v2/ownership/profile regressions with frozen source/tools.
   First failures are retained. No new producer/bootstrap/public service claim.

After this child, opaque File and affine Result layouts/flow/cleanup plus matched
VM/native service dispatch remain genuine execution prerequisites under6931.
Socket catalog/owned pair, GPU, representative library migration and the original
d03c/ed702 obligations stay open. I have not implemented or executed this proposal.
