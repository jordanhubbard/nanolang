# My multi-instance service transport

I use service-binding version 3 to retain multiple nominal File and TCP
catalog instances in one NanoISA v2 module. I preserve the existing version 1
and version 2 encodings. My new table describes metadata; it supplies no host
authority and does not admit mixed executable code yet. I track flow, dispatch,
source lowering and grant integration separately under #990.

## Wire table

I encode integers in little-endian order. My header is 16 bytes:

| Offset | Width | Meaning |
| --- | --- | --- |
| 0 | 2 | Service-binding version, exactly 3 |
| 2 | 2 | Reserved, zero |
| 4 | 4 | Instance count, 1 through 64 |
| 8 | 4 | Instance entry size, exactly 64 |
| 12 | 4 | Reserved, zero |

Each 64-byte entry has the following fields:

| Offset | Width | Meaning |
| --- | --- | --- |
| 0 | 4 | Instance ordinal, equal to its table position |
| 4 | 2 | Immutable catalog identity: 1 File, 2 TCP |
| 6 | 2 | Reserved, zero |
| 8 | 20 | Five import indices, in catalog method order |
| 28 | 36 | Nine layout slots, in catalog type order |

File has eight nominal types. I require its ninth slot to be `UINT32_MAX`.
TCP uses all nine slots, including Endpoint. Every active import index is
unique across all instances; every active layout index is likewise unique.
I forbid `UINT32_MAX` in an active slot. Import and layout indices belong to
separate tables and may have equal numeric values.

My exact extent is `16 + 64 * instance_count`, with no trailing bytes. The
largest table is 4,112 bytes. These are explicit implementation limits, not a
claim that every possible service graph fits. I refuse larger tables before
allocation. I do not allocate during raw encoding or decoding. Encoding uses
a bounded staging buffer so input and byte output may overlap; decoding uses
a staged value. Failure preserves output buffers and the caller's size value.
The size pointer must be disjoint from input and output storage.

## Nominal identity

An instance ordinal distinguishes repeated declarations of the same catalog.
For each instance I require its exact five methods, signatures, named layouts,
member tags and prior-only layout edges. A member's nominal target must belong
to that instance. A same-shaped File in another instance cannot be substituted
into its OpenResult. Layout and import table order may differ from catalog
order; the explicit indices retain identity through either order.

I validate ownership-v1 flags and function descriptors against the complete
combined map. Each instance's owner and owned-result layouts carry the exact
resource flag; borrowed parameters identify that instance's owner. Unmapped
layouts retain no catalog authority. My query returns instance, catalog,
catalog ordinal, global layout index and per-kind source ordinal separately.
The returned plan owns its metadata and remains valid after the input module
is freed. I retain the 65,536-layout bound from the single-catalog validators.

## Module boundary

I validate the complete map before retaining service ownership through either
module bridge. Serialization and deserialization preserve service, layout and
ownership bytes. Module feature bits and ordinary structural validation still
apply. An unknown or malformed version cannot fall back to a different codec.

My general verifier, VM and native translator still refuse service execution.
Existing explicitly granted File and TCP execution remains single-catalog;
this table does not combine their grants. My checked mixed-flow queries are described below. Runtime values,
cleanup, paired source lowering and supervised publication remain necessary
before a mixed source program can execute.

I test mixed File/TCP/File modules, repeated-catalog identity, import/layout
permutations, cross-instance payload substitution, exact ownership, every raw
truncation at the maximum extent, single-byte mutations, overlap, malformed
serialized metadata with corrected CRCs, allocation failure and both bridge
directions with `make -f Makefile.gnu test-multi-nominal`. Sanitizers cover the
new codec/query and transport adapters; linked surrounding objects retain
their ordinary build flags. This is not a fully instrumented whole-program run.

## My checked mixed-flow queries

I instantiate the existing ownership, CODE, body, cyclic and indirect engines
with `services_flow_config.h`. File and TCP retain their original catalog
configuration. The mixed configuration resolves every member and method through
the nominal instance map; it does not search all instances for a matching name.

My private flow API uses `instance * 9 + type` for type identities and
`instance * 5 + method` for method identities. A File instance's unused ninth
type slot remains invalid. These identifiers differ from the global layout or
import indices, the wire catalog IDs, and the per-kind source ordinals used by
aggregate instructions. I retain all of those distinctions in the query facts.
The historical `catalog_ordinal` field name in the shared report shape carries
this private instance-qualified identifier in `NvmServices*` reports.

Calls and borrows compare exact declarations, including their global nominal
index. Acquiring and consuming services use the selected instance's owner and
Result. Result payloads and constructors resolve within that instance. TCP
begin-connect keeps its Endpoint-domain obligation; every byte-write method
keeps its byte-domain obligation. Host rights, liveness and cleanup obligations
remain pending. A logical call or hosted query never discharges host authority.

The hosted preflight bounds all declared imports, including bridge-array growth
and parameter storage for up to 320 imports. I preserve the existing function,
state, graph and allocation limits. The copied hosted plan owns its input bytes
and metadata; it remains usable after the input module and buffer are destroyed.

`make -f Makefile.gnu test-services-flow` checks mixed File/TCP/File programs
through direct and indirect owned/borrowed calls, all catalog methods, Ok/Error
branches, loops and permuted maps. I also check a full 64-instance table,
wrong-instance calls, service operands, result storage and error constructors,
failed-transition preservation, query lifetime and allocation failures. These
are synthetic logical programs; Endpoint values still require runtime domain
checking. My mixed runtime, grants and paired source product remain open.

## Private mixed value lifetime

I retain one independent File or TCP lifetime core per instance in
`src/nsi_services_values.{h,c}`. My trusted, serialized host caller supplies a
copied catalog table of at most 64 entries. This table alone is not a checked
module, a host grant, or permission to publish source execution.

My value and borrow carriers retain a one-based instance identity, a catalog
tag, and the corresponding core identity. I check the target instance before
calling a service. The core still checks invocation, generation, ownership and
borrow epochs. I retain File and TCP result representations separately; their
status numbers, cleanup fields and scalar result kinds are not interchangeable.
Repeated File instances have independent capacity and lifetime state.

I stage result publication after accepted core operations. A failed checked
operation preserves caller outputs; an accepted host error remains a typed
Result. Move and take-Ok consume their source, close consumes either host Result
arm, and a live borrow prevents ordinary move/drop/close. Terminal cleanup
reclaims every instance, including abandoned borrows and unhandled Results.
I cache the first finish report and preserve per-instance cleanup records.
The aggregate cleanup count counts those records, not distinct host closes:
TCP retains both a close error and its terminal ambiguous-close history.

My [mixed lifetime evidence](evidence/mixed-service-values-20261010/README.md)
qualifies this private carrier. My next runtime integration must derive the
catalog table from the retained checked nominal plan, preserve instance IDs
through frames and VM/native calls, enforce host grants before acquisition, and
publish scalar results only after clean cleanup. This carrier does not establish
those dispatch, source, grant or release requirements.

## Private mixed checked execution

I connect the mixed lifetime carrier to the shared checked runtime and the
indirect VM/generated-native engines in `services_runtime.c`,
`services_vm_indirect_private.c` and `nvm2c_services_indirect_private.c`.
I derive the immutable instance catalog table from the retained checked plan.
My current two-catalog schema identifies the optional ninth Endpoint type as
TCP; an eight-type instance is File. This inference follows complete nominal
validation, not caller-provided names or unchecked layout counts.

I retain instance-qualified type and method IDs in constructors, Result
payloads, service arguments and results, borrowed formals and owned calls.
I compare owner identities and exact live/borrow masks separately for each
instance. Two instances can both occupy core slot zero without becoming the
same owner. I reserve a conservative core-storage bound for 64 instances and
include the concrete runtime arrays in the existing runtime byte limit.

For cyclic and indirect frames, I match every service request to the current
checked instruction and import before calling a host operation. I retain fuel
charging before effects, checked call/return staging, and cleanup before scalar
publication. Generated C embeds and compares the exact input bytes and all
instance type/import mappings, including absent File Endpoint slots. Its
execution does not call my VM or native emitter.

I qualify this private path with simultaneous File/TCP/File owners, direct and
indirect owned/borrowed helpers, Result branches and readiness loops, permuted
maps, real IPv4/IPv6, failure/fuel cleanup, malformed-input refusal and generated
provider substitutions. My [execution evidence](evidence/mixed-service-dispatch-20261010/README.md)
records the exact tested scope. I do not infer full mixed-language acceptance
from this bytecode fixture: public per-instance grants, installed packaging,
paired source lowering and source shadows remain required, followed by the
complete 5.1 platform and release gates.

## Public mixed authority and installed runtime

I expose checked mixed execution through `services_indirect_public.h` and
`services_host_grant.h`. My host copies an ordered table of 1..64
`NvmServicesHostPolicy` entries into an opaque grant. Each entry identifies File
or TCP and explicitly permits or denies that zero-based instance. File permits
temporary files; TCP permits outbound IPv4/IPv6 connections. Neither permits
listeners or arbitrary foreign calls. The caller initializes the grant output
to NULL and destroys the grant after use.

I require the entire checked instance table to match the policy count and
catalog order, with every instance allowed, before runtime acquisition. An
unused declared instance still requires authority. A policy is reusable for
another checked module with the same catalog order; it is not a bytecode digest
or an endpoint allowlist. Repeated File entries remain independently revocable.
Revocation only removes authority and remains effective until grant destruction.

I use the shared File/TCP public-call gate for creation, revocation, destruction,
VM execution and nonexecuting emission. Reentrant calls return BUSY before
reading caller arguments. Ordinary C pointer-lifetime rules still apply outside
that refusal. I reject a grant from a different runtime identity. I validate
explicit revision-1 options and fuel, including zero, and publish a scalar only
after clean resource destruction. Failures preserve the caller's scalar.

My VM entry is `nvm_services_execute_indirect_bytes`. My nonexecuting emitter is
`nvm2c_emit_services_indirect_bytes`; its generated
`nvm_services_indirect_program_<identifier>` receives the same grant, options
and scalar arguments. Generated execution authorizes its retained checked plan
before acquiring resources and does not call my VM or emitter.

```sh
make -f Makefile.gnu services-public-runtime
make -f Makefile.gnu install-services-public-runtime PREFIX=/chosen/prefix
```

I install `lib/libnano_services_runtime.a` and the required header closure under
`include/nanolang/services`. A C99 host includes
`<nanolang/services/nanoisa/services_indirect_public.h>` and links that archive
with its platform math/crypto dependencies. My bytecode/public package tests do
not establish mixed source lowering or CLI publication; those remain required.

## Paired source lowering

My C and Nano lowerers retain the source plan's ordered declaration instances.
Each nominal source type resolves through its original declaration identity;
imports use the originating request and method ordinal. I pack each instance's
layouts contiguously, retaining catalog-local field/variant references within
that instance. Owner flags apply to each instance's owner and owner-Result
layouts. Endpoint construction uses the instance's record ordinal, independently
of its global layout index, and preserves source-order field evaluation.

I keep the existing single-declaration File/TCP wire format. Graphs containing
multiple declarations use version 3, including repeated declarations of the
same catalog. My C serializer validates mixed indirect flow and derives stack
bounds before canonical serialization. My Nano serializer independently emits
the same instance table, layouts, imports and ownership declarations.

My source fixture imports File, TCP and a second File from separate modules,
holds all three owners simultaneously, and passes owners and borrows through
helpers. It compares C output against Nano lowerers running in both VM and
native code, then executes the exact checked bytes and generated C through
public per-instance grants. Selected source shadows use the same lowerers.
My established per-function instruction and overall profile bounds still apply.

I also recognize a capitalized record type after a lowercase module alias in
my Nano parser, including `tcp.Endpoint { family: 4, ... }` as a call argument.
The prior parser treated this as field access although my C parser accepted it.

My source-lowering checkpoint alone did not establish CLI publication. The
following integration connects those bytes to explicit grants, supervised
shadows and staged product publication.

## CLI publication and invocation

My source drivers select the mixed public runtime when independent lowering
emits a version-3 instance table. I authorize every declared File instance with
`--allow-temporary-files` and every declared TCP instance with
`--allow-tcp-connections`. A File/File graph needs only the File flag; a TCP/TCP
graph needs only the TCP flag. File/TCP graphs require both. The same rule
applies even when a declared instance is not used by the selected function.

```sh
nanoc source.nano --allow-temporary-files --allow-tcp-connections --emit-nvm -o program.nvm
nano_vm --services --service-instruction-limit 1000000 --allow-temporary-files --allow-tcp-connections program.nvm
nvm2c --services --entry-name example program.nvm -o program.c
```

My raw `--services` route selects the multi-instance profile, with an explicit
decimal instruction limit from zero through 1,000,000. I reject combined
execution modes and guest arguments. C emission requires a valid entry name
and no host grant; its generated entry receives an explicit grant from its host.
Existing single-catalog routes remain separate.

Source publication validates the main module, then executes the selected
shadows in the existing supervised child. I describe each shadow's requested
catalogs before execution and create a fresh per-instance grant for each one.
The public runtime independently validates the complete plan. My supervisor
retains the whole-suite deadline, durable SELECT/START/DONE records and process
group cleanup. A missing permission or failed shadow preserves prior output.

A published native executable requires fresh flags at invocation, in either
order. It embeds only the ordered catalog policy and independent generated
execution, not the compiler's grant or VM. No-shadow emission can publish bytes
or a native program without authority; executing that program still requires
its declared catalog permissions. `make install` includes File, TCP and mixed
runtime archives and headers. Source publication also accepts those installed
headers/archives when the selected product root has no source directory.

The expanded host closure also exposed an 8 KB module-object linker-command
limit. I now assemble that command dynamically using the existing exact path
quoting. I still refuse failed allocation or linking before publishing a module
cache generation. My regression executes a 37-object library built beneath a
long path containing spaces and quotes.
