# My capture metadata transport

I implement this dependency under `task_95103bbeb64a485701008b9e259ea991`
inside full shared-capture task `task_af8091f571a842bc90656e2c7f19b68e`.
This source audit uses PR937 head38275cdfb and canonical main3ff27be579.
My [wire contract](NANOISA_CAPTURE_BINDING_CONTRACT.md) remains authoritative.
This document specifies work before implementation; it is not qualification.

## Exact lifetime

My `NvmModule` owns canonical `capture_data` and `capture_size`, just as it
owns other required payload bytes. Both are absent together. A nonempty
payload owns a distinct allocation; my module destructor releases it once.
My decoded `NvmCaptureBindings` tables borrow those immutable bytes and cannot
outlive their owner. A parse, clone or replacement stages bytes and tables
privately and publishes only after validation. Allocation failure preserves the
source module and releases every unpublished allocation. The existing v2 bridge
uses fresh output storage: `nvm_v2_to_nvm_module` clears its output pointer on
failure, as required by my existing code-publication controls. A caller replacing
an owned module must stage into a separate pointer and replace its prior module
only after success. I do not change that existing API contract or claim that a
cleared output pointer preserves its prior pointer value.

My `NvmV2Module` view borrows payload bytes from its input or source module.
Conversion back to an execution module deep-copies them before the source
buffer can be released. I do not store a borrowed decoded view in a temporary
module and later publish its dangling pointers. Payload and index allocations
share a64MiB transport budget; code-validation work is bounded separately at
128MiB charged function/code-byte steps. I reject over-budget input before
allocating its payload copy. These are explicit admission limits, not claims
that every representable container fits them.

## Container and text

I add the already reserved bit10 and section15 to `nvm_format_v2.h`, extend
the section plan from14 to15 rows, and require exact feature/section pairing
in both serialization and deserialization. An empty section, pointer/size
disagreement, duplicate section, wrong counts, invalid modes, invalid sites
or trailing bytes is refused. I validate against the fully constructed
function table and code using the existing capture codec and structural checker.
No raw payload becomes an execution proof.

`nvm_v2_convert.c` preserves bytes in both directions. `nvm_format.c` refuses
v1 serialization when captures are present, as it does for other required v2
contracts; it frees the owned payload on every ordinary destruction path.
Canonical disassembly emits bounded `.capture_bindings` hexadecimal chunks.
Assembly retains their exact bytes and validates the complete result. The
public `modules/nanoisa/nanoisa.c` load/save and text routes must preserve the
same contract and diagnostic distinction. Existing capture opcodes alone do
not provide the missing function modes or site table.

## Projection and linked identity

My existing VM links separate module objects through `vm_link_module`,
`vm_link_named_module` and `linked_modules`. Their function indices remain module-relative. I preserve
that identity instead of inventing a flattened linker or rebasing indices
that did not move. Closure ownership must continue naming the actual executing
module as well as its target function. The linked verifier and all public
entry paths examine every participating module.

An identity-preserving clone copies exact bytes. A projection that changes
function indices or instruction boundaries must rebuild function records,
owner/target indices, site ordering, relative offsets and each CLOSURE_BIND
site operand together, then validate the resulting payload/code pair. It must
include the complete referenced target closure or refuse before publication.
Until that transformation is implemented and qualified, such a projection
refuses the feature; refusal does not close my full projection obligation.

My `managed_record_array_execution.inc` manually copies and frees a selected
module snapshot. Its current closed-profile absence check does not know the
new payload. I add an explicit admission decision before copying: no silent
omission followed by a successful proof. The same rule applies to retained
layout/ownership declaration projections, private service/File plans, portable
host plans and any other partial module view. Declaration-only inspection may
retain its existing narrower report, but cannot claim complete module equality
or execution readiness after discarding a required feature.

## Admission dependency

Recognizing a container bit expands transport, not execution. Before enabling
container transport I audit the ordinary and per-function verifiers, linked
verification, VM checked/decoded/public/direct/indirect/tail/callback entries,
C and LLVM translators, Wasm lowering, private ownership/service entry points
and frozen execution snapshots. Each unsupported route explicitly refuses a
capture-bearing module before dispatch, output publication or proof caching.
The new feature is not made advisory by a verifier failure and checked fallback.

Later reviewed definite-initialization, stack/type/effect and runtime proofs
replace these refusals on each supported route. Full paired source lowering,
VM/C/LLVM/Wasm execution, projections and bootstrap fixed point remain required
by5.1. Passing transport tests never closes that parent.

## Acceptance

I require original container, codec and schema controls plus real module
roundtrips and destruction after releasing the original input. Exact byte
comparisons cover local/capture modes, repeated shared sources, empty target
environments, multiple functions/sites, and canonical text and binary output.
Feature-only/section-only and malformed metadata controls preserve outputs.
I test corrected allocation-prefix failures and independent recovery, including
failure after byte allocation and during each index allocation.

Consumer controls use a valid capture payload with ordinary-looking code where
possible: opcode rejection alone must not hide missing required-feature checks.
I inspect actual refusal and unchanged output through public/private, linked
and translation interfaces. I do not execute a known unsupported instruction
path merely to reproduce a crash. Fresh Linux and Darwin ordinary and strict
sanitizer gates retain exact source/provider identities, actual terminals and
first failures. Integration and source bootstrap gates follow the complete
consumer implementations.

## My transport source checkpoint

I now own and free the capture payload in `NvmModule`, borrow it in a v2 view,
and deep-copy it on conversion to an execution module. My v2 reader checks the
exact feature/section pair; reader and writer validate payload structure against
the complete function table and code. My canonical text writer validates once
before producing module output. It does not repeat allocating validation for
each function or publish an empty successful string after a second validation
fails. My string writer also refuses a failed stream close.

The internal `asm_assemble_unverified` transport route accepts structurally
valid metadata; ordinary `asm_assemble` and the public NanoLang assembly wrapper
still require execution verification and refuse it. This distinction preserves
the public verification contract while permitting codec roundtrip controls.
Legacy v1 serialization refuses capture metadata. My source guards cover ordinary,
linked, affine and type verification, VM ownership/entry selection, linking,
FFI routes, native C/LLVM translation and closed private execution profiles.
Service-plus-capture admission remains refused. My explicit NanoISA and Forth
SEE provider manifests include the capture codec.

This is a source checkpoint, not passing qualification. I still require an
independent consumer audit, fixture review, real binary/text roundtrips,
allocation recovery, destruction and unsupported-consumer controls, followed by
fresh Linux and Darwin gates. Full capture execution and my 5.1 parent remain open.

My initial strict GCC syntax-only pass accepts fourteen changed translation
units. The LLVM unit stops at its missing Make-generated `managed_runtime_ir.h`
header; I retain that build setup failure without rerunning it or calling it a
product result. These [raw checks](evidence/capture-bindings/transport-source-review/results.json)
do not execute the product, exercise linking, or replace the full owning build.

My independent source review found two omissions in ee681548f: the explicit
installed File archive omitted the codec dependency, and my text parser enforced
the64MiB payload budget only after accumulating all chunks. I add the codec to
`FILE_PUBLIC_QUERY_STEMS` (also used by the private cyclic provider closure)
and check remaining payload budget before every capture `realloc`. My installed
File controls now compile and run a separate bridge consumer using only installed
headers and `libnano_file_runtime.a`. This fixture is source-reviewed work;
its actual installed qualification remains pending.

My first transport fixture constructs two module families: ordinary RET bodies
with required modes but no closure sites, and four functions with repeated
shared sources, copied captures, an empty target environment and forwarding.
It checks binary identity across independent module ownership, destroys both
borrowed sources before using the copy, and requires canonical text to preserve
the complete serialized module. It separately checks ordinary/function/affine/
owned/linked verification, legacy serialization and native C refusal. Verified
assembly must refuse metadata that the internal transport assembler preserves.
This fixture never executes capture-bearing instructions. Allocation-prefix,
VM/LLVM/Wasm/private-consumer and installed-only controls remain required.

My first fresh Linux GCC ordinary run at e82158064 passes180 transport checks,
977 existing codec checks and33 schema controls in7.883seconds. Tracked source
and selected tool maps remain unchanged;79 products remain under the durable
qualification root with recorded hashes. I retain [the reports and drivers](evidence/capture-bindings/transport-first-linux/checks.json).
The first setup used quoted Git path output and failed before compilation; the
second lacked the Git baseline required by schema-check. I corrected these
external preparation steps in fresh directories without changing product source
or weakening schema checks. The original codec pass and both setup failures
remain separate from the final successful focused run. Broader both-host,
allocation, installed-package and execution-refusal qualification remains open.

My next fixture interposes only capture payload allocation in the v2 bridge and
the two capture decoder table allocations. It tests each transient and persistent
failure, then an independent successful attempt, across copying to an owned
module, creating a borrowed v2 view and producing canonical text. It requires
the existing NULL-on-failure bridge contract, exact source bytes and unchanged
input view, and release of every tracked payload/table. It does not claim to
fault every unrelated module allocation. Source review precedes execution.

My fresh Linux GCC ordinary run at b494b6da7 passes 180 transport checks,
541 allocation-enabled transport checks, 977 codec checks and 33 schema
controls in 8.888 seconds. Source and selected tools remain unchanged, the
process group is gone, and 83 products remain under the recorded qualification
root. I retain [the allocation reports](evidence/capture-bindings/transport-allocation-linux/checks.json).
The reviewed fixture snapshots input structure bytes with `memcpy` before
comparing them, preserving padding rather than assuming assignment copies it.
These results cover the scoped allocation sites above. Both-host sanitizer,
VM/LLVM/Wasm/private-consumer and installed-only acceptance remains open.

My next consumer fixture uses ordinary RET bodies with structurally valid capture
metadata to isolate admission from opcode support. It checks all four public VM
entry APIs, unchanged result storage and empty frames/stack on refusal, linking
refusal, and a plain-module execution control. Native LLVM and Wasm translation
must refuse before changing an existing output stream. This fixture is awaiting
source review and execution; it does not establish capture execution support.
