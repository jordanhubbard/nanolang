# I bind one real local file to a service owner

I record task_f9ac5bb2adbf44198a5bdb8ed41309ff under
`task_d03c232dc067e75cbc2fb2b7fb84ee46`, from canonical
`bfdde227`, before implementation. This checkpoint contains only my contract.
I require independent review before production, fixtures or resource operations.

## My existing boundaries

My original d03c acceptance promises real FileHandle, Socket, GPU and capability/
service handles, ownership through Result/error paths, generation and rights,
representative NSI and standard-library integrations through both frontends,
actual cleanup, and stale/duplicate ownership negatives. I preserve that scope.
The parent is blocked and unowned in the ledger; this locally coordinated child
is not dispatched to another worker.

My existing `tests/test_affine_integration.nano` explicitly simulates fd42.
Its qualified transitive-wrapper and inline-owner evidence establishes language
ownership, not operating-system file closure. `src/nsi_cap.c` implements a
separate capability table; consume revokes a token but owns no host file.
`src/nsi_runtime.c` currently mints descriptive handles for filesystem/net/GPU
methods. `src/nsi_gen.c` maps resources to int and emits demonstration bodies.
My existing stdio, websocket and GPU adapters expose integer handles. These are
inputs to this migration, not evidence that d03c is already complete.

I first qualify a private native C local-file adapter. I do not modify generated
NSI bindings, NanoISA import descriptors, source producers, VM dispatch or AOT
selection in this child. Their currently refused foreign owner boundary needs a
separate reviewed contract. STRING-field and mixed-array owners keep their own
ongoing work; a service registry can hold host state without adding those fields
to a language owner.

## My private context and identities

I propose `src/nsi_file.c/.h` with an opaque, single-threaded file-service context.
The context owns its capability table and every live host `FILE *`. Neither a
host pointer nor a descriptor number is a caller token. I expose a typed token
containing the existing `NlCap` identity; the table retains exact interface type
`nsi:nanolang/filesystem#File` and service `nsi:nanolang/filesystem`. Context
identity is part of validation: a token from another service context is refused.
The context is not a process-isolation or adversarial sandbox boundary.

I validate token, generation, service/type and required rights before accessing
host state. I then resolve the matching private slot. A slot contains at most one
live host file. A stale token cannot select a newly reused slot, and equal host
descriptor numbers do not imply equal service identities. Rights remain in the
trusted table, not in caller-editable token fields. My first subset supports
READ, WRITE and TRANSFER; consuming close is always permitted to the current
owner, so lack of WRITE never prevents cleanup. Other rights, attenuation,
external revocation and direct access to the backing capability table remain
outside this private API.

I retain the existing bounded64-slot capability capacity. I need a separately
reviewed internal retirement/reuse operation: it accepts only the exact revoked
entry, makes that slot reusable after its host association is gone, and never
resets or reuses an issued generation. Mint/transfer must fail before generation
wrap or reserved-zero publication. Existing public capability functions keep
their behavior; the file adapter uses checked internal lifecycle entry points.
I do not obtain reuse by calling table restart while other owners are live.
If this factoring cannot preserve existing users, I stop for a revised contract
before changing their semantics. A transfer may need one spare slot; capacity
failure preserves its source owner. Repeated bounded-live acquisition/close must
reuse capacity instead of exhausting a lifetime issuance quota.

I empty the registry on context disposal, attempting each still-live file close
once and invalidating its tokens. Disposal keeps the capability table and context
storage so the disposed state remains queryable; destruction frees both after
disposal. Disposal returns the first close error after attempting remaining files. A failed
file close cannot prevent other files from being cleaned. The context has an
explicit disposed state until its separate C storage destruction; a caller must
not use the pointer after destruction. Nested/concurrent use is not admitted.

## My finite operation and Result contract

I acquire only an actual temporary file through `tmpfile()` in this first slice.
I accept no arbitrary pathname, directory authority, remote service or device.
The temporary file has ordinary binary update semantics; requested READ/WRITE
rights limit the private operations despite the underlying stream mode.
I reserve adapter/table capacity before acquisition where possible. An acquired
file remains private until its complete token-to-slot association can be
published. A failed acquisition publishes no token. Any later mint or publication
failure closes the acquired stream once and restores the reserved slot.

My C status/result records are a private representation of the eventual language
Result contract. They do not claim current language union/foreign ABI support.
I keep handle outputs unchanged on failure. I specify byte progress separately;
I do not claim to roll back host writes or bytes already read.

| Operation | Success | Failure and owner state |
| --- | --- | --- |
| create context | One empty owned context. | No published context; every partial allocation freed. |
| acquire temporary file | One new live token; requested rights and exact identity. | Caller token output unchanged; no acquired host file escapes. |
| read(token, buffer, capacity) | READ checked; byte count and EOF explicit; token stays live. | Invalid token/rights/arguments touch no stream or output. Host I/O error reports bytes already read and host error; owner stays live for close. |
| write(token, bytes, length) | WRITE checked; progress explicit; token stays live. | Invalid token/rights/arguments touch no stream or output. Partial host progress is reported; owner remains live. |
| rewind(token) | READ checked; update stream transitions to reading from start. | Validation failure has no stream effect; host seek failure retains owner and reports error. |
| transfer(token) | New token owns the same file; old token is invalid; no host close. | Preparation/capacity failure leaves old owner live and destination token unchanged. |
| consume-close(token) | Token retired; one host close attempted; no owner returned. | Invalid/stale/duplicate token performs no close. Once validated consumption starts, token is invalidated even if host close reports error; error says consumed, never retryable ownership. |
| dispose context | Every remaining stream consumed once; all tokens invalid. | First host close error retained while remaining entries are cleaned; context is disposed. |

I represent `consumed` explicitly in close results. A rejected token has not
consumed an owner; a host error after an accepted close attempt has consumed it.
I never retry `fclose` on a stream whose close was already attempted. A host error
is not a claim of durable flush or of operating-system recovery. I preserve its
error evidence. A fault hook that reports a close error after actually closing
the stream can test this ownership decision without leaking a real descriptor.

I validate null buffer/length combinations and size bounds before host access.
A zero-length operation has zero progress and retains ownership. For write→read
acceptance I use the explicit rewind operation; I do not depend on an implicit
stdio direction switch. I preserve embedded zero bytes and compare lengths as
well as bytes. There is no string conversion or host locale dependency here.

## My update-stream and error boundary

I track each stream's last nonempty I/O direction: neutral, read or write. Both
write→read and read→write refuse with DIRECTION until a successful explicit
rewind positions the stream. I do not rely on EOF or a flush to waive the
read→write rule. A failed positioning call leaves the direction restricted;
a successful `fseek(stream, 0, SEEK_SET)` clears the error indicator and returns
the direction to neutral. Zero-length I/O changes neither position nor direction.
Transfer carries the stream's direction to its new owner unchanged.

I capture errno immediately after each host operation, before table retirement,
other closes or deallocation. An acquisition/publication error keeps its primary
status and a separate cleanup-error field if rollback close fails. Consuming close
reports its saved host error after retirement. Disposal saves the first complete
error result before cleaning subsequent slots. No later cleanup overwrites it.

My private token also carries a process-local monotonically allocated context
identity, distinct from the capability generation. Serialized tokens and concurrent
context creation remain outside this native private API; I fail before context
identity wrap. This prevents a token from a destroyed context selecting a later
context at a reused C allocation address. I do not strengthen the existing public
capability API's security claims or change its random-token behavior.

## My commit and cleanup order

Acquisition reserves private storage, acquires the file, establishes capability
identity and association, and publishes the token last. Transfer validates the
current owner and rights, prepares a distinct valid destination, commits the
association and invalidates the source before publishing the destination.
Preparation cannot consume the source. No fallible allocation follows a transfer
commit. Close validates and detaches exactly its own association, makes the token
unusable, then attempts the host close once. Failure never searches by raw host
handle or closes another slot. Registry entries survive capability-table growth
or reuse through indices; no cached internal pointer crosses a mutating call.

My adapter owns no arbitrary external pointer. The test harness owns caller
buffers and its independent sentinel file. The adapter must not close that
sentinel during failure recovery, disposal or duplicate/stale rejection. I test
identity after slot reuse, not merely an empty-table rejection.

## My ordered qualification

1. I review implementation against this contract before execution. I inspect
   the capability retirement/generation boundary separately, retain all existing
   capability tests and budgets, and keep the new adapter unselected by existing
   source, VM, NSI runtime and native compiler entry points.
2. I freeze source and real tool identities. On Linux and Darwin I acquire real
   temporary files, write/read exact bytes including NUL and empty input, transfer
   without closure, consume-close, and verify ordinary descriptor cleanup. I use
   private test interception to count actual acquisition/close operations, plus
   independent sentinel and byte observations. Mocks alone are insufficient.
3. I test wrong rights, cross-context tokens, stale pre-transfer tokens, duplicate
   close, reuse with an old token, wrong type/service, capacity, generation limit,
   repeated bounded-live use and disposal with multiple live files. Invalid
   tokens are ordinary checked controls; no crash or historical artifact replay.
4. I deterministically fail each adapter allocation and publication boundary,
   stop at the first terminal failure and retain it. I require unchanged handle
   outputs, exact retained/consumed state, zero escaped resources, unaffected
   sentinel and subsequent recovery. Host acquisition/read/write/seek/close error
   hooks remain separate from allocator faults. If storage is reserved before
   acquisition, I do not claim an unexercised post-acquisition allocation point;
   publication-failure cleanup still exercises the acquired-file path.
5. I run strict native builds and applicable GCC/Clang sanitizers, existing
   capability/NSI controls and exact unaffected caller behavior. I retain source,
   actual compiler identities, commands, statuses and artifacts. This qualifies
   only the private local-file adapter after independent review and canonical
   integration, not generated bindings or public execution.

## My later dependency order remains required

After this private lifecycle qualifies, I record a separate exact NSI ownership/
Result descriptor and generated binding contract. I then connect that contract to
paired C-seed/selfhost source production and explicit verified VM/AOT service-call
and cleanup paths. I preserve original function/local/layout identity and mandatory
shadows; I do not wrap an int extern in a resource shell and call it verified.
Public admission waits for matched runtime tag/rights/error checks and cleanup.

Representative socket and GPU service adapters follow their real lifecycle and
platform contracts with actual target evidence. Capability transfer/error/restart
coverage and representative standard-library integrations remain d03c acceptance.
I do not replace them with universal hardware support, container semantics or a
new release gate. ed702 metadata/translator preservation, full affine28f2 and the
5.1 product/release parents remain open. I claim no implementation or acceptance
from this contract-only commit.

## My first private production checkpoint

I add `src/nsi_cap_private.h` and private lifecycle functions to the existing
capability implementation; existing public function bodies remain unchanged.
Private consume validates the live token, marks it revoked and retires that exact
slot atomically. Private transfer mints a distinct destination before retiring
the source, preserving source/output aliasing and failure rollback. Retirement
also invalidates any old Forth slot binding rather than letting slot reuse retarget
it. Generation exhaustion is checked before mint, without resetting the counter.

`src/nsi_file.c/.h` remain unselected by existing build/runtime entry points.
The fixed registry is allocated with the context; its capability table is the
second allocation. Every host stream is acquired only after context creation.
Mint rejection after tmpfile exercises acquired-stream rollback; I do not invent
a per-file allocator fault where the implementation has no allocation. Context
identity is a separately checked monotone uint64 counter. File slot identity,
capability generation/secret, rights and exact type/service are checked together.

I require explicit positioning before either nonempty direction change. Each
fread/fwrite/fseek/fclose captures errno before any subsequent inspection or
cleanup; error results retain consumed state and partial progress where relevant.
Close/transfer detach the old registry entry before private token retirement and
restore it if preparation refuses. Successful transfer has no fallible operation
after capability commit. Disposal closes all remaining privately stored streams
even if a later token check unexpectedly refuses, while retaining the first error.
I have run only a diff check. No build, fixture or host file operation has run.

## My qualification freeze boundary

I clarify disposal versus destruction before qualification: disposal closes all
files and prevents every later operation, retaining the empty table/context
storage; destruction frees that storage. My tests distinguish the two phases.
I compile production once as ordinary linked objects and separately include it
in a private instrumented C fixture to reach exact generation/identity limits
without billions of operations. The latter may inspect private state but grants
no public setters or hooks. Host error wrappers perform real closes before
reporting a requested close error. I retain strict warnings and actual allocator,
file and sentinel accounting. Existing NSI checks in this fresh C-only checkout
do not establish source-client compilation if no compiler binary is installed.
