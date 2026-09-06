# Nano Service Interface

I turn a module boundary into a versioned contract. An NSI document is UTF-8
JSON. `nsi_version` is `0`. I reject any other version. I reject invalid
UTF-8. I reject a name without an `id`. I reject duplicate ids in one
document. I reject omitted `params`, omitted type `kind`, `opaque`, and
unknown keys such as `c_type`. That is ABI inference, and I fail closed.

## Identifiers

| Kind | Prefix | Shape |
| --- | --- | --- |
| Interface | `nsi:` | `nsi:<namespace>/<name>` |
| Method, type, error, parameter | `nsi:` | `<interface-id>#<token>` |
| Capability | `cap:` | `cap:<namespace>/<name>` |

Ids are ASCII: letters, digits, `:`, `/`, `.`, `_`, `-`, and `#` only in the
method/type/error/parameter fragment. Names are not the identity.

Example: [schema/nsi/examples/log.nsi.json](../schema/nsi/examples/log.nsi.json).

## Parameters

A method must include `params` (the array may be empty). Every param has an
`id`, a `name`, a `type` id, and these fields:

| Field | Values |
| --- | --- |
| `direction` | `in`, `out`, `inout`, `return` |
| `ownership` | `borrow`, `transfer`, `copy` |
| `lifetime` | `call`, `caller`, `callee`, `resource` |
| `mutability` | `immutable`, `mutable` |
| `optional` | JSON boolean |
| `streaming` | `none`, `in`, `out`, `bidi` |

`idempotent` on a method is optional and defaults to false. A newer document
may not drop idempotence from a method a client already relied on.

`type` is an `nsi:` identifier. It must name a type in this document or a
core scalar: `nsi:core/string`, `nsi:core/int`, `nsi:core/bool`,
`nsi:core/unit`, `nsi:core/bytes`, `nsi:core/float`.

Streaming must match direction. Unknown enumerations fail closed.

## Types

A type must set `kind` to one of: `record`, `variant`, `array`, `string`,
`binary`, `resource`, `callback`, `async`.

| Kind | Extra fields |
| --- | --- |
| `record` | `fields`: `{id,name,type}` |
| `variant` | `cases`: `{id,name}` and optional `type`; at least one case |
| `array` | `element`: `nsi:` type id |
| `callback` | `method`: a method id in this document |
| `async` | `result`: `nsi:` type id |

Errors may carry a `version` token (`[A-Za-z0-9._-]+`).

## Compatibility

`nl_nsi_compat(older, newer)` asks whether a client written against `older`
can call an implementation of `newer`. Adding a method is compatible.
Removing a method is breaking. Existing parameter contracts must match.
A newer method may add an optional `in` parameter. Newer variants may add
cases; they must keep old cases. Record fields may not be added or removed.
Error ids and version tokens must remain. Different interface ids are
breaking.

`nl_nsi_session_hello` applies the same rule before any call. Transport
version is `0`.

## Generation

`src/nsi_gen.c` emits NanoLang, Nano Forth, Python, Rust, and C++ stubs,
C dispatch by method id, mocks, documentation, example request/response
frames, validation tables, compatibility-test comments, a language index,
and NanoISA `IMPORT` / `TRAP cap` descriptors from one document. Generated
NanoLang includes shadow tests. I do not infer a C ABI.

`make test-nsi-gen`.

## Modules

Portable contract fields live under `nsi` in `module.manifest.json`:
interface id and version, schema path, required capabilities, isolation,
resource budgets, restart policy, and adapter. Platform build fields stay
in `module.json` (`c_sources`, `headers`). A `c_sources` key on the
manifest fails closed.

I inventory privilege, state, payload size, latency, and failure behavior
in [schema/nsi/inventory.json](../schema/nsi/inventory.json). Graphics
modules such as `sdl` and `glfw` share `nsi:nanolang/graphics` and differ
only in adapter/path. `make test-nsi-manifest`.

## Invocation

`src/nsi_runtime.c` dispatches on method ids, not symbol names. Request,
response, error, cancel, deadline, hello, and stream frames are JSON with
`nsi_version` 0. In-process, mock, and local-process adapters share the
client call. Local-process uses a `socketpair` child. Bounded queues
backpressure. Idempotent methods replay by `request_id`. Callers present
auth and capabilities or I refuse. Malformed JSON and extra payload keys
fail closed. Resource methods return generation, rights, type, and service
id; they do not return host pointers.

`make test-nsi-runtime`. The unchanged NanoLang client is
[tests/nsi_client.nano](../tests/nsi_client.nano). The unchanged Forth
client is [tests/nsi_client.fs](../tests/nsi_client.fs).

Trusted computing base: [NSI_TCB.md](NSI_TCB.md).

Capability runtime, shared memory, POSIX fabric, and the editor as a
fabric client: [NSI_FABRIC.md](NSI_FABRIC.md)
(`make test-nsi-cap test-nsi-shm test-nsi-fabric`).
