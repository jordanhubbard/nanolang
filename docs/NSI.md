# Nano Service Interface

I turn a module boundary into a versioned contract. v0 is identifiers, an
optional parameter contract, and optional typed payloads.

An NSI document is UTF-8 JSON. `nsi_version` is `0`. I reject any other
version. I reject invalid UTF-8. I reject a name without an `id`. I reject
duplicate ids in one document.

## Identifiers

| Kind | Prefix | Shape |
| --- | --- | --- |
| Interface | `nsi:` | `nsi:<namespace>/<name>` |
| Method, type, error, parameter | `nsi:` | `<interface-id>#<token>` |
| Capability | `cap:` | `cap:<namespace>/<name>` |

Ids are ASCII: letters, digits, `:`, `/`, `.`, `_`, `-`, and `#` only in the
method/type/error/parameter fragment. Names are not the identity. Two contracts
that share a name and disagree on an id are different contracts.

Example: [schema/nsi/examples/log.nsi.json](../schema/nsi/examples/log.nsi.json).

## Parameters

A method may omit `params`. If `params` is present, every entry has an `id`,
a `name`, a `type` id, and these fields:

| Field | Values |
| --- | --- |
| `direction` | `in`, `out`, `inout`, `return` |
| `ownership` | `borrow`, `transfer`, `copy` |
| `lifetime` | `call`, `caller`, `callee`, `resource` |
| `mutability` | `immutable`, `mutable` |
| `optional` | JSON boolean |
| `streaming` | `none`, `in`, `out`, `bidi` |

`type` is an `nsi:` identifier. It may name a type in this document or a
core scalar such as `nsi:core/string`. I do not infer ABI from names.

Streaming must match direction: `in` streams need `in` or `inout`; `out`
streams need `out`, `inout`, or `return`; `bidi` needs `inout`. Unknown
enumerations fail closed.

## Types

A type may omit `kind` (opaque named type) or set `kind` to one of:
`opaque`, `record`, `variant`, `array`, `string`, `binary`, `resource`,
`callback`, `async`.

| Kind | Extra fields |
| --- | --- |
| `record` | `fields`: `{id,name,type}` |
| `variant` | `cases`: `{id,name}` and optional `type`; at least one case |
| `array` | `element`: `nsi:` type id |
| `callback` | `method`: a method id in this document |
| `async` | `result`: `nsi:` type id |

`nsi:core/string`, `nsi:core/int`, `nsi:core/unit`, and `nsi:core/bytes`
are core scalars. They do not appear in `types[]`. Errors may carry a
`version` token (`[A-Za-z0-9._-]+`). I do not infer a C ABI from these
kinds.

Example: [schema/nsi/examples/types.nsi.json](../schema/nsi/examples/types.nsi.json).

## Compatibility

`nl_nsi_compat(older, newer)` asks whether a client written against `older`
can call an implementation of `newer`. Adding a method is compatible.
Removing a method is breaking. Existing parameter contracts must match.
A newer method may add an optional `in` parameter. Newer variants may add
cases; they must keep old cases. Record fields may not be added or removed.
Error ids and version tokens must remain. Different interface ids are
breaking.

Wire request/response frames are not in v0. These rules apply to NSI
documents and will apply to frames when transport lands.

## What v0 is not

v0 does not describe generated bindings or a wire frame. Those are later
Phase 16 items. Loading a document does not migrate a module. `module.json`
and `module.manifest.json` stay the current build and discovery metadata.

I do not generate clients from v0. I do not claim a service fabric.
