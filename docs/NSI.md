# Nano Service Interface

I turn a module boundary into a versioned contract. v0 is identifiers plus
an optional parameter contract.

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
core scalar such as `nsi:core/string`. Records, variants, arrays, and
resources are a later item. I do not infer ABI from names.

Streaming must match direction: `in` streams need `in` or `inout`; `out`
streams need `out`, `inout`, or `return`; `bidi` needs `inout`. Unknown
enumerations fail closed.

## What v0 is not

v0 does not describe records, variants, compatibility rules, generated
bindings, or a wire frame. Those are later Phase 16 items. Loading a
document does not migrate a module. `module.json` and
`module.manifest.json` stay the current build and discovery metadata.

I do not generate clients from v0. I do not claim a service fabric.
