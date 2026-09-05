# Nano Service Interface

I turn a module boundary into a versioned contract. v0 is identifiers only.

An NSI document is UTF-8 JSON. `nsi_version` is `0`. I reject any other
version. I reject invalid UTF-8. I reject a name without an `id`. I reject
duplicate ids in one document.

## Identifiers

| Kind | Prefix | Shape |
| --- | --- | --- |
| Interface | `nsi:` | `nsi:<namespace>/<name>` |
| Method, type, error | `nsi:` | `<interface-id>#<token>` |
| Capability | `cap:` | `cap:<namespace>/<name>` |

Ids are ASCII: letters, digits, `:`, `/`, `.`, `_`, `-`, and `#` only in the
method/type/error fragment. Names are not the identity. Two contracts that
share a name and disagree on an id are different contracts.

Example: [schema/nsi/examples/log.nsi.json](../schema/nsi/examples/log.nsi.json).

## What v0 is not

v0 does not describe parameter direction, ownership, lifetimes, streaming,
compatibility rules, generated bindings, or a wire frame. Those are later
Phase 16 items. Loading a document does not migrate a module. `module.json`
and `module.manifest.json` stay the current build and discovery metadata.

I do not generate clients from v0. I do not claim a service fabric.
