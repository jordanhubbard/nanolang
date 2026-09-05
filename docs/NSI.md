# Nano Service Interface v0

I use NSI documents to describe values at service boundaries without inferring
a host ABI. Version 0 defines typed payloads only. It does not define evolution
rules, wire encodings, generated bindings, or invocation.

An NSI document declares an interface version and a list of named types. Every
declaration, field, and variant case has a stable positive integer identifier.
Names are readable labels; identifiers are the durable identity.

## Payload Types

I accept these scalar references: `bool`, `i64`, `u64`, `f64`, `string`, and
`binary`. Strings are Unicode text. Binary values are uninterpreted bytes.
Arrays use `{ "kind": "array", "element": ... }`; their element may be a
scalar, another array, or a named declaration.

Named declarations have one of five kinds:

- `record` contains identified fields. A field may be optional.
- `variant` contains identified cases. A case may carry one typed value.
- `resource` is an opaque service-owned handle type. This slice assigns no
  transfer or lifetime semantics to it.
- `callback` declares parameters, a result, and whether completion is
  asynchronous. An asynchronous callback represents its eventual result; this
  slice defines no transport frame or scheduler.
- `error` is a record-shaped failure with its own major and minor version.

The complete example is
[`schema/nsi/examples/types.nsi.json`](../schema/nsi/examples/types.nsi.json).
The normative JSON Schema is
[`schema/nsi/v0/schema.json`](../schema/nsi/v0/schema.json).

## Closed Vocabulary

I reject unknown declaration kinds, unknown type-reference kinds, and unknown
properties. I do not guess what a future producer meant. A future NSI version
can define new vocabulary under a new contract.
