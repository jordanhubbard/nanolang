# I retain explicit C ABI metadata without admission

My optional `module.json` member `typed_abi` describes C storage bindings. Its absence performs no parser allocation and adds no build-context bytes. Its presence is strictly parsed, independently copied into canonical JSON, owned by `ModuleBuildMetadata.typed_abi`, and freed with that metadata. Captured compiler-invocation metadata borrows this immutable string for the existing parent lifetime; its selective destructor does not acquire ownership.

```json
{
  "typed_abi": {
    "version": 1,
    "target": "provider-declared-target",
    "types": [
      {"semantic": "int", "c_type": "int64_t", "kind": "int"},
      {"semantic": "Packet", "c_type": "ProviderPacket", "kind": "record", "fields": ["value", "owner"]}
    ],
    "functions": [
      {"name": "make_packet", "symbol": "provider_make_packet", "parameters": [0], "result": 1}
    ]
  }
}
```

Object keys are unique and closed at every schema level. Top-level schema keys are exactly `version`, `target`, `types`, `functions`. Object order is canonicalized lexically; all array order remains significant. Version is exactly integer1. `semantic` retains an annotation selector for later resolution in the actual provider's module context; it is not already a nominal key or execution proof. `target` also remains an unverified declaration until attachment.

Every type row requires `semantic`, `c_type`, `kind`. `c_type` is a declared C typedef identifier, not an injected type expression. Header existence and prototype agreement remain the later generator's responsibility. Scalar kinds are `int`, `float`, `bool`, `u8`, `string`; `opaque` and `function` also use declared typedefs. Those kinds permit no further members. Records and tuples additionally require ordered `fields`, whose strings are C member paths consisting of identifiers separated by dots. Unions instead require discriminator member path `tag` and a nonempty ordered `variants` array; each variant has a C constant identifier `tag` and ordered `fields` paths. Enums require a nonempty ordered `variants` array of C constant identifiers. Duplicate field paths or variant constants refuse.

An array row instead requires `abi: "dyn_array_v2"` and an `element` index into this storage-binding array. This names the already defined DynArray ABI; no unnamed pointer/count ABI is admitted. Other foreign array storage needs a separately explicit conversion schema before use. The index is a mapping reference, not a replacement semantic type: future declaration crossvalidation must match it against the existing complete shared type graph.

Each function row has exactly `name`, `symbol`, `parameters`, `result`. Name and symbol are C-identifier-shaped strings; `parameters` is an ordered array of storage-binding indices, and `result` is one index or null for void. A tuple result uses its explicit tuple binding. Indices must be nonnegative integers within the actual type array. Type semantic selectors and function names are unique. Symbol aliases may repeat, but later exact prototype/owner crossvalidation must establish compatibility.

I bound this private schema to4,096 type rows,65,536 function rows,65,535 fields/variants/parameters per row,255-byte C identifiers,4,096-byte other strings/member paths, a conservative1MiB canonical-copy size preflight and1,048,576 validation work units. Repeated name comparisons charge their byte lengths. The grammar bounds nesting independently of cJSON's general nesting limit. The surrounding existing module parser continues to require complete UTF-8 input without literal or escaped NUL. Missing, unknown, duplicate, fractional and malformed fields refuse; I do not silently drop them.

Successful canonicalization deep-copies the validated tree and serializes sorted objects, then copies the result into ordinary malloc-owned storage so cJSON allocation hooks do not change metadata disposal. Every fallible allocation is cleaned up before publication. The build fingerprint receives a versioned marker and the canonical bytes only when the schema exists. The required SDK include is added to the canonical input list and regenerated with `scripts/generate_native_sdk_inventory.py`; this is dependency closure, not a final release inventory freeze.

The owning C fixture covers copied lifetime after input destruction, stable parse/serialize/reparse, old-schema absence, syntax/refusal controls and every measured allocation prefix in both failure modes with fresh recovery. The existing module-generation probe gains a read-only mode that prints canonical metadata plus actual production build-context hashes with and without the schema. Python controls check key-order normalization, original absent behavior, roundtrip, meaningful schema mutations, bounds and exact refusals. No foreign provider, generated adapter, compiler bootstrap or installed execution is invoked by these controls.

Semantic declaration binding, generated C prototype/layout checks, immutable image attachment, callback/COP transport and full installed SDK execution remain required. This source checkpoint only parses and fingerprints metadata.
