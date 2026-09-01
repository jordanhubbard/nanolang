# NanoISA v2 Module Format

Roadmap 4.0, Phase 12, Module format and tools.

NanoISA v2 semantics have landed in code — verified call signatures (#136),
linked callable handles (#137), instruction-boundary maps (#142), layout-driven
`AGG_*` operations, and the separation of serialized bytecode from verified and
dispatch IR (#144). The on-disk format that must carry those semantics has not.
This document specifies it.

## The problem

The current `.nvm` format is the v1 layout. Three gaps make it unable to
express v2:

**The header has no version or size discipline.** It carries `format_version`
but no ISA version, no feature bits, and no total size, so a loader cannot tell
which instruction set a module was built against, cannot negotiate optional
capabilities, and cannot bound the file before reading the section directory.

**Most sections are declared but never written.** `nvm_serialize` writes five
sections — `STRINGS`, `CODE`, `FUNCTIONS`, `DEBUG`, `IMPORTS`. `GLOBALS`,
`METADATA`, `MODULE_REFS`, `STRUCTS`, `ENUMS`, and `UNIONS` appear in
`NvmSectionType` and are accepted by the deserializer, but nothing produces
them.

**The sections v2 requires do not exist at all.** There is no signatures
section, so verified signatures live only in memory. There is no layouts
section, so the `layout` operand of `AGG_PACK`/`AGG_GET`/`AGG_SET`/`AGG_TAG`
has no on-disk referent. There is no links section, so linked callable handles
cannot be serialized.

A module written today therefore cannot round-trip v2 semantics. Freezing this
format for 4.0 would either ship an ISA whose modules cannot express it, or
force a second format break immediately after a major release.

## Goals and non-goals

I will:

- Give the header an ISA version, feature bits, and a total size, so a loader
  can validate and negotiate before trusting any offset.
- Serialize every structure v2 semantics depend on: constants, signatures,
  layouts, globals, imports, links, functions, code, and metadata.
- Make the module the unit of separate compilation, with imports resolved once
  into typed call descriptors at link time.
- Size runtime structures from serialized declarations rather than from
  compile-time maxima.
- Preserve byte-exact string payloads, including embedded zero bytes.

I will not:

- Support v1 modules. See [Migration](#migration).
- Specify the verified or dispatch IR. Those are private runtime
  representations derived from this format, not serialized forms of it.
- Specify the co-process wire protocol, which is a separate transport.

## Module model

I choose the **separately linked** model. A module is a self-describing unit
that names its external dependencies symbolically and is bound to them at link
time.

The alternative — flattening every dependency into one module before
serialization — produces a simpler format, but it discards separate
compilation, retires the `CALL_MODULE` path added by #137, and leaves nothing
on which the 4.3 service-contract work can build. The cost of the linked model
is that a linker must exist and be verified; that cost is paid once and is
already partly paid.

Consequences for this format: `SIGNATURES` and `LINKS` are core sections, not
optional ones, and every cross-module call site carries a link index rather
than a module/function index pair.

## Header

All integers are little-endian. The header is 40 bytes.

| Offset | Size | Field | Notes |
| --- | --- | --- | --- |
| 0 | 4 | `magic` | `'N'`, `'V'`, `'M'`, `0x02` |
| 4 | 2 | `format_version` | Starts at 1 for the v2 lineage |
| 6 | 2 | `isa_version` | NanoISA version the code was assembled against |
| 8 | 4 | `feature_bits` | See below |
| 12 | 8 | `total_size` | Byte length of the entire file |
| 20 | 4 | `header_size` | 40; lets the header grow without a magic bump |
| 24 | 4 | `section_count` | Number of section directory entries |
| 28 | 4 | `entry_point` | Function index of `main`, or `0xFFFFFFFF` for none |
| 32 | 4 | `flags` | Reserved; must be zero |
| 36 | 4 | `checksum` | CRC32 of bytes `[header_size, total_size)` |

`magic[3]` distinguishes the format lineage; `format_version` versions the
layout within it; `isa_version` versions the instruction semantics. These are
three independent axes and conflating them is what made v1 unextendable.

`string_pool_offset` and `string_pool_length` are removed. The section
directory already locates every section, and duplicating one section's extent
in the header created two sources of truth.

### Feature bits

| Bit | Name | Meaning |
| --- | --- | --- |
| 0 | `FEATURE_LINKED` | Module has unresolved links; `LINKS` is present and non-empty |
| 1 | `FEATURE_FFI` | Module calls foreign functions; `IMPORTS` is non-empty |
| 2 | `FEATURE_COPROCESS` | Module requires the co-process host |
| 3 | `FEATURE_DEBUG` | `DEBUG` section present |
| 4 | `FEATURE_CLOSURES` | Module constructs heap closures |
| 5-31 | — | Reserved; a loader rejects a module setting an unknown bit |

Rejecting unknown feature bits is what makes the format extensible: a v2.1
module that uses a capability a v2.0 runtime lacks fails loudly at load rather
than subtly at execution.

## Section directory

`section_count` entries of 24 bytes each, immediately after the header.

| Offset | Size | Field |
| --- | --- | --- |
| 0 | 4 | `type` |
| 4 | 4 | `flags` (reserved, zero) |
| 8 | 8 | `offset` from start of file |
| 16 | 8 | `size` in bytes |

64-bit offsets and sizes match `total_size`; v1 mixed 32-bit section offsets
with no total size at all.

The existing directory validation (`nvm_validate_section_directory`) already
rejects duplicate singleton sections, overlaps, gaps, partial fixed-width
records, and trailing data using subtraction-only bounds arithmetic. That logic
carries forward unchanged and is the one piece of the v1 loader worth keeping;
it is widened from 32-bit to 64-bit arithmetic.

## Sections

Section types are renumbered — the clean break makes v1 numbering irrelevant.

| Type | Name | Required |
| --- | --- | --- |
| 0x01 | `METADATA` | yes |
| 0x02 | `CONSTANTS` | yes |
| 0x03 | `SIGNATURES` | yes |
| 0x04 | `LAYOUTS` | yes |
| 0x05 | `FUNCTIONS` | yes |
| 0x06 | `CODE` | yes |
| 0x07 | `GLOBALS` | yes |
| 0x08 | `IMPORTS` | when `FEATURE_FFI` |
| 0x09 | `LINKS` | when `FEATURE_LINKED` |
| 0x0A | `DEBUG` | when `FEATURE_DEBUG` |

`STRUCTS`, `ENUMS`, and `UNIONS` are subsumed by `LAYOUTS`; `STRINGS` by
`CONSTANTS`; `MODULE_REFS` by `LINKS`.

Every section begins with a `u32` count and pads each variable-length record to
a 4-byte boundary.

### CONSTANTS

Replaces the string pool with a typed constant pool.

```
count       u32
per entry:
  tag       u8      NanoValueTag
  _pad      u8[3]
  length    u32     payload byte length
  payload   u8[length]   padded to 4
```

Strings store an explicit `length` and the payload is copied verbatim, so
embedded zero bytes survive. This is the serialized half of the roadmap item
requiring stored string lengths; the runtime half is converting `vmstring_cstr`
call sites to use `vmstring_len`.

Because constants are typed and length-prefixed, a loader can preinstantiate
them once at load time rather than allocating and interning on every execution
of `PUSH_CONST`.

### SIGNATURES

New. The single source of truth for call shapes.

```
count         u32
per entry:
  param_count   u16
  result_count  u16
  param_tags    u8[param_count]     padded to 4
  result_tags   u8[result_count]    padded to 4
```

Functions, imports, links, and indirect call sites all reference a
`signature_idx`. Verification compares signature indices rather than
re-deriving shapes from three different encodings, which is what makes
"regularize calls around verified signatures" checkable at load time instead of
at call time.

### LAYOUTS

New. Gives `AGG_PACK`, `AGG_GET`, `AGG_SET`, and `AGG_TAG` an on-disk referent.

```
count         u32
per entry:
  kind        u8      0=struct 1=tuple 2=union 3=enum
  _pad        u8
  field_count u16
  name_idx    u32     CONSTANTS index, or 0xFFFFFFFF for anonymous
  per field:
    type_tag  u8
    _pad      u8[3]
    nested_layout_idx u32   0xFFFFFFFF when the field is scalar
    name_idx  u32
```

A layout is closed: every nested index refers to a lower-numbered layout, so
the table is acyclic by construction and a verifier can validate it in one
forward pass.

### FUNCTIONS

```
count           u32
per entry:
  name_idx      u32
  signature_idx u32
  code_offset   u64    byte offset into CODE
  code_length   u64
  local_count   u16
  upvalue_count u16
  max_stack     u16    verifier-proven maximum operand depth
  flags         u16
```

`arity`, `result_tag`, and `result_count` move into `SIGNATURES`. `max_stack`
is new and is what lets the verifier discharge the "maximum operand depth"
obligation statically instead of relying on a runtime stack limit.

### GLOBALS

```
count       u32
per entry:
  name_idx  u32
  type_tag  u8
  flags     u8      bit 0 = mutable
  _pad      u16
  init_idx  u32     CONSTANTS index, or 0xFFFFFFFF for zero-initialized
```

This section is the prerequisite for sizing globals dynamically. The VM
currently embeds `NanoValue globals[4096]` in every instance; with a serialized
count it allocates exactly `count` slots, and the verifier gains a real bound
to check `LOAD_GLOBAL`/`STORE_GLOBAL` operands against — today those opcodes
fall through to the verifier's `default:` "valid by decode success" arm.

### IMPORTS

```
count             u32
per entry:
  module_name_idx u32
  symbol_name_idx u32
  signature_idx   u32
  kind            u8    0=ffi 1=coprocess
  _pad            u8[3]
```

Parameter counts and type tags move into `SIGNATURES`, which removes the
variable-length tail that made v1 import entries awkward to validate. The
argument-count ceiling becomes a property of the signature table and is checked
once at load time — see #150 for the four inconsistent ceilings this replaces.

### LINKS

New. One entry per external call target.

```
count             u32
per entry:
  module_name_idx u32
  symbol_name_idx u32
  signature_idx   u32
  flags           u32   bit 0 = weak (may resolve to null)
```

`CALL_MODULE` takes a single link index rather than a module/function index
pair. Linking resolves each entry once into a typed call descriptor; the
descriptor is what #137's callable handles already build in memory, so this
section is its serialized form.

### METADATA

```
count       u32
per entry:
  key_idx   u32     CONSTANTS index
  value_idx u32     CONSTANTS index
```

Module name, version, source language, and producer. Free-form so that adding
a key is not a format change.

### DEBUG

Unchanged from v1 in spirit — `(bytecode_offset, source_line, source_col)`
triples — but with `u64` bytecode offsets to match `CODE`.

## Removed limits

v1 fixed several maxima at compile time. All become dynamic, sized from the
serialized counts:

| v1 constant | Value | v2 |
| --- | --- | --- |
| `NVM_MAX_SECTIONS` | 16 | `section_count`, bounded by `total_size` |
| `NVM_MAX_STRINGS` | 4096 | `CONSTANTS.count` |
| `NVM_MAX_FUNCTIONS` | 512 | `FUNCTIONS.count` |
| `VM_MAX_GLOBALS` | 4096 | `GLOBALS.count` |

`NVM_MAX_FUNCTIONS` at 512 is the most limiting of these in practice and is not
diagnosed when exceeded.

## Verification requirements

Loading is not trusting. A v2 module is rejected unless:

1. `magic` is `NVM\x02` and `format_version` is supported.
2. No unknown `feature_bits` are set.
3. `total_size` equals the actual byte length; `header_size` is 40.
4. The section directory validates — no overlap, no gap, no duplicate
   singleton, no trailing data, all arithmetic non-wrapping.
5. Every required section for the declared feature bits is present.
6. Every index is in range for its target table: constant, signature, layout,
   function, global, import, link.
7. Every layout's nested indices are lower-numbered.
8. `checksum` matches.

Only after these does instruction-level verification run. Several verifier
obligations that are currently unmet become mechanically checkable once this
format exists — global bounds from `GLOBALS.count`, call arity and result shape
from `SIGNATURES`, aggregate field counts from `LAYOUTS`, operand depth from
`max_stack`, and linked-call signatures from `LINKS` (`OP_CALL_MODULE`
verification is presently `(void)instr;`).

## Migration

Clean break. `magic[3]` becomes `0x02` and `format_version` resets to 1.

A v1 module is rejected at load with an explicit message rather than
misinterpreted:

```
module 'foo.nvm' was built for NanoISA v1 (NVM\x01);
rebuild it with nanoc 4.0 or later
```

No dual-path loader is carried. The break is what makes it possible to retire
the roughly 64 opcodes in `spec/nanoisa.yaml` that no frontend emits — the
`STRUCT_*`/`UNION_*`/`TUPLE_*` families superseded by `AGG_*`, the `MEM_LOAD*`
and `MEM_STORE*` families, `PICK`/`ROLL`, the polymorphic arithmetic retained
only for assembler compatibility, and the no-op GC scope operations. Keeping a
v1 read path would require keeping all of them decodable, which would freeze
the dead half of the instruction set into a major release.

Since `.nvm` files are build artifacts rather than distributed packages, the
rebuild cost falls on the build system rather than on users.

## Open questions

- **Section ordering.** Should the directory require sections in ascending type
  order? It makes validation a single pass and canonicalizes output, at the
  cost of forbidding a producer from streaming sections as it finishes them.
- **Constant deduplication.** Whether identical constants must be coalesced by
  the producer, or whether that is a size optimization the format leaves
  unspecified.
- **Signature interning.** Same question for signatures; deduplicating them
  makes signature-index comparison a sufficient equality test, which would
  simplify the verifier.
- **`max_stack` provenance.** Whether the producer computes it and the verifier
  confirms it, or the verifier computes it and the field is advisory. The
  former is cheaper to check; the latter cannot disagree with reality.
- **Weak links.** Whether `LINKS` really needs a weak flag before 4.4's
  capability work gives it a use.

## Roadmap items this closes

From Phase 12, "Module format and tools":

- Design a NanoISA v2 module header with format version, ISA version, feature
  bits, total size, and bounded section directory.
- Serialize required code, constants, signatures, globals, imports, layouts,
  links, metadata, and optional debug sections.

From "Runtime representation":

- Dynamically size globals from serialized declarations.
- Preinstantiate module constants.
- Consistently use stored string lengths and preserve embedded zero bytes
  (serialized half).

From "Module model":

- Choose one coherent flattened or separately linked module model.
- Resolve imports once into typed call descriptors.

It also unblocks, without closing, the verifier obligations that require
serialized bounds: global bounds, call arity, aggregate counts, operand depth,
and linked-module call signatures.
