# NanoISA Primitive String and Aggregate Operations

Roadmap 4.0, Phase 12, Portable ISA design.

I keep a string or aggregate opcode in the portable ISA only when it is
justified by *representation* or by *measured cost*. Everything else is an
algorithm that composes from those primitives and belongs in a runtime
library, not in the instruction set. This document records that classification
so future changes have a written rule to test against.

## The two justifications

**Representation.** The operation reads or builds the value's internal layout in
a way a library written in NanoLang cannot express portably or safely: the
stored byte length, an indexed byte or element, an aggregate field or variant
tag, or the construction of the value itself. Removing such an opcode would
force every frontend and runtime to agree on private layout details, which is
exactly what a portable ISA exists to hide.

**Measured cost.** The operation is not strictly primitive, but the benchmark
harness (`make benchmark-nanoisa`, see `docs/NANOISA_OPTIMIZATION_POLICY.md`)
shows it on a maintained workload often enough that a dedicated opcode pays for
itself against the composed alternative, and the composed form would multiply
allocations or retained values. A measured-cost primitive must survive the same
acceptance criteria as any other NanoISA optimization.

An opcode that satisfies neither justification is a library algorithm. It is
lowered to a runtime call rather than a portable instruction (see the companion
roadmap item on moving trimming, case conversion, splitting, replacement,
formatting, parsing, and collection algorithms into runtime libraries).

## String operations (0x40-0x4F)

Retained as primitive:

| Opcode | Justification | Reason |
| --- | --- | --- |
| `STR_LEN` (0x40) | Representation | Returns the stored byte length; preserves embedded zero bytes that a `strlen`-style library scan would truncate. |
| `STR_CONCAT` (0x41) | Representation | Allocates and lays out a new string value from two payloads; the allocation and interning are runtime concerns. |
| `STR_CHAR_AT` (0x45) | Representation | Indexed byte access into the payload; the layout-aware bounds behavior is not portably expressible in a library. |
| `STR_EQ` (0x44) | Representation + measured cost | Length-prefixed payload comparison used pervasively (map keys, matches); the composed byte loop is measurably hotter and cannot short-circuit on stored length as cheaply. |
| `STR_FROM_INT` (0x46) | Representation | Constructs a string value from an integer; needs the runtime allocator and numeric formatting invariants. |
| `STR_FROM_FLOAT` (0x47) | Representation | Constructs a string value from a float; same allocator and formatting invariants. |

Reclassified as library algorithms (retired from the primitive set):

| Opcode | Composes from |
| --- | --- |
| `STR_SUBSTR` (0x42) | `STR_LEN`, `STR_CHAR_AT`, `STR_CONCAT` |
| `STR_CONTAINS` (0x43) | `STR_LEN`, `STR_CHAR_AT`, `STR_EQ` |
| `STR_TRIM` (0x48) | `STR_LEN`, `STR_CHAR_AT`, `STR_CONCAT` |
| `STR_TO_LOWER` (0x49) | `STR_LEN`, `STR_CHAR_AT`, `STR_CONCAT` |
| `STR_TO_UPPER` (0x4A) | `STR_LEN`, `STR_CHAR_AT`, `STR_CONCAT` |
| `STR_STARTS_WITH` (0x4B) | `STR_LEN`, `STR_SUBSTR`, `STR_EQ` |
| `STR_ENDS_WITH` (0x4C) | `STR_LEN`, `STR_SUBSTR`, `STR_EQ` |
| `STR_SPLIT` (0x4D) | `STR_CONTAINS`, `STR_SUBSTR`, array primitives |
| `STR_REPLACE` (0x4E) | `STR_CONTAINS`, `STR_SUBSTR`, `STR_CONCAT` |

None of the reclassified string opcodes touch layout that a NanoLang library
cannot reach through the retained primitives, and none is benchmark-justified as
its own instruction. They keep their assigned opcode values as assembler
compatibility instructions; no frontend needs to emit them once the equivalent
library word is available.

## Aggregate operations

The aggregate primitives are already layout-driven and every one is justified by
representation. There is nothing to reclassify here.

| Opcode | Justification | Reason |
| --- | --- | --- |
| `AGG_PACK` (0xFB) | Representation | Builds an aggregate value from a layout descriptor, variant, and field count; only the runtime knows the physical layout. |
| `AGG_GET` (0xFC) | Representation | Indexed field/element read from the packed layout. |
| `AGG_SET` (0xFD) | Representation | Indexed field/element write into the packed layout. |
| `AGG_TAG` (0xFE) | Representation | Reads the discriminant tag stored in the aggregate header. |

The named array (`ARR_*` 0x50-0x5F), struct (`STRUCT_*` 0x60-0x67), union
(`UNION_*` 0x68-0x6F), tuple (`TUPLE_*` 0x70-0x77), and hashmap (`HM_*`
0x78-0x7F) opcodes are the language-specific lowering that the regular
`AGG_PACK`/`AGG_GET`/`AGG_SET`/`AGG_TAG` set replaces. Among those, only the
element-count and container mutators that expose representation
(`ARR_LEN`, `ARR_GET`, `ARR_SET`, `ARR_NEW`, `ARR_PUSH`, `HM_LEN`, `HM_GET`,
`HM_SET`, `HM_HAS`) are representation-justified; the slicing, removal, key/value
enumeration, and search variants (`ARR_SLICE`, `ARR_REMOVE`, `HM_KEYS`,
`HM_VALUES`, `HM_DELETE`) are library algorithms over those primitives. They
remain only as compatibility instructions during the transition to the regular
aggregate set.

## Rule for future changes

Before adding a string or aggregate opcode, state which justification it claims.
A representation claim must name the layout detail that no composition of
existing primitives can reach. A measured-cost claim must cite a benchmark run
under `docs/NANOISA_OPTIMIZATION_POLICY.md`. An opcode that can prove neither is
a library word.
