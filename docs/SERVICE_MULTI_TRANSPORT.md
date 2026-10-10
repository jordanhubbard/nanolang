# My multi-instance service transport

I use service-binding version 3 to retain multiple nominal File and TCP
catalog instances in one NanoISA v2 module. I preserve the existing version 1
and version 2 encodings. My new table describes metadata; it supplies no host
authority and does not admit mixed executable code yet. I track flow, dispatch,
source lowering and grant integration separately under #990.

## Wire table

I encode integers in little-endian order. My header is 16 bytes:

| Offset | Width | Meaning |
| --- | --- | --- |
| 0 | 2 | Service-binding version, exactly 3 |
| 2 | 2 | Reserved, zero |
| 4 | 4 | Instance count, 1 through 64 |
| 8 | 4 | Instance entry size, exactly 64 |
| 12 | 4 | Reserved, zero |

Each 64-byte entry has the following fields:

| Offset | Width | Meaning |
| --- | --- | --- |
| 0 | 4 | Instance ordinal, equal to its table position |
| 4 | 2 | Immutable catalog identity: 1 File, 2 TCP |
| 6 | 2 | Reserved, zero |
| 8 | 20 | Five import indices, in catalog method order |
| 28 | 36 | Nine layout slots, in catalog type order |

File has eight nominal types. I require its ninth slot to be `UINT32_MAX`.
TCP uses all nine slots, including Endpoint. Every active import index is
unique across all instances; every active layout index is likewise unique.
I forbid `UINT32_MAX` in an active slot. Import and layout indices belong to
separate tables and may have equal numeric values.

My exact extent is `16 + 64 * instance_count`, with no trailing bytes. The
largest table is 4,112 bytes. These are explicit implementation limits, not a
claim that every possible service graph fits. I refuse larger tables before
allocation. I do not allocate during raw encoding or decoding. Encoding uses
a bounded staging buffer so input and byte output may overlap; decoding uses
a staged value. Failure preserves output buffers and the caller's size value.
The size pointer must be disjoint from input and output storage.

## Nominal identity

An instance ordinal distinguishes repeated declarations of the same catalog.
For each instance I require its exact five methods, signatures, named layouts,
member tags and prior-only layout edges. A member's nominal target must belong
to that instance. A same-shaped File in another instance cannot be substituted
into its OpenResult. Layout and import table order may differ from catalog
order; the explicit indices retain identity through either order.

I validate ownership-v1 flags and function descriptors against the complete
combined map. Each instance's owner and owned-result layouts carry the exact
resource flag; borrowed parameters identify that instance's owner. Unmapped
layouts retain no catalog authority. My query returns instance, catalog,
catalog ordinal, global layout index and per-kind source ordinal separately.
The returned plan owns its metadata and remains valid after the input module
is freed. I retain the 65,536-layout bound from the single-catalog validators.

## Module boundary

I validate the complete map before retaining service ownership through either
module bridge. Serialization and deserialization preserve service, layout and
ownership bytes. Module feature bits and ordinary structural validation still
apply. An unknown or malformed version cannot fall back to a different codec.

My general verifier, VM and native translator still refuse service execution.
Existing explicitly granted File and TCP execution remains single-catalog;
this table does not combine their grants. Mixed flow analysis, runtime values,
cleanup, paired source lowering and supervised publication remain necessary
before a mixed source program can execute.

I test mixed File/TCP/File modules, repeated-catalog identity, import/layout
permutations, cross-instance payload substitution, exact ownership, every raw
truncation at the maximum extent, single-byte mutations, overlap, malformed
serialized metadata with corrected CRCs, allocation failure and both bridge
directions with `make -f Makefile.gnu test-multi-nominal`. Sanitizers cover the
new codec/query and transport adapters; linked surrounding objects retain
their ordinary build flags. This is not a fully instrumented whole-program run.
