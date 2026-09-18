# My private packed scalar array foundation

I build on my boxed leaf-value core under parent 488. This is storage and
coercion machinery, not an executable profile or opcode extension. My static
VM audit is src/nanovm/heap.c packed_store/packed_load: int and float occupy 8
bytes, bool and U 8 occupy 1, and reads materialize the declared element tag.

My private packed writes accept this exact matrix:

| Destination | Input | Result |
| --- | --- | --- |
| int | int | Identical 64 payload bits |
| int | U 8 | Unsigned byte zero-extended |
| U 8 | U 8 | Same byte |
| U 8 | int | Low 8 bits, including negative input modulo 256 |
| float | float | Identical binary 64 bits, including NaN and signed zero |
| float | int | Signed integer to binary 64, round-to-nearest ties-to-even |
| bool | bool | Canonical 0/1 value |

I require canonical bool/U 8 input payloads for those input tags. Other pairs
return TYPE in this private API before changing storage or references. The VM's
other packed_store arms read raw union members and assume matching types; I do
not infer a portable contract for those fallbacks or heap pointer bits. Before
later opcode admission, static element-shape eligibility must prove a supported
write or a separate semantics prerequisite must resolve it. This bounded core
does not replace VM-accepted behavior with new runtime checks in published code.

A PACKED_SCALAR_ARRAY descriptor retains its element_tag and buffer capacity.
Generic NmsValue append/set/get/pop/length operates on it without promoting to
boxed storage. Get/pop preserve its declared tag; missing get or empty pop is
zero/VOID. Private out-of-range set remains STATE with no mutation, pending the
separate matched opcode status adapter. Packed arrays own no child references.
Legacy string-only accessors refuse this kind. Existing split and boxed arrays
retain their established behavior.

I validate tags and conversions before allocation. Width is 1 or 8; widened
capacity/byte arithmetic and allocate-copy-commit growth preserve handle aliases
and old contents on failure. Matching float bits are copied, never evaluated.
Signed integer conversion avoids implementation-defined uint 64-to-int 64 casts;
ordinary finite controls pin default nearest-even rounding on native and Wasm.
The private allocator uses aligned storage; buffer bytes need not expose a host
union representation. Descriptor table copies preserve element_tag and teardown
clears it. Capacity bytes contribute to accounting, but no child traversal runs.

I require paired ordinary VM reference conversions and actual native LLVM/Wasm
core checks for integer endpoints, all bytes, negative byte wrapping, bools,
binary 64 payloads and integer precision/tie boundaries. Native sanitizer and
import-free Wasm gates cover alias updates, table/buffer growth, failed allocation,
unchanged refused output, reclamation, independent instances and terminal disposal.
Production package hashes/ABI and all current split/string/leaf acceptance pass.
No failed historical artifact or mismatched-union execution is needed.

Parent 488 retains unresolved raw-write semantics, element-shape eligibility,
nested/nominal child tracing, cycles and full mutable opcode lowering. Parent 51 da,
Darwin sanitizer 7 ba and evaluator 791 a remain open.
