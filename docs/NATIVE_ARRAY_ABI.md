# My native array boundary

I currently use native array ABI version 1. `DynArray` still has a one-byte
element width; I reject records larger than 255 bytes. This change does not
remove that limit.

An array-bearing C export declares its compiled layout version beside its
definition:

```c
#include "runtime/dyn_array.h"
DynArray *example(void) { /* implementation */ }
NANO_EXPORT_ARRAY_ABI(example);
```

For NanoVM artifact imports with array parameters or results, I resolve
`example__nano_array_abi` through the selected artifact handle. I require the
declaration and function to belong to the same loaded image, and the declared
version to equal my runtime version. An absent declaration means legacy
version 1; it will not inherit version 2 when my layout changes. I reject a
mismatch before foreign execution and cache the failed call descriptor.

My C-seed native emitter checks direct and qualified array-bearing extern
calls, and unqualified extern function values before exposing their addresses.
It looks up the exported declaration in the process symbol namespace and
checks it against the function address supplied by the native linker. A
declaration from another image is rejected; I do not rebind the function.
Static archive declarations must remain exported by the executable. My Linux
native link already uses `-rdynamic`; my Darwin fixture checks archive exports.
Hidden or stripped declarations count as absent, not as verified metadata.

This is a trusted declaration by C code, not proof of its memory safety,
signature correctness, pointer provenance or ownership. Self-hosted native
emission, the separate C-source backend and VM legacy logical imports still
need corresponding coverage before I widen the layout. Qualified extern
function-value syntax currently fails typechecking; qualified calls work.

`make test-array-abi-loader` checks matching and mismatched declarations,
unmarked legacy artifacts under versions 1 and 2, missing modules, and a
declaration supplied by another dependency. `make test-vm-ffi` also checks
actual VM dispatch: matching and legacy array results succeed; an incompatible
function whose body aborts is rejected without entering it.

`make test-native-array-abi` checks C-seed generated executables using matching,
unmarked and incompatible shared libraries, including array arguments and
results, qualified calls and unqualified function values. It also checks
static archives and rejection of unsupported qualified function-value syntax
before executable publication. The loader fixture exercises the native guard
against missing version-2 declarations and wrong-image declarations as well.
