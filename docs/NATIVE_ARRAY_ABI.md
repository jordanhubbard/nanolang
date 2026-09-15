# My native array boundary

## In-process VM arrays

I copy int, float, bool, u8 and string array arguments into call-scoped native
arrays. Repeated arguments share one native array; a returned argument maps
back to the original VM array identity. I convert the result and all modified
arguments before publishing changes to those VM arrays. Conversion failure
leaves their storage unchanged, but cannot undo native side effects.

I release temporary arrays and writable string snapshots after the call.
Native code must not retain or free these borrowed arrays or snapshots. String
elements must be NUL-free UTF-8; native null string elements become empty
strings. Replacement native string pointers must remain readable through
copy-back. I do not free independently returned arrays or replacement strings:
their ownership contract remains separate work. Metadata checks do not make
arbitrary foreign pointers safe to dereference.

I reject nested and record array layouts at this boundary. My co-process
transport still needs mutation and alias propagation; this in-process frame
does not establish isolated-call parity. `make test-ffi-array-copyback` checks
scalar/string conversion, aliases, growth, cleanup and injected allocation
failures. `make test-vm-ffi` also exercises array-bearing typed dispatch.

## Layout declarations

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

Legacy VM logical imports retain their existing function-resolution policy.
After resolution I check the declaration in the process namespace against
that actual function address. I do not use the declaration to select another
function. These calls also reject incompatible or wrong-image declarations
before execution and cache failure.

My C-seed native emitter checks direct and qualified array-bearing extern
calls, and unqualified extern function values before exposing their addresses.
It looks up the exported declaration in the process symbol namespace and
checks it against the function address supplied by the native linker. A
declaration from another image is rejected; I do not rebind the function.
Static archive declarations must remain exported by the executable. My Linux
native link already uses `-rdynamic`; my Darwin fixture checks archive exports.
Hidden or stripped declarations count as absent, not as verified metadata.

My self-hosted native emitter applies the same guard to direct and qualified
calls and unqualified extern function values. Local function variables keep
their own binding. Its native driver selects loader/export flags for Linux
and FreeBSD after identifying the host; an unsuccessful host query stops
compilation. I test the Linux flag selection, not Linux execution, on Darwin.

This is a trusted declaration by C code, not proof of its memory safety,
signature correctness, pointer provenance or ownership. The separate C-seed
C-source backend still needs corresponding coverage before I widen the layout.
Qualified extern
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

`make test-selfhost-array-abi` exercises the same shared/static fixtures with
my self-hosted compiler, plus a local function variable that shadows an
array-bearing extern. Both compilers currently reject qualified extern
function-value syntax before output, with different frontend diagnostics.
