# My native array boundary

## Sanitizer toolchain preflight

My FFI gate first loads, unloads and reloads both native artifact fixtures in
a standalone process that does not link my VM. On this Darwin arm64 host,
Apple Clang 21 builds 2100.1.1.101 and 2100.3.34.2 fail that probe with duplicate
ASan global registration; the full suite can instead crash in registration.
Changing global dead-stripping mode did not resolve the tested failure.
Homebrew Clang 23.1.1 passes the probe and all 27 FFI tests with ASan/UBSan.
This comparison isolates the failure outside my VM; it does not identify an
upstream source-code defect or establish behavior on other hosts.

I select the passing installed toolchain explicitly, without disabling ASan
checks or changing production image lifetime:

```sh
ASAN_OPTIONS=detect_leaks=0 make \
  'CC=/opt/homebrew/opt/llvm/bin/clang -isysroot /Library/Developer/CommandLineTools/SDKs/MacOSX26.sdk -fsanitize=address,undefined -fno-omit-frame-pointer' \
  test-vm-ffi
```

Those paths describe this host. The compiler and SDK must exist on the machine
running the test. This command instruments the fixtures and test driver; make
does not rebuild existing VM objects merely because CC changes. A clean,
separate-object sanitizer build is required for whole-runtime coverage.

I also ran the FFI dependency closure from fresh object storage using that
compiler, with ASan/UBSan and UBSan halt-on-error enabled. All 27 FFI cases,
35 protocol cases, eight protocol fuzz checks and the focused SDL tests passed.
This instruments the compiled VM/runtime dependencies, not the operating
system or installed third-party libraries. Darwin leak detection is disabled.

To repeat that build on this host without reusing normal objects:

```sh
sanitizer_dir=$(mktemp -d /tmp/nanolang-ffi-sanitized.XXXXXX)
sanitizer_cc='/opt/homebrew/opt/llvm/bin/clang -isysroot /Library/Developer/CommandLineTools/SDKs/MacOSX26.sdk -fsanitize=address,undefined -fno-omit-frame-pointer'
ASAN_OPTIONS=detect_leaks=0 UBSAN_OPTIONS=halt_on_error=1 \
  make -j4 OBJ_DIR="$sanitizer_dir/obj" CC="$sanitizer_cc" \
  test-vm-ffi test-cop-protocol test-cop-fuzz
ASAN_OPTIONS=detect_leaks=0 UBSAN_OPTIONS=halt_on_error=1 \
  NANO_TEST_OBJ_DIR="$sanitizer_dir/obj" CC="$sanitizer_cc" \
  python3 -m unittest tests.test_sdl_image_array_boundary
```

Run this without another FFI test invocation: test executables and native
fixture paths remain shared even when the dependency objects are separate.

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
mailbox carries mutation and alias snapshots; the pipe transport still needs
migration before I claim isolated-call parity. `make test-ffi-array-copyback` checks
scalar/string conversion, aliases, growth, cleanup and injected allocation
failures. `make test-vm-ffi` also exercises array-bearing typed dispatch.

## Layout declarations

My collections exports `nl_hm_keys`, `nl_hm_values`, `nl_set_values` and JSON
export `nl_json_object_keys` declare the canonical array ABI. The
`test-collection-array-exports` gate checks their markers, string element
layout, empty results, set deduplication and snapshots after their source
objects are freed, normally and with ASan/UBSan.

These snapshot functions copy strings. My native array destructor still frees
only the pointer buffer, not those copies. The fixture explicitly frees its
known copies; I do not present that as runtime leak freedom. Reclaiming copied
strings must account for elements that escape their arrays before adding
automatic cleanup. That ownership contract remains open.

`make test-sdl-image-arrays` loads my production cleanup adapter with fake SDL
entry points and exercises VM dispatch in-process, through the mailbox, and
through a 2,000-element pipe call. Prefix cleanup clears duplicate slots outside
the prefix; repeated cleanup destroys each of two distinct handles once. The
focused ASan/UBSan run instruments the adapter, its native array runtime and
test driver, not every prebuilt VM object. This does not test real GPU texture
ownership or copies of a handle stored in independent arrays.

My call-envelope codec uses `CA`, version byte 1, and a value-count byte,
followed by values. Marker `0xff` followed by a prior value index represents
a top-level array alias. Requests contain arguments; replies contain argument
snapshots followed by the result. I bound the count to 17, reject forward or
non-array references and trailing bytes, and preserve one owned reference per
decoded slot. Reply application validates the original alias topology and
element types before publishing any array storage changes.

My mailbox single-call dispatcher uses this envelope in both directions.
Batches with array arguments use ordered single-call crossings, so a later
call sees earlier mutations. Scalar-only batches retain the packed fast path.
I test actual forked dispatch, returned aliases, repeated handle clearing,
invalid-result rejection and switching from a scalar batch to a single call.
My version-2 pipe transport uses the same envelopes and rejects old wire
versions. Its worker bounds request/reply payloads at 16 MiB and reports
encoding failure as an error, never an empty successful result. A forked pipe
fixture checks repeated calls with 2,000-element aliased arrays in both
directions. The standalone worker builds against the same request handler.

My parent pipe exchange uses one monotonic deadline across request writes,
response header and response body. It temporarily enables nonblocking I/O,
restores descriptor modes and the calling thread's signal mask, and contains
SIGPIPE from its writes. I test a full stalled request pipe, a partial reply,
and a dead peer. Failed exchanges reset the channel. Teardown closes the
request pipe and kills the owned worker after a 50 ms grace period if needed;
foreign code cannot extend that grace period by ignoring SIGTERM.

My default launcher creates mailbox signals and large-payload pipes for the
same worker. Requests that exceed the mailbox use the pipe without restarting
native state. A regression alternates 2,000-element cleanup calls over the
pipe with scalar mailbox queries, checking the same PID and persistent native
counts, including repeated cleanup.

A mailbox call whose reply exceeds the slot signals a pipe spill, then sends
the already-computed call envelope over its data pipe. The parent receives it
within the original deadline without issuing another foreign request. I test
a native function that grows one element to 2,000 and counts executions; two
calls produce counts one and two, preserve VM identity and keep the worker PID.
Variable-size batch results use ordered single calls to retain this spill path.
Replies beyond the 16 MiB limit or allocation failure still fail closed after
native side effects; I do not retry foreign execution or claim rollback.

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
