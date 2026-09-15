# My compiler filesystem AOT boundary

The compiler's `fs_walkdir` import is bound to an exact retained library
artifact. Its declaration returns `array<string>`; the native export returns
`DynArray*` and publishes the native array ABI version. My AOT string arrays
use a different representation (`nsarr_t`). A name-only replacement would
discard artifact identity, and a pointer cast would not adapt the storage.

I correct the translator diagnostic to distinguish artifact imports requiring
an exact library binding and typed value adapter from builtin signature
mismatches. I retain rejection; the adapter is not implemented in this change.

The AOT tests now pin the filesystem array import with an explicit artifact
path, and distinguish artifact rejection for every existing builtin adapter.
`make test-nvm2c` passes 617 checks with zero failures. The subsequent
`make test-one-ir-compiler` still fails at `fs_walkdir`, now identifying the
artifact-backed boundary. Its downstream native compiler execution is not
reached. Logs:

- `/tmp/nanolang-aot-artifact-diagnostics-final.log`
- `/tmp/nanolang-one-ir-artifact-boundary.log`

The shared directory walker returns copied strings; native array element
ownership and escaped-element lifetimes remain tracked by
`task_93bb44374587a757753418fc28c2095d`. I do not introduce an unsafe finalizer
or relax artifact binding to bypass that work. Exact artifact loading, typed
native-array adaptation and compiler execution remain under
`task_419c47bdc8fc42e4b52eb6af1a0e9a71` and the release parent.
