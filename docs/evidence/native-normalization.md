# Native normalization storage

I replace the public module normalizer's fixed output buffer and the
generated-native normalizer's fixed component array with the shared
`nl_normalize_path` helper. It allocates output and offsets from checked input
sizes, frees partial allocations on failure, and preserves lexical root and
relative-parent behavior. It does not resolve symlinks.

The module still returns owned native storage. Generated-native code copies
the normalized bytes into GC string storage and frees the intermediate native
allocation. Its existing null-input behavior remains empty text; module null
input remains `.`. General null-input contract unification is not claimed.

## Verification

`make test-path-normalize` rebuilds the C seed and passes the new regression.
I link the complete filesystem module and execute an extracted generated-native
normalizer body with a GC-allocation stub. Both handle roots, unresolved parents,
700 components, 700 leading parents, 700 cancellations and a 5,000-byte component.
The extracted-body probe does not establish complete compiler/GC integration.

`make test-directory-walk` passes six cases in 3.958 seconds with one host
path-limit skip. `make test-nvm2c` passes 828 checks. `git diff --check` passes.

`path_relpath` still calls the old bounded private normalization helper and
uses its own bounded token/output arrays. MAC
`task_82bd388637824cc889b12204d226e75b` remains open until that caller is fixed
and verified. This checkpoint removes the public normalization limits and the
generated-normalizer component overwrite; it does not complete all path work.
