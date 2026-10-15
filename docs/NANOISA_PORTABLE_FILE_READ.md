# My portable text and byte reads

I accept `nvm2llvm --portable-file-read` and `nvm2wasm --portable-file-read`
for exact empty-namespace `file_read`, `vm_file_read`, `nl_os_file_read` and
corresponding `_bytes` imports. Every argument is STRING. Text results are
STRING; byte results are ARRAY with catalog-derived `u8` element identity.
My existing `--portable-read-text` option continues to reject byte imports.
Default closed profiles continue to reject imports.

I run bounded array-origin analysis for this profile, including ordinary
mutation, aliases, calls and graph lifetime selection. A declared byte reader
produces a fresh packed `array<u8>` origin. I reject unsupported imports and
incompatible array writes before publishing translated output. Runtime operand
checks still precede every host effect; the general structural verifier's type
report alone is not an execution proof.

I preserve binary NULs and all 256 byte values. I cap a portable read at
1,048,576 bytes, report excess as `NPR_LIMIT`, and return owned empty arrays for
missing files or read/close errors, matching my existing byte-reader semantics.
The managed result copies host scratch. Later reads cannot overwrite it;
ordinary aliases of the same result observe its mutations.

Native embeddings bind byte authority separately with `npr_module_bind_bytes`.
They can bind `{npr_file_read_bytes, host}` from a copied allowlisted
`NprFileHost`. A text binding supplies no byte authority. I revoke byte authority
with `npr_module_bind_bytes(NULL)` while inactive. Context lifetime and serialized
access remain obligations of my trusted embedding.

My Wasm file-read profile retains two exact imports:
`nanolang_host_v1.read_text` and `nanolang_host_v1.read_bytes`. Each accepts five
i32 offset/length values and returns an i32 status. I link only those two allowed
unresolved symbols. Both calls share the existing bounded workspace and busy
check; neither receives a native pointer or filesystem grant from the module.

My Node embedding accepts separate copied allowlists:

```javascript
const instance = createFileReadInstance(moduleBytes, textPaths, bytePaths);
const result = instance.call('nano_try_entry');
instance.call('nano_dispose');
instance.close();
```

My Python adapter exposes `create_file_read_instance(module_bytes, text_paths,
byte_paths)` with the same split. Its actual Wasmtime43 execution remains a
qualification requirement when the pinned engine is available. Syntax and
pure envelope-parser checks do not establish engine execution.

I install these adapters, headers, native archive and translator sources with
`make -f Makefile.gnu install-portable-read-runtime PREFIX=...`. The Wasm wrapper
finds the installed source package relative to its own prefix after relocation.

My self-hosted host catalog keeps `array<u8>` in source typing and emits the
coarse ARRAY wire tag. C AOT also recognizes these exact byte-read builtins and
copies their bytes into its owned numeric-array representation. Its builtin
filesystem authority remains the ordinary C product's host authority.

I test this integration in `tests/test_portable_bytes_execution.py`, alongside
read-text, packed-array, mutable-array and origin-analysis controls. Broader
aggregate results, filesystem write/metadata/directory calls, process/environment
capabilities, linked module graphs and compiler artifact contracts remain part
of my required 5.1 implementation scope.
