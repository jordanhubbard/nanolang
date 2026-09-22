# My compiler-support artifact adapters

My609 Darwin bootstrap builds Stage1 and passes hello, then canonical Stage2 refuses `nlc_native_array_abi`. The same catalog lacks `nlc_runtime_root`. My separate Linux ten-second shadow terminal does not establish this cause.

Before implementation, I retain the existing owner-bound artifact-import contract. `nisa_register_artifact` validates exact result and arity, resolves the declaration owner through module bindings, builds that owner's immutable artifact, and emits `NVM_IMPORT_ARTIFACT`. I add only these two contracts:

| Actual symbol | Parameters | Result | Native result owner |
| --- | --- | --- | --- |
| `nlc_native_array_abi` | none | int | scalar |
| `nlc_runtime_root` | none | string | independent snapshot of provider's borrowed thread-local string |

I call the actual compiler_support providers. The first reads `NANO_DYN_ARRAY_ABI_VERSION`; the second performs existing SDK preparation and installed/in-tree root discovery. I do not replace either with a constant, pathname guess, builtin-name alias or arbitrary foreign admission.

My Nano catalog needs exact zero-argument arity/result entries. My nvm2c artifact table needs corresponding exact typed adapter rows. Its existing generated adapter opens the selected absolute artifact, resolves the actual symbol, invokes its typed zero-argument signature, and snapshots the root before another provider call. My VM and COP descriptor dispatch already support zero-argument integer/string results; string marshaling creates a VM-owned copy. The wire schema, import kind, provider implementation and general FFI admission remain unchanged.

I require additive exact/wrong result, arity, import-kind, relative-path and unknown-symbol checks. Actual-provider controls must build compiler_support, exercise both helpers through VM and generated native C, compare ABI to the real provider, compare root to selected SDK discovery, and retain one root result across subsequent calls. Catalog shadows must preserve owner binding and unchanged-output refusal. Fresh full canonical bootstrap and the original public source suites remain separate acceptance gates.

My first source inventory touches `src_nano/compiler/nanoisa_codegen.nano`, `src/nanoisa/nvm2c.c`, and their existing artifact fixture/catalog controls. I coordinate these files with the SDK owner; its full typed-provider closure remains open. MAC access is currently blocked by an unmanaged local port34113 and I do not disturb that listener.
