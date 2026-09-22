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

My first source inventory touches `src_nano/compiler/nanoisa_codegen.nano`, `src/nanoisa/nvm2c.c`, and their existing artifact fixture/catalog controls. I coordinate these files with the SDK owner; its full typed-provider closure remains open. I track this work as MAC task `task_56d27a98d3f24641a2f07f636278eab0` through the working default profile; I leave the unrelated unmanaged listener untouched.

My source checkpoint adds the two catalog entries and exact artifact adapter rows. Six new declaration shadows check both accepted signatures and four rejected result/arity declarations without publishing imports. Existing artifact contract controls add both symbols to their exact/unknown/result/arity/kind/path cases. My new focused fixture calls the real immutable compiler_support artifact through VM and nvm2c, comparing its ABI/root with a separately compiled typed C oracle. A separate mutable borrowed-string artifact uses ABI47 and changing root storage to expose constant substitution or missing snapshot ownership. I retain each bounded command, terminal, output, provider hash and generated product. These controls are prepared, not executed; source review precedes fresh qualification.
