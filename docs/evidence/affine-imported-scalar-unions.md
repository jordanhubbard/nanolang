# I retain imported scalar-union identity through source emission

I qualified the Darwin checkpoint at production commit
`3d81a28c7dd54716e198371b3e0266411f42c889`. This is the bounded imported
source child of `task_a18a9f752536469faafc4d3ebec01dfd`; it is not the mixed
`UNION_VARIANTS` plus `ARRAY_FIELDS` gate and it is not a release claim.

The complete source suite passed four methods in 42.834 seconds:

```text
test_distinct_instances_verify_and_execute_in_vm_and_native ... ok
test_imported_aliases_and_concrete_instances_execute_in_vm_and_native ... ok
test_imported_identity_and_shadow_refusals_preserve_prior_output ... ok
test_refusals_preserve_prior_output_on_every_frontend ... ok
```

The imported acceptance compiles the same dependency and root with C-seed
NanoVirt, Stage 1 and Stage 2. All dependency and root shadows run. Each
published module verifies, dumps both concrete union instances, executes in
NanoVM, translates through nvm2c and runs under strict Clang ASan/UBSan with
leak detection. The two self-hosted artifacts were byte-identical at the
focused checkpoint:  `e0d5ba38658c789427968dc297d111f9f896382201cfdbf99bf6c076ca1f0b98`.

The refusal matrix covers incompatible concrete instances, wrong substituted
payloads, a separate same-spelled dependency declaration and a failed
dependency shadow. Every C-seed, Stage 1 and Stage 2 refusal retains the prior
artifact bytes. C-seed currently refuses two same-spelled dependency unions at
module loading; the self-hosted path retains distinct generated declaration
identities and refuses the cross-declaration value. Neither route silently
unifies them.

## My retained identities

The full log is retained at
`/private/tmp/nanolang-affine-imported-union-3d81a28c7.log` with SHA-256
`2c86c8cc60e41ac4c3e169804905cfd9f5df14d47e45cd9e49bd19a7530f7f3a`.

| Input | SHA-256 |
| --- | --- |
| `bin/nano_virt` | `4069a28290a7bcb2adcd4f8a4656982414f7a790b1765b17f01f407ef9c61817` |
| `bin/nanoc_stage1` | `831115e4eb7c93b9381f020715718c788b5dccefbbb1eb30ba6158b119d01a98` |
| `bin/nanoc_stage2` | `61246e5eb4b08ae672eef07c22f7d60a506b37262c6203f262dc4132753b1dcb` |
| `bin/nano_vm` | `9eb65ab3651889e38df99112d42b573f4db83e25d2d29cb526496cc868b4acf7` |
| `bin/nvm2c` | `6c1e186668b4b1380b39e1ad508186ee80bac78db3107702817798650b37431f` |
| `bin/nanoisa` | `a2522cacba0434f68cfa1c13d45c65e2b80503a1019e4ea29423220d34addd26` |
| `src/nanovirt/borrow_codegen.inc` | `2d4571e3c2fefa65ccf85647ddf3acd3d88ae28b73a98babec97c75d1e3de36a` |
| `src_nano/compiler/nominal_bindings.nano` | `35d59ee4fc3dd436043f056dda6d91401d49ec1bbb98c37953f2248e480b0c34` |
| `src_nano/compiler/nanoisa_borrows.nano` | `62fe8495c9c26075bb2a8910a645e01a99b17b41162c6ba387e36ef278b8923c` |
| `tests/test_affine_scalar_union_source.py` | `d2de3bed6f1078c1b90d69a00bc2ddea78bfdac50edf64be14ccc16327b24040` |

Fresh bootstrap completed before the final source gate. Stage 1 and Stage 2
both compiled and ran the hello smoke; the native compiler binaries differed,
which the bootstrap reports without claiming fixed-point equality. The
canonical bytecode fixed-point gates remain separate release evidence.

Linux qualification, mixed extension conjunction, resource payloads, the full
affine parent, PR522 and 5.1 publication remain open.
