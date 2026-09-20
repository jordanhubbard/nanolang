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

Fresh Linux qualification now passes at the exact held source and current-main
integration pins. Mixed extension conjunction, resource payloads, the full
affine parent, PR522 and 5.1 publication remain open.

## My defensive backpatch checkpoint

Static review of the earlier producer found that my final
`UNION_VARIANTS` payload-length write did not check the compiler error state
after allocation-bearing contract appends. I did not reproduce a corrupting
runtime execution. At `ce90b62762270849aafd23d10d50e3e326233a0e` I route the
v3 path length, extension payload length and v2 path-version mutation through
one bounds-checked patch helper. Once contract growth fails I retain the first
diagnostic, stop variant-name work, skip every patch and keep the partial
buffer in local cleanup ownership instead of publishing it to the module.

The dedicated allocator replaces only contract-buffer `realloc`. The retained
fixture requires three growth allocations. I fail each allocation once, check
that no module is published, require my original `available contract storage`
diagnostic, then compile and free a complete module in the same process. The
ordinary gate and the Homebrew LLVM 23 ASan/UBSan/LSan gate both pass all three
failures and all three recoveries.

My first LSan run completed the contract checks and then reported 1,652 bytes
in 154 frontend/typechecker match-binding allocations created before contract
emission. I retain that first terminal and track it separately as
`task_b46f55c35c72bd063b9a27eaf93a4816`. The corrected sanitizer harness
disables LSan only while parsing and type checking, re-enables it before every
call to NanoVirt code generation and keeps ASan/UBSan enabled throughout. It
does not suppress contract allocations or close the checker task. This scope
is selected only by `NANOVIRT_TEST_CONTRACT_LSAN_SCOPE` in the explicit
Homebrew LLVM qualification; ordinary Apple-ASan builds do not infer LSan
support or change their behavior.

The corrected Darwin checkpoint also passes:

- all 90 NanoVirt code-generation tests;
- all four affine scalar-union source methods in 41.767 seconds; and
- all nine imported-union and qualified-module-identity neighbors in 13.749
  seconds.

| Retained input or log | SHA-256 |
| --- | --- |
| `src/nanovirt/borrow_codegen.inc` | `62d427145126aa89625a4413d7b6d50571056de658346732b1518e8759395eaa` |
| `tests/nanovirt/test_borrow_contract_allocation.c` | `97607f4859cfa69b21269dea21ebab2147141095535a352641d0f86dc2bc2645` |
| `Makefile.gnu` | `16d44cef5aec796f7a5ee1ae3be0f34eebf9d0fc3f6e62c846346ec68deb85b5` |
| `/private/tmp/nanolang-affine-union-backpatch-sanitizer-final.log` | `4b5a7b281f514e3ff6179b47a36f0bdbe13b363c22955143fbd0a3d35af14b88` |
| `/private/tmp/nanolang-affine-union-backpatch-nanovirt90.log` | `aabcf1a54c161c88fc594c59a6b0eabb8643a39a02b5ad2c6595257034e8f09d` |
| `/private/tmp/nanolang-affine-union-backpatch-source.log` | `2b2f2c47044e720524915c46ff50e4e1a1cb91bbd43c68a6d8cf72201d0a2412` |
| `/private/tmp/nanolang-affine-union-backpatch-adjacent.log` | `1c0fe60280ff4353ba4c4c90c93429305c74ccc43e60f88dc8a3253b5a6f9676` |
| Homebrew Clang 23.1.1 | `570c488e53383b198796e706e91b5ce5ec45bb730683a5af5e822d56a2eb1888` |

## My fresh Linux and current-main integration

The independent read-only source review passes the imported declaration,
concrete-instance, match-lowering and guarded-cleanup scope. At exact PR917
head `c15e91de2cf5097bc7cf28e55515819c12e7926b`, a fresh isolated Linux tree
passes bootstrap in 279.54 seconds, all three injected contract allocation
failures and recoveries, 90/90 NanoVirt methods, all four source methods and
all 25 module/generic identity neighbors. Its 28,584 tracked entries, 20,337
provider/source entries and nine selected tools remain unchanged at their
recorded boundaries. MAC evidence `ev_14bee9b6eedd4c40bc3cef22df5d830f`
records report SHA-256
`7e6f971062bbf4e943194071b623980c9b1bbcfb12ac30bee08732373941fa17`.

I then replay the qualified commits onto canonical base
`8ed0a0ff6e30969b9721aa0cf89db011ec4f4506` without replacing the paired File
parser or shared ARRAY_FIELDS work. Production pin
`b8c7f75487da7754f92b7be1a38c5375476a0467` passes fresh bootstrap in 281.35
seconds, all three contract failures and recoveries, 90 NanoVirt methods,
546 ordinary and 856 allocation-path affine checks, ownership contracts,
non-admitting ARRAY_FIELDS/declaration/record-array neighbors, all 25 identity
methods and all four source methods. I retain two setup terminals separately:
the counted-runtime package selected Clang without its documented GCC 13 path,
and my first direct source command omitted `nanoisa_emit` and `nano_virt`.
The corrected failed package method passes in 5.235 seconds; the prepared
source gate passes in 38.668 seconds. No failed output was executed.

The integrated report SHA-256 is
`11e54ec049c29ceda0fe6882804db8da22cb57764c3ad61ab20de233d4af1bc6`;
its manifest SHA-256 is
`76ca90161ef0c0a646acd9b63098e6659c4902671a50ab0fb84bdd52ea2c8348`.
All 57,287 tracked source entries, four compiler tools and five source tools
remain equal at the recorded boundaries.

The zero-payload parser cleanup is now present through PR927; this slice adds
no empty lexical binding or new affine authority. The frontend binding leak,
complete kind 1 plus kind 2 conjunction, resource payloads, full affine parent,
PR522 and release publication remain open.
