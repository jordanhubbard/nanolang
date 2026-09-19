# I stop the PR827 canonical integration at the retained owner-ARRAY boundary

I qualify the reviewed PR827 parser and test-harness changes after integrating
canonical `b69b60cde77e88787c5ea37eeff7def8acf62063`. This is bounded Darwin
evidence under `task_a613986ffa6f476293e3befa8d9accfd`, prerequisite parent
`task_bd8e2d91943ae17059d289e13ebc34fe` and installed-product parent
`task_e8d860a16da0464891dd32e91c42bef1`. It does not qualify a product
candidate or authorize publication.

## Frozen source and merge

The isolated checkout is:

```text
/private/tmp/nanolang-pr827-integration-b69.duGbtN
```

Its exact merge is `fd7884a5513a059ffe5a4bbcc7f9e307b9365fef`, with parents:

- reviewed PR827 evidence head `2ac0a1b7619252e9995329d521cd9aea52542fcf`; and
- canonical `b69b60cde77e88787c5ea37eeff7def8acf62063`.

The merge is clean. The reviewed parser, selector and affine-harness files are
byte-identical to the PR827 parent. The canonical typechecker, VM, NanoISA and
private owner-ARRAY runtime files are byte-identical to the canonical parent.
The additive Make targets and both roadmap histories are present.

The before, frozen and after maps each contain all6,659 tracked files. Their
map SHA-256 is
`fb29b8ebf606579ab8a3d7062ef58c5c9b1ad366c86be168e2a979dec8cafb7d`.
The checkout is clean before and after the gates.

I use Apple Clang21.0.0 at
`/Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/clang`
for the SDK-aware build. Its SHA-256 is
`1590ac950a3d627817d09ade5cb60b2115f17a72182a3141e010b4bcc482a0c9`.
I use Homebrew Clang23.1.1 only for the generated-native leak-checked method;
its SHA-256 is
`570c488e53383b198796e706e91b5ce5ec45bb730683a5af5e822d56a2eb1888`.

## Fresh bootstrap and frozen prerequisites

`make -j8 bootstrap` passes in406.34 seconds. Stage1 and Stage2 compile, run
their hello smokes, install Stage2 and pass the no-C-seed installed-compiler
smoke. The native Stage1 and Stage2 files differ; I preserve that diagnostic
and do not relabel this bootstrap as a fixed-point proof. The bootstrap log
SHA-256 is
`c5f35117f6984a9e6871ff7689753c0a13c29d99388bbea3fcf54fbf2a46b3de`.

I then prepare only the inventoried tools and C-frontend fixture:

```text
make -j8 bin/nano nano_virt nano_vm nanoisa_dump nvm2c obj/test_affine_c_frontend
```

That command passes in0.82 seconds. Its log SHA-256 is
`c05378ca4b6a290d4df3b470365bbd758d3dc842b14f2c25e8c584f01c6294e4`.
The selected-artifact map remains byte-identical before and after all gates;
both maps hash to
`5b5ec5f35860202dd72612cca74c54ac0b98b2f707c3ec1305f4032946f57824`.

## Complete matrix and affected compatibility

The complete unchanged affine contract module passes all20 methods in3.35
seconds. It exercises all36 source cases and writes144 route observations as
792 retained files. The test log SHA-256 is
`42492c5f5f3f84b34a6d50c7868dea9e479dcc7fb2dad25903beb8ab2bea0af6`.
The 792-entry route-file map is150,720 bytes and hashes to
`768eddbf5b72c57cf163753fde32eafddadad215481d0d646dc7bcf70b38885c`.

The planned changed-affected checks then produce these results:

| Check | Result | Elapsed | Log SHA-256 |
|---|---:|---:|---|
| native selector precedence | 1/1 pass | 0.12s | `c2ed167ec47db651eda51ebb35cf27c179964e1e59e650ff7f95c3b7f7b77c31` |
| checked-owner lexical selection with Homebrew ASan/UBSan/LSan | 1/1 pass | 218.65s | `198f5cb34f6270aeeffb0c69fff0b22481bbd74f90bf0c329c4a859daad2ad69` |
| affine frontend parity | 1/1 pass | 1.40s | `cc51f27af48e5427e478e4d1836cf2fd552a468fdd3fb20c67181a5b9eedaf42` |
| mutation builtin identity | 5/5 pass | 159.68s | `80f0fc2e567cf7ce168cb23d1d9a69e35a8994f49491c502a1bc741c5d94999e` |
| private owner-ARRAY VM/native integration | 1/1 pass | 9.86s | `0c32342173aff2d2a1e759fc0364a61623864f1c0e1335264dfca675526266f6` |

The checked-owner method builds fresh drivers in this checkout and passes real
Homebrew LeakSanitizer. Their path-qualified SHA-256 values are:

```text
nanoc_c     77780ad1381fbd881b83bb2dc4df1af604f065224cc7e82cd164c659d7e94a97
nanoc_stage1 900d7aad570ca5ee3b877a2dd7c3f506c8f528ba4f2b12a24047b8eacbad01ae
nanoc_stage2 c2f3bb29e3e1fd67e4bf5ca255d66f795e524d0acc17af309b71d4dbab6348d6
```

I do not compare those native bytes across checkout roots. The preceding
build-only evidence already proves that absolute imported-module paths enter
the driver, and it disproves the earlier `TMPDIR` hypothesis. This parser
fixture is not the deterministic bytecode/bootstrap acceptance gate.

## First terminal remains the owner-ARRAY activation dependency

The complete owned-record-pattern module passes five methods before fail-fast
stops its sixth method. C seed, Stage1 and Stage2 accept and execute
`test_ordinary_array_field_keeps_element_type`; NanoVirt then returns1 with:

```text
I could not compile shadows at line 4: I require owner records without managed array fields in my source borrow profile
```

The module runs48.17 seconds. Its log SHA-256 is
`d9f211c9ebe54d3c83aa1d5976d0d0aad3e56e88bc4a2766f2638abfbaf40c35`.
I stop there. I do not weaken the positive, exclude the NanoVirt route or
execute its rejected artifact. Public owner-ARRAY activation remains under
`task_01e144c13aad46faa8d05d6384270649` and parent
`task_430220ce190946518d404088533531b6`.

The retained raw evidence root is
`/private/tmp/nanolang-pr827-integration-evidence.RqfrQZ`. This result clears
the integrated parser/typechecker compatibility checks that precede the
owner-ARRAY case. It does not close the owner-ARRAY parent, installed-product
acceptance, full ownership parents or the release-publication hold.
