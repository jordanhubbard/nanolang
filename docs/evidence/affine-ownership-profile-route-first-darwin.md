# My first Darwin affine-profile route outcome

I retain the first fresh qualification outcome for
`task_e8d860a16da0464891dd32e91c42bef1` and its discovered dependency
`task_bd8ebe2ba75457af1686f095275229dd`. This run does not qualify the route
matrix and does not authorize release publication.

## Frozen input

- source commit: `254fa51e8dd49bb71b15c65406a355d172fcaf5d`
- pull request: `#827`
- detached checkout: `/private/tmp/nanolang-e8d-affine-profile-gate-254fa51e`
- evidence root: `/private/tmp/nanolang-e8d-affine-profile-evidence-254fa51e`
- private build cache: `/private/tmp/nanolang-e8d-affine-profile-cache-254fa51e`

The before, frozen and after maps each contain5,874 tracked files and are
byte-identical. Their manifest SHA-256 is
`53c6cfa297e3d0d769aca0f7bb8c5d214215d71738948446066c0212a90e3e7c`.
The checkout remained a clean detached `254fa51e` tree.

I used GNU Make3.81, Python3.14.6 and Apple Clang21.0.0. `/usr/bin/cc`
resolved to
`/Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/clang`,
whose SHA-256 is
`1590ac950a3d627817d09ade5cb60b2115f17a72182a3141e010b4bcc482a0c9`.
The SDK path was
`/Applications/Xcode.app/Contents/Developer/Platforms/MacOSX.platform/Developer/SDKs/MacOSX.sdk`.

## Preparation

Fresh `make -j8 bootstrap` passed in372.52seconds. Stage1, Stage2 and the
installed compiler smokes passed, including installed operation with
`bin/nanoc_c` temporarily unavailable. The native Stage1 and Stage2 binaries
differ, as the bootstrap reports; I make no fixed-point claim from this run.
The bootstrap log SHA-256 is
`4cc55973e849966f1c7bf74271d7cb4e8f934a2d7c994f67496539886de2a17a`.

The reviewed test-only runner then built under `-Wall -Wextra -Werror` in
0.54seconds. Its build log SHA-256 is
`2ea1a582a65c119378e5de8c954c807ed3a489df58467476e95c07ce2af2bd6a`.
The selected tool manifest before the matrix contains:

```text
4d344016b2ac211b0bbde63fc16ee8a13e4cea6c3874efd58b51de77ca63323c  bin/nanoc_c
b3ac51df6bd3adc0014700c29a9572d80ce2fcab60c41f14d7147809c0ffce03  bin/nanoc_stage1
bf5a24a4c594b02e27c11725c6bd6d1a07519c58dbbc1d56eff7080f80e91e6d  bin/nanoc_stage2
bf5a24a4c594b02e27c11725c6bd6d1a07519c58dbbc1d56eff7080f80e91e6d  bin/nanoc
48458e4ab92adecaeac428e2f577e1909497131ebb0f07de5ee611b2f6c5e2a1  obj/test_affine_c_frontend
```

The before and after selected-tool maps are byte-identical. Their manifest
SHA-256 is
`afafc77dc0f0615b796c0f55905486638842737487398f44fafa98cf04efcad4`.

## First terminal result

I ran:

```text
python3 -m unittest -f -v tests.test_affine_contract_boundaries
```

The command stopped at its first failure after0.10seconds. The test-only C
frontend accepted the unchanged `both_arms` source. The next authority,
`nanoc_c --target c`, returned1 with:

```text
Warning: Function 'probe' is missing a shadow test
[c_backend] I require a supported exact C value representation.
```

The matrix log SHA-256 is
`f1e35284e18df6af906c1dada28129e06977019679a3bd251771190623ffedcb`.
No later case, Stage1/Stage2 explicit-C route or canonical selection gate ran.

The failed assertion compared the unexpected return status before checking
the known prior output. The temporary case directory was then released. I
therefore have no artifact evidence from this run that the prior output was
preserved. I do not infer preservation from backend implementation. A reviewed
harness correction must make that assertion observable on an unexpected
refusal before the next qualification.

Static inspection after the terminal result found the same diagnostic at
`src/c_backend.c:153`; `cb_node_storage` refuses every resource
`AST_STRUCT_DEF` at lines206-207. Every exact affine case retains the shared
`resource struct FileHandle` prefix. This observed route is therefore broader
than the approved two-entry public-C exception table. I have not changed that
table, the production backend, a source case or an ABI.
