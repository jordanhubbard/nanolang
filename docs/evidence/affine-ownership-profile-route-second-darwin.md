# My corrected Darwin affine-route matrix stops at a Stage1 terminal

I retain the next fresh qualification outcome for
`task_bd8ebe2ba75457af1686f095275229dd`, its discovered dependency
`task_a613986ffa6f476293e3befa8d9accfd`, and installed-product task
`task_e8d860a16da0464891dd32e91c42bef1`. This run does not qualify the route
matrix and does not authorize release publication.

## My frozen input and qualified compiler provenance

- corrected source commit: `6cfba1e68a734d55deb01ae6219afc98c7b44d1d`
- pull request: `#827`
- detached checkout: `/private/tmp/nanolang-e8d-affine-profile-gate-6cfba1e6`
- evidence root: `/private/tmp/nanolang-e8d-affine-profile-evidence-6cfba1e6`
- private build cache: `/private/tmp/nanolang-e8d-affine-profile-cache-6cfba1e6`

I did not repeat bootstrap. I copied the three compilers, installed `nanoc`
selection and test-only C frontend runner from the retained fresh-bootstrap
gate at `254fa51e8dd49bb71b15c65406a355d172fcaf5d`. That earlier gate passed
bootstrap in372.52seconds; its bootstrap log SHA-256 is
`4cc55973e849966f1c7bf74271d7cb4e8f934a2d7c994f67496539886de2a17a`.
The copied files match its sealed five-entry manifest before preparation,
after preparation and after the matrix. The manifest SHA-256 is
`afafc77dc0f0615b796c0f55905486638842737487398f44fafa98cf04efcad4`:

```text
4d344016b2ac211b0bbde63fc16ee8a13e4cea6c3874efd58b51de77ca63323c  bin/nanoc_c
b3ac51df6bd3adc0014700c29a9572d80ce2fcab60c41f14d7147809c0ffce03  bin/nanoc_stage1
bf5a24a4c594b02e27c11725c6bd6d1a07519c58dbbc1d56eff7080f80e91e6d  bin/nanoc_stage2
bf5a24a4c594b02e27c11725c6bd6d1a07519c58dbbc1d56eff7080f80e91e6d  bin/nanoc
48458e4ab92adecaeac428e2f577e1909497131ebb0f07de5ee611b2f6c5e2a1  obj/test_affine_c_frontend
```

The corrected checkout has5,876 tracked inputs. Its complete before and after
maps are byte-identical; each map has SHA-256
`b42ba0117275c3f779baaf0ce94750642f39032a676d53033e8c2cd5dfd7633d`.
The checkout remained a clean detached tree. My ten selected compiler,
frontend and runtime-tool hashes are also identical before and after the
matrix; each selected-tool map has SHA-256
`f0e6fd425a27126279986fbe7928536cfa59125d73adb3e666c9dbf97108e212`.

I used GNU Make3.81, Python3.14.6 and Apple Clang21.0.0. `/usr/bin/cc`
selects
`/Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/clang`,
whose SHA-256 is
`1590ac950a3d627817d09ade5cb60b2115f17a72182a3141e010b4bcc482a0c9`.
The selected SDK path is
`/Applications/Xcode.app/Contents/Developer/Platforms/MacOSX.platform/Developer/SDKs/MacOSX.sdk`.

I built only `bin/nano`, `nano_virt`, `nano_vm`, `nvm2c` and `nanoisa_dump`
from the frozen checkout. Preparation passed in3.97seconds and did not rebuild
or replace the copied compilers or runner. The preparation log SHA-256 is
`e5f0f3adb6f4f691dfc0795852884a185bb39fdce0864707e713f1462994aedb`.

## My corrected matrix command

I ran the reviewed harness with one durable observation directory per
case/authority and fail-fast behavior:

```text
AFFINE_PROFILE_EVIDENCE_DIR=/private/tmp/nanolang-e8d-affine-profile-evidence-6cfba1e6/route-observations \
NANO_BUILD_CACHE=/private/tmp/nanolang-e8d-affine-profile-cache-6cfba1e6 \
python3 -m unittest -f -v tests.test_affine_contract_boundaries
```

The command stopped after2.54seconds. Twelve complete unittest methods passed.
The harness retained91 completed route observations across23 source cases:
23 C-frontends,23 public-C routes,23 Stage1 routes and22 Stage2 routes. Their
500-file manifest has SHA-256
`58607ae94004c1f736f3b573d1be939b37d73c21940ab4198e49ec8388db2b97`.
The full matrix log SHA-256 is
`594c14ef0c5268285aa6537c9839d900606b7f60d8daffc831015f55978710f7`.

## My first terminal result

The unchanged `owner_257` source has SHA-256
`86f1b89e0dbcf662f0d9da8ac41e9abca3e987e0f285fe14c69fbcb4b5968909`.
The test-only actual C frontend and C-seed public-C route both refused it as an
ownership negative. The next route invoked copied `nanoc_stage1` SHA-256
`b3ac51df6bd3adc0014700c29a9572d80ce2fcab60c41f14d7147809c0ffce03`.
It terminated after0.0563seconds with Python return code `-11`, empty stdout
and empty stderr. I describe that observed signal terminal; I do not attribute
a cause from it.

The route observation was written before the harness assertion. Both the
known prior artifact and observed output exist, contain14bytes and have
SHA-256
`399983caf28e4ace0eececef5319b89d0a449fd63985adcf5a7a9d200c909e3f`.
The prior output was therefore preserved. The exact observation JSON SHA-256
is `5b41bb5b36619f46093e1a448fd6465389ba7c724e2bf2ef609847ed1ecd2fb4`.

I did not replay the failed compiler invocation or execute its retained prior
artifact. I did not run the remaining13 source cases, Stage2 for `owner_257`,
the deeper callable/union public-C gates, checked-owner selection, aggregate
affine gates or an installed-product gate. I made no production correction.
Those gates remain ordered after static diagnosis, reviewed correction and a
fresh complete route matrix.
