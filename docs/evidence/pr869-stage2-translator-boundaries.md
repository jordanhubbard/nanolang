# PR869 Stage 2 translator boundaries

I qualified the bounded Stage 2 shadow-deadline and NanoISA-to-C repairs on
ARM64 Linux. I did not run a product `test-quick`, prove a new fixed point,
merge PR522, or authorize a release.

## Frozen inputs

- Product integration base: `67e3e211eab405959c11e6deaf424f6151d3214a`
- Qualified production/test commit: `3534d3afc3f28bc4575a7633a01fc89aff3ea173`
- Integrated PR869 production/test commit: `723c96830008377ad4ff8a62ef374c68e81d3cdb`
- Both commits have stable patch ID
  `9ffd58456444adcc37e982aaa4d0d0ce1da58d8b`; the commit hash changed only
  when I applied the qualified patch after PR869's evidence-first commits.
- Tree: `48ac3e6a48927486e3a90ad463938989f24b718c`
- PR861 merge `59ffceccf422047c91395e1cf5d08e62e5a1d64c` is an ancestor.
- Clean gate tree: `/tmp/nanolang-pr869-final-gate.ZZVNzr/repo`
- The tracked tree was clean after the gates. I removed only the generated,
  untracked `tests/nanovm/test_vm` binary left by the known failing Make target.

Selected host tools were:

| Tool | Version | SHA-256 |
| --- | --- | --- |
| `/usr/bin/cc` | GCC 13.3.0 | `a20520ee21543f243d40636a9181a142c45ecd989de31ab86b99a8ea5ada870d` |
| `/usr/bin/make` | GNU Make 4.3 | `abbecc214fcadef6530d1cc137bde7ceb74cae3f6b992c7d10a8669a7314c7e0` |
| `/home/jkh/.local/bin/python3` | Python 3.14.7 | `088ab609cb7433b189a75cc8c355b6b5b91352d3ed4ba3bac8402c3e3bea68d6` |
| `/usr/bin/git` | Git 2.43.0 | `aa6540695d076182256dd6e96c8b302e4d56381e3000bbfd5c71bbdfe94a4942` |

## Corrections

I verify a complete immutable shadow module before starting its unchanged
10-second execution supervisor. I carry that exact proof into VM
initialization, classify ordinary graphs once, and invalidate the fast path
when modules change. Owned/resource routes still require their existing
admission proof.

I retain finite exact scalar tags when a boxed join or exact global physically
stores the same scalar representation. I keep non-Boolean boxed scalar globals
dynamic, retain Boolean-array element facts through checked array operations,
and keep unrelated heap/scalar/VOID joins refused.

The compiler also reaches one exact physical array-element conversion from
`optional T` to `T`. I admit that conversion only for an immediate array
element with a present, matching payload. Direct optional-to-exact conversion,
a missing payload, a mismatched payload, and an optional field nested inside a
record remain refused.

## Preserved first terminals

I retained these outcomes instead of relabeling them:

1. The original Stage 2 execution budget included 11.83 seconds of whole-module
   verification and timed out before shadow execution.
2. After moving verified work outside the supervisor, translation reached the
   exact function-33 boxed join refusal.
3. The first array-shape draft leaked exactness through recursive records:
   1,362 assertions passed and 42 failed. The corrected immediate-edge rule
   passes 1,412/1,412.
4. My first manual native smoke omitted `nano_aot_runtime.o` and Linux
   export-dynamic flags. `libstd.so` therefore could not resolve its intentional
   `gc_*`/`dyn_array_*` host imports. I did not execute that failed binary again.
5. The first clean scalar-suite invocation lacked `bin/nanoisa`; all cases
   stopped before assembly. I built the selected assembler/VM/translator tools
   and used new test artifacts.

## Passing gates

- `make -j8 test-nvm2c-shapes nvm2c nanoisa_dump nano_vm`: 1,412 shape
  assertions passed.
- `python3 -m unittest -v tests.test_native_scalar_joins`: 11 methods passed
  in 1.546 seconds in the development tree and 1.557 seconds in the clean gate
  after its explicit tool prerequisites.
- The fresh 457,176-byte compiler module
  (`1474811c82e478362410de46e2bd456b9240dc9953d5b3ceb0963adbae4d2fef`)
  translated in 7.02 seconds with exit 0.
- Strict generated-C compilation with the canonical AOT runtime and Linux
  export closure passed in 1:43.34. The rebuilt compiler passed `--help`,
  compiled unchanged `examples/language/nl_hello.nano`, and the result printed
  `Hello from NanoLang!`.
- Fresh `make -j8 bootstrap` passed in 4:41.02 with 6,488,488 KiB peak RSS:
  Stage 1, Stage 2, both hello smokes, installed-compiler smoke, and the
  no-C-seed check passed with mandatory shadows and the default deadline.
- The NanoVM target reached 271,506 passing assertions, including every new
  proof-carrying/ordinary-graph control. Its sole failure is the pre-existing
  stack-slice underflow fixture (`expected 2, got 5`), already reproduced on
  the untouched baseline; I did not change it or call this target green.

Final selected binaries were:

| Artifact | SHA-256 |
| --- | --- |
| `bin/nanoc_c` | `d13397636390a7ec4d686a63b68ec20133d643bef837941232866f0317f6ff92` |
| `bin/nanoc_stage1` | `05f69b53c417b87d48ef6c35c330607a1aaa9ff70b1ff3e14c26b7f110a68109` |
| `bin/nanoc_stage2` | `ba197ec062a33fa8cb91a0acf2cf71696a70592119137ed931491ccb94e4e498` |
| `bin/nano_vm` | `4a96952e78c9442a2d0c556e6eeed90d57fb1427937a0fbcc4aeb33e6ba638dc` |
| `bin/nvm2c` | `6da85dc272265e4a306dc104f1d5e9a794f0ddde9387724a35f9011c7c99cd5b` |
| `bin/nanoisa` | `0b9c7576eba9ea0b42f8d1b56dc79044c03e6a03a1c41703f65a162e2c473ddb` |

## Raw-log seals

The retained Sparky logs have these SHA-256 values:

- corrected shape build: `220d232697bffc05a8224d4f0e8283d8d5f5fb9ae7d3dc1fd086883be3da6dc8`
- corrected development scalar suite: `7bcd6260ff3dd99968437df17ecc16feaf8ab37d5cc84a545b0c4c6229c26cc8`
- compiler translation: `25e74743137da8e291ae4859515e92ba3a1cc1435984bebe44e0b84b3cef4044`
- canonical native link: `c43daa9b9c823d54d234c104d10a1d23324cd0c733a3b717cf0cc5898c7b4a9a`
- canonical native smoke: `818ed30e38288859bd12796cb205ccebf2f76ed08252ab2ce1a9919f547026e1`
- clean bootstrap: `e596ff8794339b610ea2709dbc69f5ad1010338b44398824e4a2921701be108f`
- clean focused Make target: `2ef5cdc2e10d97ff791b9d4592002480d89eb440862075625dc7b0f9ff469f02`
- missing-assembler terminal: `b9bcef92d1238f1036012f803f3b5613b00fa0b7859d0807e4d34fc0e9be523b`
- corrected scalar prerequisites: `b85e586619eeb865c388ac058330dbd47b1c4bd869a9afd437199c72cf3abe51`
- corrected clean scalar suite: `d393d04c7937116e8ef621f0bde98c814cb5542d11c6f76c0b4ccaea475920bf`

The bounded repair is qualified. Full PR869 product acceptance, cross-platform
qualification, fixed-point gates, PR522, and release publication remain open.
