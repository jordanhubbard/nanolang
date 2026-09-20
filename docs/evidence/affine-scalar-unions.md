# My affine scalar-union evidence

I qualify the bounded implementation on Darwin at commit
`271086b0989527b4fc7258691920114de434d07b` (tree
`50fc22d58a21418051505903f85e30ebc6fc6ed0`), based on canonical main
`f1606e2c84e67491e9652a5bf71944d235216d95`.

## My retained terminals

- The first retained-layout draft published `.types 1 0 1` with one record
  layout. Verification refused the missing union layout.
- The field-less layout draft reached affine analysis, which refused the union
  local because its descriptor had no exact retained identity.
- The first current-main three-compiler gate passed C-seed but Stage 1 and
  Stage 2 generated duplicate C labels for repeated guarded `Some` arms. I
  retain that terminal and do not count it as acceptance.
- The first direct canonical test incorrectly requested `--emit-nvm` from
  `nanoc_c`, whose documented CLI has no such option. I retain that harness
  boundary; canonical emission is qualified through Stage 1 and Stage 2, while
  C-seed remains in the native source matrix.

## My accepted gates

Two fresh current-main bootstraps completed through Stage 1, Stage 2,
installed-compiler hello and no-C-seed hello. The second includes the guarded
self-hosted C lowering. Both native stage comparisons recorded unequal binaries
and explicitly did not treat that as a fixed-point proof.

The complete source matrix passes 17 methods in 126.219 seconds under
`nanoc_c`, `nanoc_stage1`, and `nanoc_stage2`. Accepted artifacts execute;
resource-bearing and unresolved cases refuse while preserving their prior
output. The guarded method additionally passes `--emit-nvm` under both
self-hosted stages, NanoVM execution, strict `-Wall -Wextra -Werror` native
translation, and native execution.

The raw matrix log is retained at
`/private/tmp/nanolang-a18-current-generic.log`, SHA-256
`8c6bd0126ea693d88e1f6a131515439d4108508c04fb30addde0611c009231a3`.

My low-level affine gates pass:

- 314 normal and 343 allocation-state checks;
- 441 normal and 751 allocation-bytecode checks.

The raw log is retained at `/private/tmp/nanolang-a18-current-affine.log`,
SHA-256
`9b434142a2beb54cad359648488dd784893c3a0a0a9fc7db3981185bc8b97cc8`.

The adjacent owned-value lifecycle passes under Homebrew LLVM LeakSanitizer:
all ten cases and 1,847 checks. Its raw log is retained at
`/private/tmp/nanolang-a18-current-owned.log`, SHA-256
`7e257a04fa7010e5757c559d90b780eeffa49accf886de670f059076474b9de4`.

The selected compiler/runtime artifact map is retained at
`/private/tmp/nanolang-a18-current-tools.sha256`; the map SHA-256 is
`01b3db481cb7fbbe1f24c8f1722fa6816f8d3fe6aa3cc484a083ff2ae5ba5e50`.

## My boundary

This proves the scoped scalar-union, guarded-match, affine-analysis, NanoVM,
and native-AOT paths on this Darwin source. It does not prove the held product,
raw VM/native compiler fixed points, other aggregate profiles, Linux parity,
PR522, or release readiness.
