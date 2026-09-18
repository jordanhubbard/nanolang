# My helper-local ownership evidence

I qualify `task_74170cc4b1784e68970017da2a3ec64a` as a runtime prerequisite.
My production checkpoint is `c7c9fd9ce07808da7b10021007cabb95bd06e07c`.
I distinguish caller roots from helper-local roots with origin and invocation
generation in native references. My VM already retains both facts. I preserve
borrowed-only helper parameters, two activations and exact affine exit checks.
My source producers still refuse helper-owned local declarations.

I run ordinary verified modules with different nested records at caller and
helper slot 5, local/path borrows, inherited/local reborrows and two helper
calls per entry. Shared and exclusive caller parameters retain their authority.
I qualify successful execution and false assertions through all four VM APIs,
with zero remaining frames/reference activations and baseline heap objects.

At that checkpoint:

- `make -j4 test-helper-local-owners test-owned-assertions test-multi-caller-references test-affine-bytecode`
  passes 1,309 helper lifecycle checks, 106 helper allocation checks, 959
  assertion lifecycle checks, 2,004 multi-caller checks, 93 atomic binding
  checks, 89 caller owner-allocation checks and 441/751 affine checks.
- `CC=clang-18 python3 -m unittest -v tests.test_helper_local_owners` passes
  the same four normal/assertion modules with strict generated C and
  ASan/UBSan/LSan. GCC also passes those native checks.
- Native allocation injection visits each allocation until normal completion;
  successful and assertion-failing runs leave no live native owner allocation.
  VM heap allocation injection likewise preserves its initial object count.
- `make -j4 test-caller-references test-verifier test-nvm2c` passes 1,548
  caller checks, 43 caller allocation checks, 55 parameter allocation checks,
  96 verifier checks, 1,365 shape checks and 2,422 native checks.
- Canonical v2 serialization preserves each module and regenerates identical C.
- Integration `a6840b7e` includes main `d1d3c9a6`. Only the additive roadmap
  conflict required resolution. The runtime production and new tests are
  byte-identical to `c7c9fd9c`; the integrated focused gate passes again.

I retain logs at `/tmp/nanolang-helper-local-gates.log`,
`/tmp/nanolang-helper-local-clang.log` and
`/tmp/nanolang-helper-local-adjacent.log`. I do not execute malformed modules
or historical failing artifacts. Broader source admission and normative
ownership parents remain open.
