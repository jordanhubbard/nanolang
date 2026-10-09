# Selected resource-union execution

I now carry exact selected resource-union ownership through constructors,
local moves/stores, destructive unpack, consuming calls and returned unions or
records containing unions. Matching refines the tested operand; failed tests
exclude variants, and joins intersect those exclusions. I retain the existing
256-variant metadata limit and bound worklist convergence for these facts.
Native lowering uses the shared converged stack depths and selected unpack
counts. VM transfers clear source roots and reserve space before mutation.

My final command exits zero:

```
make -j2 test-owned-union-runtime test-affine-state test-affine-bytecode test-affine-scalar-union-runtime test-owned-runtime test-owned-transfers test-ownership-contracts CC=/opt/homebrew/opt/llvm/bin/clang
```

`runtime-final.log` records 18 paired VM/native cases and 17 ownership refusals,
2,380 selected-union runtime checks, and 96 VM heap allocation-failure checks.
The paired native harness retains strict compilation, ASan/UBSan with leak
detection, exact values, peak-live bounds and exhaustive allocation failure.
All existing tests also pass: 413/445 state, 606/1009 bytecode, 184/275 transfer,
32 VM allocation and 1,521 owned-runtime checks, scalar-union execution and
ownership transport. The first number in each pair is the ordinary test and
the second is its allocation test.

Coverage includes empty, one-owner and two-owner variants; nested unions;
record-wrapped union returns; exact consuming signatures; both branch choices;
a repeated construct/consume loop; duplicate, discard, copy, overwrite,
unknown-arm unpack, wrong payload/layout, live-owner join, incomplete match,
repeat consumption, borrowed projection and tag-extraction refusals. Each
ordinary VM case executes twenty times and returns heap counts to baseline.

I preserve earlier runs:

- `bytecode-first.log`: adjacent state, bytecode and transport gates pass.
- `runtime-first.log`: strict compilation refuses adjacent string literals in
  the new refusal-case array; explicit parentheses correct that test syntax.
- `runtime-second.log`: the initial twelve paired union cases pass alongside
  scalar-union and record-runtime controls.
- `runtime-expanded.log`: the new heap fault test incorrectly assumes every
  injected allocation failure must prevent success. `vm_union_new` calls
  `calloc(0, ...)` for Empty, and explicitly permits null for zero fields.
  I continue injection through every later position, checking exact results
  and zero leaks on successful execution as well as memory-error cleanup.
  I retain the original failure and never raise the attempt bound.

This establishes the raw standalone VM/C transfer contract. Both source
frontends, linked contracts, other translators, Linux and final 5.1 acceptance
remain required. The issue stays open.
