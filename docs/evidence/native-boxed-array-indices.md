# My checked native array indices

I permit an inferred boxed scalar index to reach my existing generated
`nvalue_require_int` check. I still reject statically incompatible kinds;
boxed bool/string/void values abort at the checked use. Integer bounds checks
remain before the write. I do not infer that an arbitrary boxed value is int.

My minimal valid fixture first calls a helper with an absent index on a path
that performs no update, then calls the same helper with integer zero. NanoVM
executes it, but my old classifier refused its merged optional index. The new
classifier emits the existing guarded extraction.

Nine native regression cases cover int, bool, string and record arrays, alias
visibility, bool/string/void indices and both bounds failures. `make test-nvm2c`
passes 1,800 structured-C checks and 1,076 shape constraints. Four valid cases
also execute in NanoVM. I found that NanoVM accepts all five invalid cases:
its array-update path currently substitutes zero for noninteger indices and
silently ignores invalid ranges. I keep the strict native checks and track this
separate repair as `task_f64074441cf64f47b5f40ccefc78233c`.

The unchanged full compiler gate advances beyond `parser_mark_owned`, then
refuses conflicting boxed/string-array facts from `generate_expression`
(function 483, offset 249) to parameter 2 of `mb_resolve` (function 648).
That next prerequisite remains on my roadmap. The full compiler gate and
`task_600074c773904b119b39bdafd85c07a5` remain open.

Local evidence: `/tmp/nanolang-boxed-index-gate.log`,
`/tmp/nanolang-boxed-index-vm-parity.log`,
`/tmp/nanolang-boxed-index-fullcompiler.log`, and retained compiler bytecode
and assembly under `/tmp/nanolang-boxed-index/`.
