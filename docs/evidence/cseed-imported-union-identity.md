# My imported C-seed union declarations

I resolve declared function parameter and result kinds before either main or
module registration copies their metadata (MAC
`task_a2f464df8ba84a4ab4fc52c509e96904`). My ordinary registration path already
recognized declared unions; my module path retained the parser's record
placeholder and rejected valid module-local matches.

My four-line production change uses the existing nominal declaration helper
in `src/nominal_types.c`. I preserve names, annotation trees, callback
signatures and ownership rules. I do not relax type equality.

At production commit `6f965303`, my fresh normal acceptance is:

- `make -j4 bootstrap test-parser test-typechecker`: bootstrap and both unit
  suites pass.
- `python3 -m unittest -v tests.test_cseed_imported_unions`: the final eight
  methods pass in 8.293 seconds. I execute qualified and `from` imports,
  module callbacks, empty variants and returned unions. Wrong nominal and
  payload types, and a failed module shadow, preserve the previous output.
- Strict Clang passes the original seven methods in 6.178 seconds and the
  added unbound generic-function control in 1.422 seconds.
- The existing C-seed qualified constructor control plus module-signature,
  parameter-nominal and union-signature suites pass all sixteen methods in
  17.531 seconds.

- `NANO_CALLBACK_COMPILERS=nanoc_c python3 -m unittest -v tests.test_generic_function_values tests.test_resource_callback_boundary`:
  all 31 methods pass in 16.925 seconds, including the built NanoVirt/VM control.

My C function parser expects `(` immediately after a function name and has no
explicit function-formals list. Unbound single-uppercase names remain implicit
function variables; my `identity(T) -> T` int/bool control passes. Existing
ordinary function registration already gives exact declared unions precedence.
Explicit generic union payload formals keep their existing exclusion list.

I retained a separate failed positive control: an imported union named `T`
passes checking but native emission writes `typedef struct void*`. My native
name helper still treats every single-uppercase name as a free variable.
This is open task `task_c117b20ef5d64339b79cf22e86b42ef7`; I preserve the exact
control in `tests/acceptance_cseed_single_letter_union.py`, runnable explicitly
with `python3 -m unittest -v tests.acceptance_cseed_single_letter_union`. I do
not include that failure in my bounded success count or claim complete nominal
union naming support.

My retained logs are `/tmp/nanolang-cseed-imported-union-*.log`. The original
formal-control run passed the unbound generic case and failed the declared
single-letter native case; `formals.log` retains the compiler diagnostics and
`obj/nano_modules/.nano-module-8zaLYy/source.c` retains its generated C. These
are ordinary new compiler inputs, not historical abort replays. Product task
`dd74` remains separate.
