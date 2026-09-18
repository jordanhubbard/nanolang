# My selected constructor call context

I normalize a brace expression to `ASTUnionConstruct` only when its spelling
matches a variant of a uniquely declared union. This repairs parenthesized
`(Choice.Some { n: 7 })` arguments that previously became an ordinary
`ASTStructLiteral` named `Choice.Some`. My existing postfix constructor path
already produces the union node. Unknown variants, duplicate union names and
unresolved qualified names keep the existing fallback; I do not guess from a
dot or coerce a selected payload variable into its enclosing union.

My call checker validates actual union-constructor nodes against the declared
union identity. I require the selected variant, every payload field exactly
once, and matching concrete field types after generic substitution. Unknown
payload types and incomplete generic contexts are refused. An uninstantiated
constructor can receive context; explicitly supplied generic arguments must
match the complete canonical identity. A phantom-parameter AST shadow checks
that equal integer payloads do not make `Phantom<string>` acceptable as
`Phantom<int>`. This is an AST-level defensive control, not a claim about
support for every explicit generic constructor spelling.

I keep equality and ownership checking unchanged. I do not use the broad
`apply_return_type_hint` helper for arguments. Direct, computed/function-value
and supported qualified calls use the constructor check. Native emission
receives parameter context from the declaration or function-value signature,
so the existing tagged-union construction emits `Box<int>` rather than an
uninstantiated `Box`. Ordinary record construction keeps its existing path.

## Measured acceptance

My integrated production source is `89d90f255f8d85588da518c41b93a46260e021c3`,
rebased onto main `3a33b182`. The final fixture and Makefile cleanup are
`912d1e79`; compiler source is unchanged between those checkpoints. A fresh
`make -j4 bootstrap` passes all stages and installed-compiler smoke checks
(`/tmp/nanolang-constructor-context-bootstrap-integrated.log`).

Thirteen methods in `tests.test_constructor_call_context` pass in 43.617s.
Each runs both Stage1 and Stage2. They cover direct and empty constructors,
parenthesized let/return values, function-variable and computed calls,
concrete generics, qualified calls, wrong unions/variants, missing/extra/
duplicate fields, wrong payload types, selected-variable refusal and failed
shadow output preservation. Positive products execute their assertions;
negative cases must preserve an existing output and fail before C compilation.
The log is `/tmp/nanolang-constructor-call-integrated.log`.

Ten explicitly selected C-seed controls pass in 5.745s
(`/tmp/nanolang-constructor-call-cseed-integrated.log`): direct/empty,
parenthesized let/return, concrete generic, generic function-variable/computed,
wrong union, unknown variant, missing/extra/duplicate fields, payload type,
selected variable and failed-shadow preservation. I do not count these as
complete cross-frontend parity.

Thirty adjacent generic-function/resource-boundary methods pass across both
selfhost stages in the 79.242s combined run
(`/tmp/nanolang-constructor-call-adjacent.log`). Its remaining VM method could
not start because I had omitted `nano_virt` from the tool build. After
`make -j4 nano_virt nano_vm`, that unchanged method passes separately; I retain
`/tmp/nanolang-constructor-adjacent-tools.log` and
`/tmp/nanolang-constructor-adjacent-vm.log`. I do not describe the initial
combined runner as wholly successful. Existing generic resource callback
refusals, fixed resource callbacks and ordinary/phantom generic controls remain
checked without changing ownership code.

The repeatable selfhost target is `make test-constructor-call-context`.
`NANO_CONSTRUCTOR_COMPILERS` names the exact compiler routes for individual
controls; its default is `nanoc_stage1,nanoc_stage2`.

## Retained failures and limits

My initial new parser shadow used reserved word `use` as a placeholder call
name. Parsing that fixture failed before its constructor access; the retained
empty-list error is in `/tmp/nanolang-constructor-context-build.log`. The
corrected ordinary identifier `consume` passes. An isolated root-shadow-only
parser probe was diagnostic work, not bootstrap acceptance.

The first Stage1 run passed ten of eleven methods and exposed missing native
generic context through function values (`nl_Box` instead of `nl_Box_int`).
I retained `/tmp/nanolang-constructor-call-stage1.log`, repaired that handoff,
and reran the affected gates. I then added duplicate-name and explicit-generic
identity controls and rebuilt after each production change.

The first complete C-seed attempt passed nine of eleven methods and retained
two separate refusals in `/tmp/nanolang-constructor-call-cseed.log`:

- `task_f00d97409c26413781ff85f06993e66d`: a nongeneric `fn(Choice)->int`
  local/computed call is rejected as a signature mismatch while its concrete
  generic counterpart passes.
- `task_a2f464df8ba84a4ab4fc52c509e96904`: an imported module's own function
  parameter `Choice` is rejected as non-union when matched. The selfhost
  routes accept the same source.

Both remain open with their original fixture methods and diagnostics. This
repair completes only `task_6961296c51014326bb3a532c33fab2e0`'s tested selfhost
normalization/emission scope. Forward or ambiguous declaration resolution,
general module-qualified constructor spelling, broader C signature identity
and release acceptance are not established by this gate. Existing resource
restrictions remain in force; I change no ownership traversal or permission.
