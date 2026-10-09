# I preserve qualified imported constants in native compilation

I repair the C-seed interpreter and native emitter on parent 30e3a8e77.
My Linux Strict examples failure is [job 113893374626](https://github.com/jordanhubbard/nanolang/actions/runs/37952140447/job/113893374626).

I retain the old two-module shadow failure in `baseline.log`. The permanent
`tests/test_native_imported_constants.py` suite passes all three methods through
actual shadow execution, C generation, linking and executable assertions:
module-owned int/string/bool/float/INT64_MIN literals, local record receivers,
and ordered arguments evaluated once. `tests.log` records this result.

I compile all four affected SDL_image examples successfully with
`CC=/opt/homebrew/opt/llvm/bin/clang`; `sdl-native.log` retains each command and
exit status. I do not run their graphical main functions.

I resolve only immutable literal declarations here. General mutable/nonliteral
legacy native globals, Linux rerun and complete release qualification remain
outside this result. I track release acceptance in #988, #982 and #976.

## I retain the callee's source context

My transitive probe after 7290d53e1 exposed missing source context in isolated
module emission (`transitive-baseline.log`). I now set and restore the original
source file in module transpilation and both interpreter function-call paths.
The fourth permanent test deliberately gives root and bridge different meanings
for the same alias and exercises direct calls and a selectively imported function
value. All four methods pass (`context-tests.log`). All four SDL examples compile
again (`context-sdl-native.log`).

`make test-env-scoping test-transpiler test-eval` passes (`direct-neighbors.log`):
52 environment assertions, ten lexical methods, native builder checks, two
assertion-literal methods, and the full evaluator suite. These targets now depend
on their actual C providers instead of the self-hosted bootstrap. I retain their
recipes unchanged and keep separate bootstrap qualification mandatory.

The optional qualified function-value probe refuses at typechecking
(`qualified-function-value-refusal.log`). I record it separately in #982; I do
not claim that literal field access establishes qualified function-value parity.
