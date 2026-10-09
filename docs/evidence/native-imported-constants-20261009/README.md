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
