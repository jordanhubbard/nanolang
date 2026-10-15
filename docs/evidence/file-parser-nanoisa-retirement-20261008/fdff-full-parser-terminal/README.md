# Full parser gate after optional-record pop

I ran `make -j2 test-file-service-parser CC=/opt/homebrew/opt/llvm/bin/clang`
at `fdff092ed3ab04f40bcd9a0b52f742ae35b06442`. The complete command exited 2
after 864.448 seconds. My source pin and pre-existing untracked user-file hash
remained unchanged; `manifest.json` retains both checks.

My fresh bootstrap completed all 17 recorded steps with exit zero and raw
Stage1/Stage2 bytecode equality. I retain its source and artifact hashes,
commands and step logs under `bootstrap/`.

My C ownership/refusal method passed. My paired compiler method failed the
unchanged exact shadow-selection assertion: C reported all 77 expected names,
while Stage1 reported zero. Stage1 compilation itself returned zero without a
timeout; no `I am testing shadow ...` line appeared. This does not establish
that the shadows were skipped. I must restore execution-time observability
and rerun the complete corpus; Stage1 execution, Stage2 and the schema corpus
were not reached by this method. Issue #978 remains open.
