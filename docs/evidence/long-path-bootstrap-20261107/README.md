# My complete long-path Stage 1 bootstrap

I pass a fresh `make -j2 bootstrap1` at an actual 316-byte absolute checkout path on Darwin arm64 from code-only commit `fbdfc32e846a43ad7dce617243e2e4d5d0eca61b`. I use fresh objects, binaries and a module cache within that detached worktree. I copy no earlier build outputs or receipts. I omit only historical evidence from the sparse checkout.

My seed compilation, Stage 1 self-compilation, NVM verification, exact host-closure comparison, native translation and native linking pass. The resulting Stage 1 compiler compiles my hello example and executes it successfully. The complete Make invocation returns zero and creates its Stage 1 checkpoint; my qualification helper checks the manifest phase and exact worktree root.

I retain my source pin, actual commands, timings, output log, manifest and manifest checksum here. The complete worktree and phase logs remain under `~/.worktrees/nanolang/qualify-long-path-60pd9bks/`. My preceding failure remains under `~/.worktrees/nanolang/qualify-long-path-wbjccc27/` and `docs/evidence/long-path-host-closure-20261107/`; I do not reinterpret it as an infrastructure incident.

This passes my long-path acceptance for #1055 and the cache-generation repair for #1077 locally. I still require integration of these code branches and their combined required gates. This is Stage 1 evidence; it establishes neither Stage 2, full retained-execution admission nor a release.
