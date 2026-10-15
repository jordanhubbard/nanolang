# My pinned native host generations

Linux x64 job113754852492 at 18fd40099 fails Stage 1 with only a runner-local log path in hosted output. My Linux ARM64 reproduction first stops on omitted scratch-copy fixture files; I retain those setup failures and restore the tracked tests. The complete fixture copy reaches Stage 1 and reports a cached shared-link validation failure for nanoisa.

The native-code guard's marker identifies the exact refused input: nanoisa.o inside the seed's immutable .nano-gen-* cache directory. I previously allowed only declared .nano-build-* snapshot inputs under the seed cache root. Linux shared-link validation also relinks the already-published object. I record exact generation directories from the verified seed host closure and allow only their declared object filenames. Other generations in the same cache root, undeclared objects, generated C, unrelated cache roots and symlink escapes remain refused.

All five focused guard/bootstrap mutation methods pass. I retain source/tool/host mutation refusal and raw module comparison; I do not normalize import paths. A fresh complete Linux bootstrap is running and remains required. CI now retains per-step bootstrap logs and manifests on failure.
