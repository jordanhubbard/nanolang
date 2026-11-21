# My retained Linux raw Stage 2 timeout

I retain authoritative bootstrap run `run-s7hqibs_` at `74644c29f` under
[#1161](https://github.com/jordanhubbard/nanolang/issues/1161). My original
`make bootstrap2` executes the actual `bin/nanoc_stage1.nvm` in NanoVM, compiling
`src_nano/nanoc_v06.nano --emit-nvm`. It times out after 1,800.135 seconds at the
unchanged 1,800-second generation deadline. I preserve the separately configured
30-second complete compiler-shadow deadline and the guarded host closure.

My last successful phase recorder event is `typecheck-start` at
`2026-11-21T19:28:39.727001+00:00`. I have no subsequent typecheck completion
marker. A phase marker does not identify a function-level runtime cause.
The generation publishes no raw or native Stage 2 product: neither its work
output nor either final Stage 2 path exists. I have not compared raw Stage 1
and Stage 2 bytes.

My preceding Stage 1 passes raw generation and verification, C translation,
native linking, native hello compilation and actual hello execution. Its raw
generation takes 1,716.641 seconds. The native Stage 1 compiler also passes
separately recorded component tests; those results cannot substitute for the
failed NanoVM generation or establish its runtime cause.

My manifest preserves 35 logical inputs and 117,592,538 raw bytes, compressed
losslessly to 27,380,989 unique bytes. I retain raw/compressed SHA-256 identities,
the complete original step and source/tool/host manifest, both phase clocks,
all original step logs, actual raw seed/Stage 1 and native Stage 1 products,
actual selected VM/translator/runtime/helper tools, exact immutable host-library
bytes, generated Stage 1 C and the native compiler guard. Source files remain
identified by the pinned Git revision and the original per-file receipt hashes.

I also preserve the refused Linux perf control. It obtains no target samples;
I change no system settings. My subsequent instrumented native compiler profile
is a separate diagnostic. It cannot qualify the timed-out NanoVM generation.
Darwin's separate result remains uncollected while Puck refuses SSH with a local
account lock. The raw equality, paired bootstrap and complete 5.1 release gates
remain open.
