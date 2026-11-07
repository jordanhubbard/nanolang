# My compiler-option preservation repair

I reproduce two plan-parser errors under #1046. A real C compiler accepts `-DINPUT=src/utf8.c`, but my old parser treats it as an input filename. It also discards a definition such as `-DOBJECT=unused.o` as a linker input. My independent refusal fixture compiles without that macro, then requires a compiler error when the macro is requested. The old parser drops it and wrongly reports success; the repaired parser preserves the option and the compiler refusal.

I classify option tokens before filename suffixes. I preserve C/object/archive/shared-library-valued definitions and skip paired option arguments when reporting the actual translation unit, including a forced `.c` include. I retain the original refusal, compiler-accepted option control and old/new macro diagnostic evidence.

All eleven planner/process controls pass. My final two-worker default `make check-fast` passes all eight gate controls (one explicit Darwin/glibc skip), the fixture/prerequisite checks, 223 Clang compiles in 15.3 seconds, 223 GCC compiles in 31.1 seconds and the bootstrap-tool control. I compare the same two-mode planner before and after this parser repair: all 1,246 existing command vectors remain unchanged. The original Make inventory contains none of these affected option forms, so this repair does not explain the separate Linux profile-audit failures.

This repairs option preservation and reporting. The broader profile coverage/runtime, Linux audit diagnostics and full release gates remain open under #1046 and #976.
