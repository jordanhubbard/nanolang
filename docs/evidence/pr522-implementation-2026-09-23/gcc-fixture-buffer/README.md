# GCC translator fixture buffer

I size the record-consumer assembly buffer from twice the reader buffer bound plus 128 bytes of fixed assembly text. GCC bounds each conditional string argument independently; the original 1,024-byte destination did not cover its computed maximum of 1,101 bytes. The new bound is 1,152 bytes. I retain every assertion, declaration order and local-copy case.

With GCC 13.3 on Ubuntu 24.04 ARM64, the complete test translation unit fails before correction and compiles afterward under `-Wall -Wextra -Werror -std=c99 -O3 -D_GNU_SOURCE`. The configured logs record the exact commands and exit codes. My first diagnostic commands omitted the Makefile's `_GNU_SOURCE` definition; I retain their logs, including the unrelated missing POSIX declarations, separately.

The original hosted Linux x64, ARM64 and coverage failures are retained in the adjacent `isolated-sanitizer-gates` evidence. Final hosted acceptance remains open under `task_a131be9bdb044185976c63716eb12600`.

My complete local `make test-nvm2c` exits zero: 2,558 translator assertions, 1,614 shape assertions, the opcode-coverage method and all three sanitizer-driver methods pass. This is ordinary local execution; the GCC before/after check compiles the complete translation unit without linking or running it.
