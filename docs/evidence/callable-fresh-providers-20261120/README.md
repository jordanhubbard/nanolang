# My fresh-provider callable ownership regression

I resume #1140 from `3405117b9` in a new isolated worktree. My committed regression
is `93646bc0d`. I compile every COMMON/RUNTIME provider locally; I do not copy
cached objects or use a seed-built source compiler for this witness.

I run `make -j8 test-eval test-eval-callable-lifetimes` with the original GCC
flags. The complete existing interpreter suite passes. My new acceptance target
correctly exits nonzero: nine fixed 64-call routes retain 576 callable names after
actual environment teardown. Every scalar result, returned signature/name
independence check, caller-survival check and restored symbol count passes.
Each route also creates one independently owned callable: environment teardown
releases that exact observed allocation. The regression does not accept leaks
as a passing outcome and remains separate from the original suite until the
ownership correction is qualified.

| Route | Outstanding names after teardown |
| --- | ---: |
| direct | 0 |
| bound | 64 |
| declaration return | 64 |
| binding return | 64 |
| scalar recursion | 0 |
| recursive callable return | 256 |
| parameter invocation | 0 |
| parameter forwarding | 0 |
| computed returned callee | 128 |

I rebuild the complete provider graph again with Clang ASan/UBSan, separate
objects and a separate compiler-input archive:

```sh
make -j8 test-eval-callable-lifetimes \
  CC='clang --gcc-install-dir=/usr/lib/gcc/aarch64-linux-gnu/13' \
  OBJ_DIR=obj-callable-asan \
  COMPILER_INPUT_ARCHIVE=lib/libnano_compiler_inputs_callable_asan.a \
  CFLAGS='-Wall -Wextra -Werror -std=c99 -g -O1 -fPIC -Isrc -D_GNU_SOURCE -fsanitize=address,undefined -fno-omit-frame-pointer' \
  LDFLAGS='-lm -fsanitize=address,undefined'
```

The sanitizer run retains the same nine counts and 576-name total. LeakSanitizer
also refuses process exit with 64,820 bytes in 3,185 allocations. That total
includes fixture setup and shadows; I do not assign it entirely to the measured
names. I observe no invalid-access or UBSan report in this run. A failed leak
check is not sanitizer acceptance.

I preserve the initial fixture refusal (missing main), the first observer's
omitted forwarding of test-owned returned-name frees, and buffered final output.
The corrected observer forwards those real frees and uses unbuffered stdout so
LeakSanitizer cannot hide the final cohort. My first Clang build refuses its
ambient GCC-install selection warning under Werror. I explicitly select the
installed GCC13 development directory; I suppress no warning.

`providers.json.gz` binds the local objects, dependency files, archives and
native images with SHA-256 and records post-run tool identification. Compressed
logs retain commands and all first/corrected terminals. I do not claim complete
SDK custody, a measured source-factory gain, deadline causation, Darwin
acceptance, escaped-container correctness or a production ownership fix.

My next correction must account for lexical callable bindings, returned owned
temporaries and computed callees while preserving borrowed reads, nested
signatures and escaped containers. #1140 and the complete release remain open.
