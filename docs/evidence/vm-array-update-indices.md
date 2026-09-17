# My array-update index contract

I require an integer index and check its full signed 64-bit range against the
array length before narrowing it. Wrong tags no longer become index zero;
negative, too-large and wrapping indices no longer silently discard writes.
I report type or bounds errors and release every popped heap value on rejection.
A valid update still transfers its payload and returns the same array handle.

Ten persistent-VM cases check bool/string/void/float indices, negative and upper
bounds, 2^32 and INT64_MAX, an invalid receiver, successful alias mutation,
unchanged data after failure, stack/frame recovery, exact reference counts and
heap cleanup after cycle collection. My full VM target passes 272,485 checks
plus callback, heap and stack allocation-failure tests. The same target passes
with fresh ASan/UBSan objects and default sanitizer options (including leak
checking); no diagnostic is suppressed.

```sh
make -j4 test-nanovm
make -j4 test-nanovm OBJ_DIR=/tmp/nanolang-vm-index-asan-ci-obj \
  CFLAGS='-Wall -Wextra -Werror -std=c99 -g -fno-omit-frame-pointer -fsanitize=address,undefined -fPIC -Isrc -D_GNU_SOURCE' \
  LDFLAGS='-fsanitize=address,undefined -lm -lcrypto -lffi -rdynamic'
```

I also assemble the nine native boxed-index fixtures and run both NanoVM and
strictly compiled generated C. Both backends accept all four valid mutations
and reject all five invalid cases. The native tests remain in
`test_boxed_array_indices`; my VM reference-lifetime tests remain in
`test_array_update_index_contract`.

My additional optimized sanitizer attempt (`-O1`) stopped on GCC's existing
null-format diagnostic in `nanocore_export.c:sbuf_appendf`. I preserved that
failure and used the repository CI sanitizer flags above in a separate fresh
object directory. I did not suppress the warning or change that source.

The matching raw array-read index behavior remains a separate recorded repair,
`task_e4e7048e334e4b9d8eed2086f792bbcf`. This patch changes `ARR_SET` only.
I track this completed update boundary as
`task_f64074441cf64f47b5f40ccefc78233c`.

Local evidence: `/tmp/nanolang-vm-index-gate.log`,
`/tmp/nanolang-vm-index-asan-ci.log`, `/tmp/nanolang-vm-index-parity.log` and
`/tmp/nanolang-vm-index-sanitized.log` (the optimized attempt).
