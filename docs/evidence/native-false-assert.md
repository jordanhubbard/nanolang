# My native terminal assertion evidence

Source checkpoint: `614a7369`, based on main `42f48aed`.
I retain only the immediate decoded same-block `PUSH_BOOL 0; ASSERT` fact.
Branch entries and skipped instructions clear it. Independent joins resume
normally. My HALT implementation is unchanged.

- `make -j8 test-nvm2c` passes **2,422 native checks** and **1,269 shape checks**,
  zero failures. Log: `/tmp/nanolang-native-false-assert-full.log`.
- Four focused methods pass under GCC ASan/UBSan in **0.341 seconds**.
  Log: `/tmp/nanolang-native-false-assert-focused.log`.
- The same methods pass under Clang ASan/UBSan in **0.749 seconds**.
  I explicitly select installed GCC13 for this Clang driver using
  `--gcc-install-dir=/usr/lib/gcc/aarch64-linux-gnu/13`; no warning suppression.
  Log: `/tmp/nanolang-native-false-assert-clang.log`.
- Positive controls cover float/string helper results, independent entry at
  ASSERT and a live successor after a terminal arm. Ten unchanged reachable
  HALT/unknown-condition refusal cases preserve existing output.
- All twelve source union/expression-match methods pass in **44.167 seconds**
  using this translator explicitly with the frozen source producer from PR653.
  Both raw hosts, C-seed, canonical Stage1/Stage2, selected shadows, VM and
  sanitized native output agree. Log:
  `/tmp/nanolang-scalar-match-values-native-companion.log`.

I retain the earlier source gate at `/tmp/nanolang-scalar-match-values-paired.log`:
VM execution passed, while native translation refused empty-stack HALT in the
float helper's impossible-tag fallback. The four-scalar source fixture remains
unchanged. This is not general HALT admission or new verifier authority.

MAC: `task_d86843ef53d343bd8881dd419b072532`.
