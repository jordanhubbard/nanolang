# Native void/scalar joins

I retain exact tags when an explicit void path joins one concrete integer,
boolean, byte, or float path. I track producer provenance separately from
boxed storage: a generic VALUE carrier can contain a heap object and does not
establish this scalar contract. I conservatively combine all local writes;
parameter carriers begin unknown.

I discover destination representations before final shape classification.
This includes a loop whose first edge is concrete and whose backedge is void.
I give the destination its own boxed shape, convert only incoming edges, and
retain the existing parallel-copy ordering. Integer zero, boolean false,
byte zero and float negative zero remain distinct from void. I retain the
existing string join implementation without extending its backward-edge
contract. Heap/reference widening and unrelated concrete scalar unions remain
refused.

The initial source checkpoint is `b957c6ca`, based on merged U8 carrier PR #574.
I restacked it as `e2bd826b` onto main `925b2127`, preserving PR #575
U8 formatting and PR #576 nested references. Only additive Makefile targets
conflicted; the native source merged without conflict. I then preserved the
PR #577 allocation target on main `045fb5a2`; native production source was
unchanged. Eight join methods passed again in 1.242s and all 91 adjacent
CAST_STRING allocation/cleanup checks passed. My
eight focused methods verify and execute the same modules in NanoVM and native
C, covering both branch orders, aliases, explicit void locals, nested and
parallel joins, carried loops, signed zero, and preserved publication on
unsupported joins. Three-way void/int/bool controls cover both predecessor orders for literals, comparison results, local copies and direct-call results.

Measured checks:

- Initial seven focused methods: GCC 1.249s; generated-program ASan/UBSan/LSan 2.755s;
  freshly instrumented translator ASan/UBSan/LSan 1.516s; Clang 1.438s.
- Eight methods after the independent review controls: 1.203s.
- After restacking: 16 join/truthiness/U8 methods passed in 14.259s; all
  eight join methods passed generated-C ASan/UBSan/LSan in 2.974s and Clang
  in 1.561s.
- The fresh private translator sanitizer gate passed 2,414 native checks and
  1,092 shape checks, then verified ASan/UBSan instrumentation in both objects.
  Its broad existing harness disables leak reporting; the focused translator
  and generated-program checks above explicitly enabled it.
- Six existing scalar truthiness methods across VM/C/LLVM/Wasm: 10.995s.
- Full native compiler checks: 2,414 passed, zero failed; shape checks: 1,092
  passed, zero failed. Opcode coverage and sanitizer-driver checks passed.

Commands include `make test-native-scalar-joins test-nvm2c`,
`python3 -m unittest -v tests.test_scalar_truthiness`, and the focused suite
with `NANO_JOIN_CFLAGS='-g -fsanitize=address,undefined
-fno-sanitize-recover=all -fno-omit-frame-pointer' ASAN_OPTIONS=detect_leaks=1`.
`NANO_JOIN_NVM2C` selects the separately instrumented translator. I used the
private build made by `make test-nvm2c-sanitizers` for that focused run.

The initial Clang invocation stopped at its GCC installation-selection warning
under `-Werror`, before generated C compilation. I selected the installed
GCC13 toolchain explicitly with
`NANO_JOIN_CFLAGS=--gcc-install-dir=/usr/lib/gcc/aarch64-linux-gnu/13`; I did not
suppress the warning. Both logs remain under `/tmp/nanolang-void-scalar-clang*`.
Other measured logs use `/tmp/nanolang-void-scalar-*`.

This is bounded native join support for MAC
`task_ca11c365f9e742d090f09ab59c6de45c`. I do not claim general tagged unions,
reference joins, or completion of the full compiler product acceptance.
