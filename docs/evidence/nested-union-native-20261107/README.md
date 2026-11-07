# My nested union native storage check

I reproduce #1061 with a union-valued payload, a nested match and a helper
that takes the inner union and returns the outer union. I execute populated,
empty and negative integer cases. Each function has a useful shadow.

My prior shape implementation from `bb60655cd51cbcd1f2737faba4c8905703e3bb7b`
refuses the same emitted module with `variant-scalar to int`. I retain that
negative control in `before-refusal.log`; its refused publication preserves
an existing output. My changed source identities are in `source-identities.json`.

I preserve an inferred boxed payload's finite scalar set after a nested
projection resolves. I widen only a payload within OPTIONAL storage. Bare
scalar destinations, array elements and exact payload constraints still refuse
that conversion. Root unification retains an exact constraint even when the
inferred root has higher rank. I change no VM, verifier or runtime tag checks.

My local Darwin arm64 checks pass:

- `make test-nvm2c`: 2,443 structured-C assertions, 464 callable assertions,
  the shape gate and adjacent native effect, array, scalar-carrier and callback
  methods. One existing callback case requires a self-hosted route for its declared
  builtin-name contract and is skipped in its eleven-method group; I retain the complete output in `full-translator.log`.
- `make test-nvm2c-shapes test-nested-union-native`: 3,562 shape assertions and
  VM/native execution in both match-arm orders. Native C uses ASan and UBSan.
- `CC=/opt/homebrew/bin/gcc-16 python3 -m unittest -v tests.test_scalar_union_emission.NestedUnionNative`:
  the same native execution with GCC, ASan and UBSan.
- `cc -std=c11 -Wall -Wextra -Werror -fsanitize=address,undefined -fno-omit-frame-pointer -Isrc/nanoisa tests/nanoisa/test_nvm2c_shape.c src/nanoisa/nvm2c_shape.c -o /tmp/nanolang-nested-union/sanitized-shapes`,
  followed by that binary: all 3,562 shape assertions pass under ASan/UBSan.
- `python3 scripts/fast_gate.py --jobs 4`: 223 strict Clang and 223 strict GCC
  compilations pass.

The full translator run compiled the earlier 3,557-assertion version of my
new shape tests; my final focused and sanitized runs add five higher-rank
exact-binding assertions. Production source is identical across these runs.

I replace the old self-hosted native-refusal expectation with this shared
positive fixture. Fresh Stage 1/Stage 2 execution is still required in the
combined hosted compiler-contract gate. These C-seed/local translator checks
do not establish that qualification or complete 5.1 release acceptance.
