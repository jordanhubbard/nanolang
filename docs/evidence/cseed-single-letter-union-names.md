# My declared single-letter union names

I distinguish declared one-letter unions from free type variables when selecting
native C names (MAC `task_c117b20ef5d64339b79cf22e86b42ef7`). The earlier ordinary
imported `union T` control passed checking but generated `typedef struct void*`.
I retain that observation in [my imported-union evidence](cseed-imported-union-identity.md).

At entry to `transpile_to_c`, I snapshot the current environment's exact declared
one-letter union names into a thread-local 26-bit value. My private emission
body uses that value when choosing names, and my public wrapper restores the
previous snapshot after every inner return. I retain no environment or string
pointer. A declared union gets its ordinary `nl_` name; an unbound uppercase
variable retains its existing representation. Existing record binding and
nominal/type equality checks are unchanged.

At production checkpoint `95ce2e209f8e35dff2919b2c1976d72c9b295c2c`:

- Eight normal C-seed methods pass with GCC in 9.391 seconds and strict Clang in
  9.031 seconds. I execute local union callbacks/results, imported union T and
  the retained formerly failing positive control. I test declared T alongside
  `Box<T>` payload formals, single-letter record S and unbound generic function T.
  Wrong nominal identity and wrong payload types preserve the previous output.
- My same-process C harness emits a declared T, then a fresh environment with
  an unbound T. It checks native names and verifies that no naming context
  leaks. An enclosing U snapshot survives an inner success and an early refusal.
- Twenty adjacent imported-union, callback-signature and module-metadata methods
  pass in 19.696 seconds.

My fresh `make -j4 bootstrap test-parser test-typechecker test-transpiler test-native-nominal-context`
passes. The same-process harness passes with the final source fixtures and
normal shadow declarations.

My fresh Stage1 matrix passes six declared-union/record and mismatch methods,
but the seventh unbound generic-function control fails checking with E0010
(expected T, found int/bool). The full seven-method run took 15.262 seconds and
is retained as a failure, not a passing parity gate. I record this separate
selfhost limitation in `task_0198b105373a4cc2b3ebbbb0e9336af8`; no selfhost source
changed in this repair. The exact failure remains selectable with
`NANO_SINGLE_LETTER_COMPILERS=nanoc_stage1 python3 -m unittest -v tests.test_cseed_single_letter_nominals.SingleLetterNominals.test_unbound_generic_function`.

My fresh Stage2 passes the same six declared-union/record and mismatch methods
in 15.151 seconds. I select these methods explicitly; I do not rerun the known
unbound-function-generic refusal or imply that it passes in Stage2.

I also record a static adjacent limitation as
`task_fc3046bc294543ca837fcb4ca21298c3`: enum declaration emission still calls
the same name helper, while this bounded snapshot covers unions. A single-letter
enum can still select the free-variable spelling. I did not execute an enum
case or include enum support in this acceptance.

I retain logs under `/tmp/nanolang-single-letter-union-*.log`. I run ordinary
post-repair acceptance; I do not replay historical product aborts or attribute
those incidents to this naming repair. Product task dd74 remains separate.
