\ run_tests.fs — Master test runner for the NanoLang Forth interpreter
\ Run with: FORTH_FILE=examples/language/forth/run_tests.fs bin/nl_forth_interpreter_vm
\ Exit code = number of test failures (0 = all pass).

include examples/language/forth/test_arithmetic.fs
include examples/language/forth/test_stack.fs
include examples/language/forth/test_compare.fs
include examples/language/forth/test_bitwise.fs
include examples/language/forth/test_memory.fs
include examples/language/forth/test_rstack.fs
include examples/language/forth/test_control.fs
include examples/language/forth/test_words.fs
include examples/language/forth/test_base.fs

\ Grand total across all files
test-summary
