\ run_tests.fs — 280 T{ cases for the NanoISA Forth session.
\
\ make test-forth-examples loads Jackson tester.fr, session_prelude.fs
\ (DECIMAL and TEST-SUMMARY), then each test_*.fs through C file-source
\ REFILL. That is the gate. INCLUDE here is for a File Access Forth that
\ already has tester.fr.

INCLUDE examples/language/forth/test_arithmetic.fs
INCLUDE examples/language/forth/test_stack.fs
INCLUDE examples/language/forth/test_compare.fs
INCLUDE examples/language/forth/test_bitwise.fs
INCLUDE examples/language/forth/test_memory.fs
INCLUDE examples/language/forth/test_rstack.fs
INCLUDE examples/language/forth/test_control.fs
INCLUDE examples/language/forth/test_words.fs
INCLUDE examples/language/forth/test_base.fs

TEST-SUMMARY
