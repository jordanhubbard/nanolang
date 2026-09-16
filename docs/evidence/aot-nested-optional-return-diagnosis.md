# My nested optional return conflict

I reduced the full compiler's `env_get_type` tail-call conflict to four small
functions in `tests/nanoisa/fixtures/nested_optional_returns.nasm`. Each return
contains a record inside another record. One inner field contains a string;
the other contains an uninitialized global's void value. The caller checks
both results.

The fixture assembles and executes successfully in NanoVM. Native translation
fails in function 3 at offset 25 during `RET`, with string/optional shape kinds.
The full compiler reports optional/string kinds during `TAIL_CALL` in function
311 at offset 273. The order differs; the representation conflict is the same.

`shape_record_return` recognizes optional fields in its flat, immediate field
vector. If no immediate field is optional, it equates the entire recursive
source and destination shapes. Nested optional fields therefore reach strict
equality and fail. Node IDs now appear in the diagnostic alongside kind names;
I do not treat those IDs as stable across unrelated source changes.

I have not fixed recursive conversion. Simply making strict unification accept
string/optional pairs is unsafe: a string node can also be the optional's
payload. Promoting that shared node would confuse the payload with its wrapper.
The next implementation must keep conversion distinct from equality and retain
payload compatibility, including shared and cyclic graphs.

I added the minimized fixture to `make test-one-ir-compiler`. That acceptance
gate must execute it in both the VM and generated native code. It currently
fails, alongside the full compiler case; the empty-array source case passes.
This is an explicit remaining requirement, not a passing implementation test.

Normal and fresh ASan/UBSan suites pass 1,572 AOT and 1,001 shape checks.
The diagnostic tests verify named conflicting kinds, node IDs, stable error
storage after graph poisoning, and cleanup. Leak detection remains disabled.
