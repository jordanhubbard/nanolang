# I preserve generic array parameters through native emission

I previously assigned unresolved locals the integer representation at the end
of flat classification. That erased the distinction between an unconstrained
scalar and an array whose element storage was still unknown. My full compiler
stopped at `gen_call` (440), where `ARR_LEN` received that integer representation.

I now defer the integer fallback until after aggregate shape resolution. An
unknown local with an array shape uses my existing tagged array handle. Known
native array representations stay unchanged, including empty array constructors.
I do not guess an element type, alter the shape graph to optional, or claim new
support for generic array return values.

My new test exercises integer, boolean and string arrays, both function orders,
ordinary and tail forwarding, local copies, element tags and shared length
changes. Twelve native executions pass. Thirty-six malformed descriptor runs
trap: wrong value tag, null handle and unsupported storage kind.

I also reference every retained translated function from generated `main` using
standard C `(void)function` expressions. These do not execute the functions.
Two order variants retain an uncalled helper containing a failing assertion;
both compile under `-std=c11 -Wall -Wextra -Werror` and exit successfully.
I neither disable warnings nor remove uncalled functions from translation.

## I checked these gates

- `make -j1 test-nvm2c`: 1,670 AOT and 1,073 shape checks pass.
- `make test-nvm2c-sanitizers`: fresh ASan/UBSan objects pass the same checks.
  This gate disables leak detection; it does not establish bounded memory use.
- `make -j1 test-one-ir-compiler`: 15 of 16 test methods pass. Full compiler
  emission, strict native C compilation and `--help` succeed. Compiling the hello
  program aborts, so the end-to-end gate remains failed.
- `git diff --check`: passes.

I reproduced the remaining failure with a freshly emitted compiler and inspected
the abort in LLDB. `nhost_artifact_3`, called through `nl_canonical`, aborts at
`dlopen` of the standard-library artifact. `dlerror()` reports an unresolved
`_dyn_array_get_string` symbol. This is a runtime linkage boundary, not evidence
that compiler semantics or release acceptance are complete.

I track the parameter fix in MAC `task_9c768255d97544cc9e850ef547b90fc3`, retained
function warnings in `task_cd4036cffef64fd8a046f88153564b69`, and the runtime
failure in `task_fca5000374ef49988da33e5617e703b8`. Claims still fail with
`agent_status_unavailable`; I do not bypass the ledger's ownership checks.
