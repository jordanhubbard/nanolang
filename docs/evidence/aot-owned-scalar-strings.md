# I execute the native compiler past scalar-string arena exhaustion

My native compiler reached `cg_build` while emitting its C runtime and aborted
in `nstr_concat`. All concatenations, substrings and integer-formatting results
shared a fixed 65,536-byte arena. This limited accumulated allocations, not just
the size of any one string.

I replace that arena with individually allocated, execution-owned strings.
Each allocation holds a linked ownership header and stable bytes. I check size
addition, header and terminator space before allocation. Concatenation checks
operand-length addition; substring clamps using unsigned lengths without first
narrowing `strlen` to signed integer; formatting checks `snprintf` bounds.
Generated `main` releases these allocations after record, map and array cleanup.

Earlier strings remain valid across later allocations, including when they
escape through locals, arrays or records. I do not reclaim unreachable strings
during execution. Repeated growing concatenation can still consume quadratic
total memory, and other host string allocations have separate lifetimes. This
fix removes an arbitrary capacity limit; it is not a complete GC implementation.

## I checked the behavior

My new strict-C11 regression checks a 200,000-byte concatenation and substring,
negative/out-of-range substring bounds, 70,000 formatting allocations, both
signed integer extremes, escaped values surviving later allocations, zero owned
allocations after cleanup, repeated cleanup and generated-entry cleanup.
Six negative runs inject allocation failure and size overflow, including
concatenation overflow through a controlled length stub. They must trap.

`make test-one-ir-compiler` passes all 21 methods. In the end-to-end method I
freshly emit the full compiler's bytecode, translate it to C without an embedded
VM, compile under `-std=c11 -Wall -Wextra -Werror -O0` with the native artifact
host runtime, run `--help`, compile `examples/language/nl_hello.nano`, and run the
result with the exact expected greeting. This is the first green checkpoint
for that previously failing chain.

`make test-nvm2c` and `make test-nvm2c-sanitizers` each pass 1,670 AOT and 1,073
shape checks. The sanitizer gate rebuilds instrumented translator and shape
objects; leak detection is disabled. My focused allocation counters separately
check owned-string cleanup. `git diff --check` passes.

That result does not establish canonical bootstrap identity, general compiler
equivalence, bounded recursive frames, callback ABI acceptance, all-platform
release gates or deployed Horde SSO. Those remain separate work. I also have
newer main commits to reconcile before release.

MAC `task_b0c4ad8c9a824e64ab8fd3fa6881146e` tracks the scalar-string fix, and
`task_419c47bdc8fc42e4b52eb6af1a0e9a71` tracks the broader compiler work. Worker
claims still fail with `agent_status_unavailable`; evidence does not force
ledger closure.
