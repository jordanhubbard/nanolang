# My public C binary64 evidence

I tested production through `e894f53a` with frozen harness `d315d6f2` in
`nanolang-public-c-scalar-policy`. My [manifest](public-c-binary64.json) retains
21 source, harness, tool and imported-library hashes, checked unchanged after
all final gates, plus retained log hashes. My [contract](../PUBLIC_C_BINARY64_CONTRACT.md)
precedes implementation under local child54c918; historical parent070db worker
failure is separate and unchanged.

I preserve every binary64 literal and bit intrinsic through integer representation
and memcpy. My public C scalar arithmetic uses the shared canonical-NaN/total-zero
division policy with ordered, distinct automatic operand temporaries. Scoped
function results and checked/declaration-owned field types replace integer guesses;
unknown scalar results remain refused. I stage publication before replacing prior
output, retain first errors, and recover on later valid same-process calls.

I initialize scalar globals once in declaration order and expose a hosted int main
wrapper around my private int64 language entry. An initializer calling main observes
its full high-bit result without recursively restarting initialization. Nested calls,
branches, loops, user f64_* functions and temporary-like names remain distinct.

| Frozen gate | Result |
| --- | --- |
| Public C focused GCC | 8 methods, 4.003 seconds |
| Public C focused Clang | 8 methods, 4.659 seconds |
| Existing public C backend programs | 7 pass, no skip or expected failure |
| Shared helper GCC | 2 methods, 1.179 seconds |
| Shared helper Clang | 2 methods, 1.437 seconds |

My focused source programs compile and execute C99/C11 at O0/O2 with pedantic errors,
selected diagnostic errors, UBSan and contraction enabled. Integer observers cover
NaNs, signed zero, subnormals, rounding and separate-operation non-contraction.
One arithmetic source also passes interpreter, legacy C, verified VM and native C
with sanitizers. My API fragment uses strict GCC/Clang warnings, ASan/UBSan,
allocation failures, output retention and same-process recovery. Direct/emitted
helper gates include O0/O2/O3, contraction and LTO plus target/fast-math refusals.
These are measured host results, not universal floating-environment proof.

My fresh bootstrap passed at `1245ca51`, including both stages, hello and installed
compiler checks. Later changes are C-backend/API test corrections; I rebuilt affected
C tools and retained Stage1/Stage2 identities. I do not call this a final-head bootstrap.
The generated arithmetic provider remains identical to its shared C source.

I retain earlier outcomes: initial API fragment link failure lacked its unreachable
passive fixture stub; the adjacent suite first passed 5/7 with two checked unresolved
field refusals; a field fixture constant was initially below the Python runner; the
annotation-only attempt then checked-refused the new FLOAT field source. I corrected
these independently, kept each terminal log and verified each freeze before edits.
No rejected generated program ran, and I did not replay historical compiler failures.

My bounded scalar profile does not establish all-target C99 portability. Existing
GNU block/string expression behavior and option/closure scope remain under6ade.
Canonical FUNCREF/d099, remaining scalar-policy5009 obligations, arrays and full
release stay open. Cross-host public C nonfinite formatting is a separate continuation
that must consume the shared formatting contract rather than infer libc behavior.
