# Array consumer inference before storage fallback

I retain hosted scalar job `107522865961` from run `35965397438` at `0eb94026c`.
Its native bootstrap refuses a record versus tagged-value parameter at function
411, offset 35, parameter 1 of function 530. The completed VM Stage 1 module
reproduces the exact refusal locally: `nisa_statements_fall_through` projects an
array element into the record parameter of `nisa_stmt_terminates`.

The earlier declaration fallback boxed an array before its consumers' record
constraints could settle. I remove that early fallback. After the original
shape solver completes, I mark only still-unresolved array reads as optional
storage and solve their directed local conversions. Known record consumers
retain their existing exact constraints and representations.

I retain a rejected prototype that broadened record/tagged call merging; its
full compiler conversion failed with an integer/optional shape conflict. It is
not my final implementation. The first late-resolution gate then passes 2,518
assertions and fails two float cases. I extend the existing inferred optional
storage rule to float and expand its existing directed-conversion controls,
including mismatched payloads and refusal to widen explicitly constrained
storage. Both declaration orders of an uncalled array reader with a record
consumer are new regression cases.

The final ordinary translator gate passes 2,524 assertions. Both complete
compiler modules translate: one emitted by the native Stage 1 compiler and one
emitted by the VM Stage 1 compiler. I retain the original nine-method functional
pass before the final float correction; final qualification follows below.

The final complete functional-array gate passes all nine methods with generated
ASan/UBSan and leak detection. Both final translator gates pass 2,524 assertions;
the private sanitizer harness verifies ASan/UBSan object symbols and retains its
existing leak exclusion. The shape gate passes 1,565 assertions, including strict
GCC 13 execution; both modified translator/test sources also compile under
`-O3 -Wall -Wextra -Werror` in the isolated Linux environment.

Fresh ordinary Darwin bootstrap passes both native compiler stages, installed
compiler execution and the smoke check with the C seed removed. I delete only
the three generated bootstrap sentinels before `make -j2 bootstrap3`, selecting
Homebrew Clang and CI's existing 60-second shadow deadline. I do not extend the
bootstrap stage bounds.

In the isolated Linux checkout at 0eb94026c, GCC also compiles the complete
compiler C emitted by my corrected translator with `-std=c11 -O0 -Wall -Wextra
-Werror`; its help command succeeds. This is a targeted corrected-translator
check using the pinned VM Stage 1 module and runtime, not a native fixed point
or a claim that the Linux checkout already contains the translator correction.

The Linux standalone compiler also emits the unchanged hello source to bytecode;
that product verifies and executes successfully in NanoVM.

MAC rejects the direct completion transition; I retain its exact response.
The verified repository repair is not a claim that the ledger task is closed.
