# I retain C source import qualification

I restore the original source-only imported-shadow acceptance case by retaining
checked scalar function declarations and resolving calls in their module owner.
I emit and execute C with transitive aliases, same-spelled private helpers and
string results, without running deliberately failing shadows. Imported storage
remains outside this bounded C-source profile and preserves prior output.

My final ordinary build and complete imported-shadow suite pass (12 methods).
The adjacent callable, bool-text and expression-value suite passes 11 methods;
the binary64 suite passes eight, including standalone API service refusal and
prior-output preservation. These API/generated-C fixtures use Homebrew Clang
ASan/UBSan with leak detection, including C99/C11 and their defined optimization
variants. They do not establish hosted platform acceptance.

My separate source-only sanitizer run passes all three methods after fixing the
module loader's temporary owner leak. Only main.c, c_backend.c and module.c are
instrumented at -O0 in that CLI; other objects and external libraries are ordinary.
Leak detection is enabled. Registered declarations retain their own module-name
copies; I free the temporary checking owner before restoring the caller context.

I retain first failures: unused-helper warnings from overly broad new harness
flags; missing standalone service-predicate linkage; a qualified-name regression
caught once those API fixtures linked; and temporary imported-owner leaks. My
corrected harness flags retain C standard, declaration and return checks. I move
the real AST service predicate to a shared inline definition rather than stub it.
The repaired qualified-call path emits the resolved declaration's name.

Hosted run 35934003714 at 9964329ba still fails its scalar sanitizer worker:
compiler shadows exceed the unchanged 60-second deadline during bootstrap1.
The retained job 107426936599 log does not establish why Linux exceeds it.
The earlier Darwin stack repair is not evidence that this timeout is repaired.
Native callback failures, full hosted qualification, final-source fixed points
and release documentation remain required before #522 can leave draft.

My unchanged affine module identity suite also passes across the C seed and
existing native stages; only the C seed includes this module-loader cleanup.
