# Empty callable-loop translation

I translate the empty callable loop retained in my
[preceding loop checkpoint](../callable-array-loops-20261010/README.md).
My ordinary callable analysis still requires a resolved target set for a call
that can execute. When that set is absent, I run bounded constant flow over
locals and the operand stack. I only admit the site when this analysis leaves
it unreachable. My emitted site retains a terminal invariant trap.

I track integer and Boolean constants and literal array lengths. Conflicting
facts at joins become unknown. Backedges revisit the flow until it stabilizes.
I discard all array-length facts on mutation and calls, including aliases in
locals and on the operand stack; newly allocated arrays can establish fresh
facts afterward. I decline unmodeled reference, raw-memory and effect/control
operations. Unsupported facts and analysis limits preserve refusal.

My pass caps instructions at 65,536, locals at 1,024, transfer visits at 262,144,
and retained frame values at 16 MiB. Decoder and working storage are separate.
I do not claim complete dead-code inference or general alias analysis.

My eight focused methods pass (`focused.log`). They execute an assembly empty
loop, the original source loop with result/break/continue, and a fresh empty
array after mutation. They preserve refusal and prior output for unknown
parameters, conflicting branch facts, a backedge that reaches the call,
mutation after length provenance, and a call boundary. My translator's nvm2c
component and this new pass also run all eight methods under ASan/UBSan with
leak detection enabled (`sanitized.log`); generated positive executables use
the same sanitizers. Other linked translator components are not instrumented
in that run.

I add these tests to `make test-nvm2c` through
`test-native-dead-callable-paths`. I run with Homebrew Clang for LeakSanitizer:
`make -f Makefile.gnu test-nvm2c CC=/opt/homebrew/opt/llvm/bin/clang`.

My final full gate passes all 2,443 C AOT execution checks, 379 callable
constraints, 3,098 shape constraints, and its Python prerequisite suites
(`full.log`), including the eight new methods.

I keep callable functional operations, the component shadow deadline, and the
full 5.1 candidate/platform gates open under #976.
