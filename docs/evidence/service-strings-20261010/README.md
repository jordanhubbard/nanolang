# Checked service strings

I extend checked service execution under GitHub issue #990 with counted immutable
strings. My C and Nano source lowerers support string literals, parameters and
helper results, locals, direct and indirect calls, equality, inequality and byte
length. My File, TCP and mixed service engines execute the same checked operations
in NanoVM and generated native C. Embedded zero bytes remain part of the value.

I own each copied value in a bounded invocation pool. Equal byte sequences share
storage; dropping a local does not reclaim a distinct string before invocation
finish. I admit at most 4096 distinct strings, each at most 1 MiB, and charge the
node plus bytes to the existing runtime byte limit. I free the entire pool on
finish, including error cleanup. Dynamic host values use the same copy API;
integer scalar injection cannot construct a string identity.

I validate literal indices and operand types before execution, relocate literal
indices during canonical serialization, and deduplicate literal bytes against
metadata names. My retained failure for `nsi:nanolang/filesystem#File` demonstrates
the metadata collision that this deduplication repairs. I also retain the initial
indirect-target, hosted-opcode and runtime-payload refusals instead of reporting
only the passing executions.

## Evidence

I ran these checks on Darwin with LLVM Clang and, for the ordinary manual runtime
checks, GCC 16. My logs are beside this file. `input-hashes.json` identifies the
implementation and compiler artifacts before the final Make target. That target
rebuilds the C tools after my header comment; `final-make-tool-hashes.json`
records those tools for the five-method Make result. `driver-commands.json` retains
130 individual command results from the source execution runs.

- The updated Nano compiler, executed as bytecode, passes all four initial string
  methods in 26.005 seconds and the subsequently added TCP method in 6.542 seconds.
- The same compiler translated to native C passes those four methods in 8.732
  seconds and the TCP method in 2.744 seconds. Each run also exercises the C-seed
  native and bytecode drivers. Source tests compile selected shadows, execute VM
  and native outputs, check grant refusal, and preserve prior output on malformed
  literals, wrong operands and invalid source types.
- The final runtime pool passes ASan, UBSan and leak checks: 112357 instrumented
  assertions and 26561 linked assertions, including 436 allocation refusals.
  GCC passes the same two methods. These manual carrier checks establish copied
  storage, empty/embedded-zero values, intern limits, allocation/budget refusal,
  forged-scalar refusal and File cleanup; they are not network execution tests.
- My adjacent body, ownership, C lowering, wire and Nano lowering targets pass
  40 methods in total (5 + 5 + 13 + 2 + 15). The new builtin-probe regression
  is recorded separately above the original five body methods.
- My new Make target passes all five source string methods in 6.964 seconds.
- My checker probe regression first reproduces an assertion on a builtin fact,
  then passes through the corrected C and Nano probes in 36.517 seconds. I
  compare the reported builtin meaning, retaining each frontend's internal
  sentinel representation.
- The final Nano compiler build runs its selected compiler/dependency shadows.
  I retain its build log and native translation log. This is one updated compiler
  build, not a new Stage1/Stage2 fixed-point qualification.

I keep the final compiler artifacts and their referenced host cache under
`/private/tmp/nl51-string-compiler-final*` and
`/private/tmp/nl51-string-compiler-cache`; removing that cache can invalidate them.

## Remaining scope

I have not admitted string-bearing catalog fields or service results. Ordinary
scalar strings in service helpers are the dependency established here. Public
DNS still needs deadline supervision and authority-aware catalog bindings;
affine WebSocket bindings and exact-candidate Linux/Darwin qualification remain
required. Entry results remain int/bool; helpers may return strings. This
checkpoint does not close issue #990 or release 5.1.
