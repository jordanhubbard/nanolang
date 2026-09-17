# My concrete generic selected payload boundary

I transfer the selected fields of a fully instantiated generic union. I retain
concrete payload metadata through matched bindings and complete selected
patterns. A resource-bearing sibling does not give an ordinary or empty arm
an ownership obligation.

My initial supported route uses exhaustive, distinct, unguarded arms and
complete selected patterns. Declared locals, direct function parameters,
returns and supported branch values retain their constructor context. Nested
unions use explicit typed temporaries. My checks reject dropped owners,
repeated consumption, reuse after matching, partial moves and incompatible
joins. Resource collections and unresolved tuple/row/function payload shapes
remain unsupported.

I do not claim complete affine analysis, generic function-value signatures,
global resource ownership, arbitrary direct nested field-scrutinee inference,
borrow checking or ownership verification in NanoISA. Those remain separate
roadmap obligations.

## My native match prerequisite

My checked match result must retain its nominal identity. Contextual generic
constructors now resolve their registered concrete name; dotted nongeneric
constructors resolve the owning union declaration. I borrow those names from
the environment rather than retaining another allocated spelling. My
self-hosted emitter initializes scalar and aggregate match results with C's
`{0}` form. Integer selectors, union selectors, generic union results and
ordinary scalar results have executable controls.

## My checkpoints

At `2ed1c522`, I completed a fresh native bootstrap and all 18 ownership
methods with Stage1 (29.929 seconds). At `b00ac9ca`, my C seed passed those 18
methods plus three ordinary match-result methods (17.945 seconds). Both
self-hosted stages passed the three ordinary match methods (19.224 seconds).

At `b00ac9ca`, an isolated ASan/UBSan harness repeated all 18 ownership
parser/typechecker cases 20 times: 360 checks passed. It freed environments,
ASTs and tokens after each case. Leak detection remained disabled under
`task_00c47a5d65d04c48914864ec0de553d6`; this is not a leak-freedom claim.

My fresh bootstrap passed with compiler source `b00ac9ca`. At `00500bdb`
(the same compiler source plus a guarded-generic regression), my combined
84-method gate passed 83 methods and failed the existing C-only
`test_constructor_contexts_preserve_values` at the selected generic payload
array append. I reproduced that failure with the parent worktree compiler too;
`task_17ace696e309484bbff5a59acc2891db` retains the unchanged positive gate.
My full C typechecker tests passed. I do not call this combined gate green. Local logs are under
`/tmp/nanolang-generic-selected-transfer/` (`bootstrap-r5.log`,
`c-match-final-tests.log`, `selfhost-match-controls.log`, `asan.log`).

MAC: `task_08428ceb1d674de49383aab1ba9a78c8` and native prerequisite
`task_ce96caed26b843ecb1365def0c733b58`.
