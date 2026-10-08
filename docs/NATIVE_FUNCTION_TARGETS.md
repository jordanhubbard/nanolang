# My native function-target constraints

I must preserve function identity through NanoISA-to-C translation before I can
execute the returned-call programs on my native product route. My source lowering
already emits `FUNCREF` and `CALL_INDIRECT`. My native translator still lacks their
classification and emission; this constraint layer does not close that gap.

## Representation

I retain a distinct `NVM_SHAPE_FUNCTION` leaf with a sorted, duplicate-free set of
function-table indices. Index zero is a valid target. An empty set means I have
not resolved a producer; it never authorizes dispatch to every function of the
same arity. My shape graph has no module pointer: its caller must validate every
target index and its signature against the actual module.

I union target sets when exact shapes unify. I propagate targets only from source
to destination along storage conversions. Adding a target after an earlier solve
requires another solve and reaches every connected destination, including cycles
and fields within recursive records and arrays. Storage does not add its other
producers back into a source's target set. Exact joins can conservatively merge
sets; they do not prove that every retained target executes at runtime.

I reject function/integer unification and conversion in either direction. A
function leaf has no record or array projections. I do not extend optional scalar,
variant scalar, host import, or ownership admission merely by adding this kind.
My graph retains its existing poison-on-error rule and frees target storage on
root merges and destruction.

## Connection still required

My current native classifier collects ordinary shape constraints only during its
final pass. Indirect calls need targets earlier, to discover parameter and result
representations. I must connect target propagation to that fixed-point analysis,
including locals, globals, direct call parameters/results, branches and aggregate
storage, before using the completed shapes to emit calls. I cannot simply inspect
final shapes from an earlier pass or treat an unresolved callee as an integer.

For each indirect call I must validate every retained target's arity, result count
and argument/result representations; propagate arguments and returned values; and
include referenced functions in required-function reachability. My native emission
must evaluate the callee and arguments once, dispatch with a checked target ID,
and preserve the same record pointer ABI, frame roots and cleanup as direct calls.
A function-valued result must preserve its own targets for later calls. Unresolved
or incompatible target evidence must fail translation with prior output intact.

My acceptance remains the unchanged three native returned-call methods, the eight
VM methods, typed negative controls, VM/native aggregate and allocation parity,
and the complete compiler-product and release gates. Closures and the other open
5.1 language/backend requirements remain separate release obligations.

## Constraint validation

My `test-nvm2c-shapes` fixture covers both join orders, root-rank changes, duplicate
and zero targets, node and target-set growth, late producers, conversion cycles,
recursive record fields, array elements, source isolation and invalid conversions
and projections. At this checkpoint it passes 2,527 checks, including an explicit
Homebrew LLVM run with address, undefined-behavior and leak sanitizers enabled.
This establishes constraint behavior, not native indirect-call execution.

My adjacent `make test-nvm2c` gate also passes all 2,431 structured-C checks and
the opcode-coverage and sanitizer-driver controls. I reran the normal shape
target after adding the final late-producer and target-set-growth controls.
