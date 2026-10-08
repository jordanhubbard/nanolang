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

## Bytecode target analysis

My `nvm2c_callables` pass now collects target constraints before native
representation inference. I retain stack-value nodes at control-flow joins,
shared local/global storage nodes, and direct/indirect argument/result flows.
Each newly discovered indirect target adds that callee's argument and return
constraints; I solve again until no new call edge or mutable alias appears.
I check target indices, arities and result counts when connecting a call.

I retain function values inside records, arrays and maps. Array/map copies carry
shared element storage, including handles nested inside copied records. Writes
through either alias therefore reach calls through the other. Function scalar
assignments still flow forward, so a destination's other targets do not rewrite
an unrelated producer. My analysis is conservative and does not execute branch
conditions or distinguish map keys and array indices.

I run this pass in `nvm2c` when the module contains both function references and
indirect calls. Without a reference producer it adds no target evidence, and I
retain the existing classifier refusal. I preserve analysis storage through
classification/emission and release it on success or failure.

This pass does not validate argument value tags, admit native storage types,
resolve closures/imported callable handles, or emit calls. Other values begin as
unknown target provenance; a target set does not authorize an integer to act as
a function. My native classifier must consume the completed sets and retain all
ordinary type, ownership and signature checks. Unresolved call sites remain
unresolved, rather than gaining every function with the same arity.

## Connection still required

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

My `test-nvm2c-callables` fixture passes 204 checks for returned functions,
indirect function-valued arguments/results, branch and loop joins, globals,
record/array/map aliases, zero-valued target IDs and malformed inputs. It also
checks that the integrated translator rejects mismatched arity/result counts.
Including my archived source-generated returned-function module raises this to
300 checks and proves the expected four call-site target sets. The complete
fixture passes with fresh ASan/UBSan objects and leak detection enabled; native
indirect execution remains unimplemented. I retain the [analysis checkpoint](
evidence/native-callable-analysis-20261007/README.md) and its initial failures.
