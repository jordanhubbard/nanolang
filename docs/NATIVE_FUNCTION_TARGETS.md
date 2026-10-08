# My native function-target constraints

I must preserve function identity through NanoISA-to-C translation before I can
execute the returned-call programs on my native product route. My source lowering
emits `FUNCREF` and `CALL_INDIRECT`. My native translator now classifies named
function values and emits checked native dispatch for locals, globals, parameters
and returned functions. I also preserve function fields in native records, nested
records and record arrays, plus arrays of named functions. My self-hosted
source producer also admits these named-function containers. Captured-function
containers remain open.

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

My target-analysis pass does not itself validate argument value tags, admit
native storage types, allocate closure environments, resolve imported callable handles, or emit calls.
Other values begin as unknown target provenance; a target set does not authorize
an integer to act as a function. Unresolved call sites remain unresolved, rather
than gaining every function with the same arity.

I now collect `CLOSURE_NEW` target sets and pass capture producers to per-target
upvalue slots. Loads read those slots; stores add later producers. Transitive
closures therefore preserve callable provenance through intermediate environments.
I reject unknown targets, mismatched capture counts, stack underflow and invalid
flattened upvalue accesses. Captured arrays retain the ordinary alias constraints.
I conservatively union instances of the same closure body, but keep unrelated
function producers separate. These are target constraints, not the runtime
representation: native environment allocation and tracing are qualified below; source capture
lowering and closure arrays remain open.

## Native storage and dispatch

I use a distinct native function kind with a tagged `nmap_value` payload. Locals,
arguments and results preserve tag 11 and the module-local function index. I
check a dynamically tagged callable before dispatch and abort if its index is
outside that call site's proved target set. Ordinary integers cannot become
functions by sharing the same numeric payload. Callable slots enter tagged root registration: non-owning function IDs add no
heap edges, while closures retain their environments. Record, array, string and
map arguments retain ordinary caller and callee roots.

I classify every retained target through the shared direct-call classifier,
require compatible result signatures, and retain function references in required
function reachability. Emission reuses the direct-call builder and its record
pointer ABI. Each switch arm has its own argument conversions and direct C call;
I preserve the already evaluated callee and arguments and join results into one
native temporary. Exclusive arms reuse temporary indices while retaining their
maximum storage requirement. Returned records use the completed output shape
for subsequent field operations.

My native function-value cases now execute successfully, including void results,
functions returned through other functions, contextual byte/array/record
arguments, imported aliases and mutation of a global callee during argument
evaluation. My native record fields preserve the distinct storage kind, function
tag and target index through packing and projection. I check both storage and
payload tags before extraction, then the target set before dispatch. Function IDs
remain non-owning; closure fields retain their environment pointers and adjacent
string fields retain their ordinary roots. I test
nested records and record arrays across observed collections, plus malformed
storage tags, payload tags and target indices.

I store callable arrays in the ordinary owned word-array allocation with a
separate function-element storage kind. The owner lazily allocates environment
pointers for captured elements. Reads restore function or closure tags; missing
indices retain the void tag. Writes validate callable tags and replace both
target and environment. Aliases share mutations through locals, globals and
record fields, including growth. My collector follows environment edges and
reclaims unreachable array/environment cycles. Function IDs are not heap edges.
My deferred array-read shape preserves the optional result. Named equality uses
target indices; closure equality uses environment identity. Printing retains
my VM's distinct `fn` and `closure` notation.

My retained C-seed container module now executes in both VM and native products.
My self-hosted producer now emits that same named-function source successfully. Closures and other open release requirements remain necessary.

My acceptance remains the unchanged three native returned-call methods, the eight
VM methods, typed negative controls, VM/native aggregate and allocation parity,
and the complete compiler-product and release gates. Closures and the other open
5.1 language/backend requirements remain separate release obligations.

## Captured environments

I now lower `CLOSURE_NEW` to an owned record environment and pass that environment
explicitly to the native target. Flattened upvalue reads and writes preserve
scalar and aggregate representations. The capture fact slots follow bytecode
locals internally; they do not become source-visible local bindings. Runtime
environments remain distinct even when my analysis merges possible targets.

I preserve tag 15, environment identity, alias mutation and nested callable
captures. Native dispatch checks tag, target membership, environment presence,
capture count and the environment's stored target identity. Implicit entry and
initialization functions cannot receive captures and are refused before output.
I root the active environment, callable locals and operands, callable fields and
globals, including the indirect callee after popping it for dispatch. Existing
record-owner collection and shutdown cleanup reclaim environments.

My [native environment checkpoint](evidence/native-closure-environments-20261008/README.md)
executes the unchanged C-seed-produced canonical returned chain and exercises
managed captures across observed collection under ASan/UBSan/LSan. My self-hosted
producer still lacks lexical capture lowering. My [array checkpoint](evidence/native-closure-arrays-20261008/README.md)
now tests mixed named and captured elements, aliases, growth and cycle collection. The complete captured-function and release rows stay open.

## Constraint validation

My `test-nvm2c-shapes` fixture covers both join orders, root-rank changes, duplicate
and zero targets, node and target-set growth, late producers, conversion cycles,
recursive record fields, array elements, source isolation and invalid conversions
and projections. At this checkpoint it passes 2,553 checks, including an explicit
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
fixture passes with fresh ASan/UBSan objects and leak detection enabled; this measures target analysis. I retain the [analysis checkpoint](
evidence/native-callable-analysis-20261007/README.md) and its initial failures.

My [native execution checkpoint](evidence/native-callable-execution-20261007/README.md)
passes all three original returned-call methods, sixteen VM/native parity methods
and three native sanitizer methods. These check a sixty-target dispatch,
non-callable tags, absent targets, target zero, and record/array arguments and
results across observed collections. My broader native gate retains all 2,431
checks. These results do not establish container-function or complete 5.1 parity.

My [native function-field checkpoint](evidence/native-function-fields-20261007/README.md)
adds nested record/record-array execution and malformed-field controls. The
combined callable suite passed 24 methods; function arrays and self-hosted
container admission were still open at that checkpoint.

My [native function-array checkpoint](evidence/native-function-arrays-20261007/README.md)
adds empty construction, checked writes, alias/global mutations, optional reads,
identity comparisons, printing and observed collection. The combined callable
suite passes 28 methods. My clean [compiler-product checkpoint at f701198ad](
evidence/compiler-product-f701198ad/README.md) passes 89 methods; it predates the
record-field and array storage repairs and does not qualify their revision.

My [self-hosted container checkpoint](evidence/selfhost-function-containers-20261008/README.md)
passes the retained source and additional named-function container cases through
both products. Its 58 methods cover callable behavior and adjacent CLI/products.
I track visited signatures separately from record storage paths, retaining both
recursive callbacks and the existing record-cycle refusal. Captured closures,
current raw bootstrap and full release qualification remain open.
