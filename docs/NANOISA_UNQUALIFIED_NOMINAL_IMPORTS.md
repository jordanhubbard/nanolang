# My unqualified nominal import provenance prerequisite

My fresh 132f full Make build passes all three component shadow graphs on both
hosts. Explicit bootstrap reaches compiler/module_loader.nano and rejects the
array<Symbol> boundary at line347. That source explicitly imports Symbol and
NSType from typecheck.nano. Their original records are unmangled; the annotation
retains only the name while my exact resolver receives module_loader as owner.
No importer-to-declaration fact connects them. I retain both first bootstrap
terminals in `evidence/record-lists-132f-preparation/` and complete original
source/tool/product maps at the frozen preparation roots.

I do not repair this by admitting a globally unique spelling. Same-spelled
ordinary records from unrelated modules must remain distinct. My proposed
source checkpoint adds explicit Environment-owned nominal import bindings.
Each immutable row retains importer identity (NULL only for root), local name,
nominal kind, original declaration owner/name and registered ordinal. It owns
all copied names; no imported AST pointer escapes into this registry. Repeated
identical edges are idempotent; conflicting origins for one visible name remain
ambiguous and refuse independent of load order. Checked registration prepares
all owned fields before publication and never loses an earlier row on failure.
Environment teardown frees each row exactly once.

My declaration resolver first honors an exact local declaration. Otherwise it
may follow only an explicit visible import row and must validate the retained
kind/ordinal and original owner/name against current registered declarations.
Qualified namespace lookup and my separate foreign ABI namespace rules remain
unchanged. Canonical rewritten names continue to denote their registered owner.
Annotation rewriting must use this same resolver; the legacy env_get_struct
name-only fallback cannot silently choose a competing module.

I will register rows from the completed actual import AST and the registered
source declaration facts after loading, never by guessing a matching name in
the global Environment. Unqualified selective imports register only selected
nominal names. A namespace alias keeps its qualified policy. Unqualified plain
imports need explicit direct and transitive visibility propagation consistent
with the existing language contract; I must audit this before implementation,
including selective type aliases currently routed through function-only alias
handling. I do not silently extend that syntax or reduce accepted legacy scope.
The same mechanism covers record, enum and union declaration identity; it does
not alter enum numeric conversion or admit enum list operations.

My focused controls must include actual parsed selective and plain imports,
transitive declaration references, aliases as already supported, definition-site
field/call/return annotations, exact array/list leaves, local same-spelled
records, unrelated modules, conflicting imports in both orders, repeated
imports, undefined selected names and checked allocation rollback. Existing
qualified-import and full list controls remain unchanged. The original
module_loader bootstrap is mandatory after source/fixture review.

This is a design checkpoint. I have not changed the resolver or executed a
corrected bootstrap. Full compiler, evaluator, enum-list and release parents
remain open.


## My visibility audit before source

CANONICAL_STYLE.md explicitly leaves private visibility and selective type
aliases open; I do not claim to complete either here. MODULE_MIGRATION_GUIDE.md
requires each file to declare its own imports. The existing module loader
registers qualified aliases in their importing owner and includes each imported
AST's direct declarations without filtering pub. Its unaliased selective alias
loop skips an unchanged spelling; renamed selective symbols still require a
Function. I preserve those boundaries rather than silently enabling type aliases.

My first implementation therefore registers actual direct nominal declarations
from an unaliased plain/wildcard import, or the selected unchanged nominal names
from an unaliased selective import. A namespace alias remains qualified. A
wrapper's own import rows remain scoped to that wrapper; definition-site
function/field annotations follow their original owner and can resolve its
transitive dependency rows. I do not promote a dependency's local namespace into
the caller. Parsed controls must distinguish this definition-site transitivity
from unbound caller spellings. Qualified and selective alias refusal behavior
remains unchanged. Later full bootstrap may expose other legacy accepted
boundaries; I will record and review them rather than bypass exact provenance.


## My complete source and fixture checkpoint

My Environment owns a linked table with four optional/required copied strings
and a kind/ordinal pair per row. Registration validates the existing declaration,
allocates unpublished storage, and atomically prepends only complete rows. Exact
repeats allocate nothing; same importer/name/kind with another origin returns
false without changing the existing table. Resolution revalidates copied
name/owner against the ordinal and never chooses a global same-spelled record.
A real local declaration takes precedence. Teardown detaches and frees rows
before declaration storage. The Environment constructor's calloc initializes
the new field; no Environment-by-value copy is introduced.

The actual module loader registers only selected direct declarations after a
completed load. Renamed selective type aliases remain on the existing refusal
path. Qualified aliases retain their original table. The record binder now
uses exact declaration/import authority instead of env_get_struct's global
fallback; union placeholder classification uses the same owner-aware resolver.
Registration is transactional per row; an import failure stops processing,
without claiming rollback of previously completed imports or declarations.

The original thirteen methods remain, with one new method (fourteen total).
It exercises selective/plain/wildcard/repeated imports, definition-site field
and array/list annotations, local same-spelled records, both conflict orders,
undefined selected names, no dependency-namespace promotion, and unchanged
selective type-alias refusal across the existing producer/evaluator routes.
Actual env.c allocation hooks exercise every registration allocation for NULL
and named importer/source owners, persistent/transient failure and recovery,
retained old rows, allocation-free idempotence, conflict ordering, stale
ordinal facts, kind separation, and independent Environment teardown. All
previous allocation controls remain. Python parsing and diff whitespace checks
are the only checks run before review; no C build or source program ran.


## My cross-kind review correction, before source

Ordinary nominal annotations in parser.c use a generic nominal spelling and
TYPE_STRUCT placeholder. nominal_union_kind refines that spelling using its
declaration; kinds are not separately named by source annotation syntax. I
therefore treat importer/name as one visible nominal namespace. Registration
must reject a different kind as well as a different ordinal/owner/name for an
existing binding. Idempotence requires all facts, including kind, to agree.
I add STRUCT/UNION/ENUM pairwise both-order controls and preserve old rows on
refusal. My binder must honor local AST record declarations before an imported
union can refine their same spelling; I add that local-precedence control too.
This static finding is part of task_836004405a924cea8d45829cbe14ab61; no ambiguous
program has been executed.
