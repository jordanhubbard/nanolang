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
