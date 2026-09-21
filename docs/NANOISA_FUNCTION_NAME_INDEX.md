# My function-name candidate index

## My measured dependency

I track task_2deaad56f65c497f80546220aa1ca9d0 under the full record-list scope.
My280b observer measures2,114,144 function lookups taking5.891s exclusive within
9.980s. This includes observer overhead and does not explain the final killed
interval. I retain original50de bootstrap failures and all previous evidence.

## My unchanged lookup contract

I index only exact function names to numeric candidate slots. I do not cache a
resolved function, declaring module, generated-list status, namespace permission,
borrowed Function pointer, AST pointer or signature. Every call applies the current
rules to current rows in ascending original slot order:

1. Qualified names still require the current importer's namespace and exported
   function name, then exact declaration module owner. Alias truncation and
   existing refusal behavior remain unchanged.
2. Local ordinary array_push wins only under its existing body/nonextern rule.
3. Actual builtin registry objects retain their current precedence and identity.
4. Generated-list names still prefer real declarations using the existing
   generated-list ordinal registry, including current-module then first fallback.
5. Ordinary lookup prefers current-module rows, then the first original row.

I replace full table iteration with ascending same-bucket candidate iteration;
exact string comparison still filters collisions. Each pass reuses the same
candidate iterator start without changing predicates. Builtin lookup need not
allocate an index, except the existing array_push local-precedence path.

## My owned storage and transaction

I add one optional opaque EnvFunctionIndex pointer, initialized by the existing
calloc Environment constructor and freed before the function table. The index
owns only bucket heads and numeric next links. A checked power-of-two bucket
capacity is at least twice the positive row count, with a small minimum.
Size multiplication, capacity doubling and integer slot bounds are checked.
All heads start at the absent sentinel. Building in reverse slot order makes
chains ascend by original index, including collisions and duplicate names.
Null names are skipped exactly as current ordinary lookup skips them.

I allocate/build an unpublished replacement first. Any allocation or bound
failure frees only that replacement, leaves declaration rows untouched and
uses the original linear scan for this call. I make no new fatal OOM route.
The completed index is published only after all heads/links are initialized.
No name strings are copied or retained. Teardown does not dereference names.

The index records its source row count and table storage identity solely to
notice append/truncation/relocation; this is not a pointer-only authority cache.
Normal append explicitly invalidates before table mutation. An explicit
invalidation helper supports in-place rename/replacement and REPL publication.
After an invalidation the next lookup rebuilds or falls back; an allocation
failure never returns a partial or stale candidate set. Raw same-count name
mutation requires invalidation, as with the existing symbol index API.

## My writer and reader audit

Production registration is env_define_function; generated-list publication,
checker/import collection and evaluator declarations call it. I invalidate
before its possible realloc and row append. REPL hot reload assigns a complete
Function into an existing live slot; I invalidate immediately before assignment.
The same-name assignment currently preserves the key but I do not depend on it.

create_environment uses calloc, and free_environment owns table teardown.
No Environment-by-value copy or separate table constructor was found in this
source checkpoint. tests/test_checker_metadata_ownership.c directly truncates
function_count to zero; count validation notices this without touching old names.
Module metadata extraction creates a distinct ModuleMetadata table, not an
Environment clone. The module transpile saved_main path temporarily changes
is_extern; predicates read it live, so no name-index invalidation is needed.
Current-module/namespace switches and generated-list registry changes also remain
live predicate inputs, not index contents. No callback runs during a lookup.

I inspect all remaining pointer-based Function updates before implementation
publication. Any newly discovered name writer gets explicit invalidation and a
regression; I do not weaken original visibility or nominal identity checks.

## My required acceptance

I retain all fourteen methods and full bootstrap/whole Make gates. Focused tests
exercise collisions, duplicate owner/name order, current-module changes,
qualified aliases with different importers, builtin precedence, actual local
array_push and generated-list/real declaration precedence. I compare indexed
and forced-linear results by exact live pointer, not just name strings.

I test append/table relocation, count truncation, explicit rename/replacement,
independent Environments and teardown after borrowed AST names are no longer
available. Every new optional allocation prefix and transient failure must
return the original linear result, preserve rows/old caller values and recover
on a fresh lookup. Existing ownership allocation sweeps include these allocations
where the real env.c hook domain reaches them. I do not claim expression-bounded
arena reclamation or recoverable allocation behavior for legacy fatal allocators.

No production index or acceptance execution is part of this design checkpoint.
