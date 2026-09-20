# My evaluator module-cache lease proposal

I record this design before changing cache ownership. My owner is
`task_60bef9462e22d7eb1724212ad1811803`, a required dependency of deferred
Environment lifetime task_f8b5ecacb7e4bcc1542712b4652308ad. My ordinary list,
tuple and scheduler source checkpoints remain unqualified. I have not executed
an invalid provider lifetime or repeated an old failure.

## My actual cache callers

`module.c` has one static `module_cache`, allocated by `init_module_cache` and
released with all of its ASTs by `clear_module_cache`. `load_module_internal`
returns a cached AST early or parses/typechecks a fresh AST before publishing it
in that cache. Its three current call sites pass `use_cache=true`: `load_module`,
`process_imports`, and private object compilation. I must register a provider
before either cached or fresh metadata can enter an Environment.

`compile_module_to_object` saves the outer ModuleCache pointer, sets the active
pointer to NULL, creates a separate module_env/cache, then clears that private
cache and restores the saved pointer on every relevant return. A global
"no Environment anywhere has a lease" check would reject this isolated cleanup
even when only the saved outer cache is borrowed. I therefore propose exact
cache-generation leases, not that global restriction.

Startup `clear_module_cache` in nano, nanovirt, DAP and browser entry points can
run before their new Environment exists. Local Environment preflights cannot
protect an earlier host-owned Environment in those cases. The check must live
at the actual cache destruction boundary. Root ASTs outside the cache remain
protected by the explicit Environment/provider preflight and the documented AST
owner contract; this proposal does not change arbitrary external free_ast calls.

## My ownership objects and API

I propose a small opaque `EnvEvaluationProvider` owned jointly by its ModuleCache
and every Environment registered with it. Its state is `alive`, checked reference
count, and checked evaluation-lease count. Its address is an allocation identity,
not a reusable numeric tag; retained references prevent reuse while any
Environment still names that generation. It never points into a freed cache.

- `EnvEvaluationProvider *env_provider_new(void)` returns one live cache-owned
  reference or NULL; no cache/AST is published on allocation failure.
- `bool env_register_provider(Environment *, EnvEvaluationProvider *)` deduplicates
  an Environment's edges, allocates before publication, checks alive/count facts,
  and preserves all old edges/counts on failure. If the Environment already has
  evaluation leases, the new provider atomically receives that many leases before
  new imported declarations become visible. This permits legitimate imports
  without leaving newly reachable providers unprotected.
- `bool env_provider_close(EnvEvaluationProvider *)` refuses unchanged while its
  lease count is nonzero; otherwise it marks the generation dead before its ASTs
  are destroyed. A dead generation cannot accept new registrations or leases.
- `void env_provider_release(EnvEvaluationProvider *)` releases a checked reference;
  the small ownership object remains until all Environment/cache references are
  gone. Closing a generation and freeing its ownership object are separate steps.

The existing Environment lease acquisition becomes a two-pass, nonallocating
transaction: validate its own count and every registered live provider count,
then increment every provider and the Environment. No callbacks or mutation occur
between those passes in this sequential API. Failure changes nothing. Release
subtracts one from the Environment and each currently registered provider.
Registering a provider during an active Environment lease uses the count rule
above, so each subsequent release has a matching provider acquisition.
Environment teardown first requires zero evaluation leases, then releases its
provider references without attempting to close the caches it does not own.

## My module integration

ModuleCache gains exactly one provider object initialized before cache publication.
`load_module_internal(use_cache=true)` initializes the active cache and registers
that exact provider with its Environment before cache lookup, parsing or definition
registration. Existing cache hits must not bypass registration. The currently
unused false mode does not fabricate a shared-cache edge; its AST owner would
need its existing direct provider lifetime contract.

`clear_module_cache` first calls `env_provider_close` on the current cache's
provider. Refusal prints `I cannot clear a module cache with pending evaluator
tasks.` and exits with status 1 before any AST/path/vector free. Success destroys
only that generation's ASTs, releases its cache reference, and sets the active
cache pointer to NULL. Saved outer generations are untouched. Private module_env
cleanup may close its own unleased cache while the saved parent has queued tasks.
If a private cache itself has a queued task, it refuses rather than silently
run/cancel it or restore an outer pointer after partial cleanup.

A still-existing Environment may retain a reference to a closed generation after
its unleased cache was explicitly destroyed. Its later attempt to enqueue a task
must fail before enqueue; the dead generation cannot be revalidated by pointer
spelling. This does not license ordinary execution through already-invalid
external AST ownership. My scope is deferred-task provider lifetime, not an
ambient proof cache or concurrent compiler mutation authority.

## My required controls

I require two caches and two Environments: an A lease must block only destruction
of A's actual provider, while isolated B close/free succeeds. I cover cached hits,
new module registration, registration during active leases, multiple Environment
edges to one generation, duplicate registration, every allocation-prefix failure,
checked count overflow, close then attempted enqueue, nested private cache swaps,
failed loads, and exact restoration of saved cache pointers. Startup clear must
refuse before AST destruction when an older Environment leases that cache.
Existing imports, private module compilation and teardown controls remain intact.
No cancellation or destructor invokes NanoLang user code. This design needs
review before implementation or qualification.
