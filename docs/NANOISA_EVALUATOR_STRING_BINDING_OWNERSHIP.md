# My evaluator string binding ownership correction

I retain the first 4382 terminals without reproducing the faulty workload.
Linux reports a double free. Darwin's existing crash report identifies an
invalid free at eval_scope_release+112; its image UUID matches the frozen
nanoc_c exactly. This establishes Darwin's cleanup failure but does not identify
the original allocation history or prove identical allocations failed on Linux.

## My concrete static defect

My checked env_set_var creates a fresh unmanaged string copy owned by a binding.
An identifier expression borrows that pointer. A later owning declaration can
publish the pointer unchanged because env_define_var_with_type_info only checks
exact arena roots and retains aliases for GC-managed strings. The two owning
bindings then free one unmanaged allocation independently. IF/COND forwarding
does not change the pointer or its existing owner. This is a static ownership
defect; I have not replayed it to establish a crash.

My field and tuple-index expressions already copy string leaves through
create_string. They do not establish a general nested-string borrowing defect.
I correct that earlier hypothesis here. The crash stack alone cannot select the
precise source binding among these paths.

## My ownership inventory and proposed boundary

- env_define_var wrappers delegate unchanged to the typed definition function.
  Fresh unmanaged strings passed to this low-level API transfer to the binding;
  a blanket copy would abandon that existing owner and is not my repair.
- AST_LET and AST_PAR_LET publish expression values; identifiers and forwarding
  branches may borrow an existing unmanaged binding root. Arena snapshots carry
  separate exact root provenance and already trigger a checked binding copy.
- Both actual invocation paths copy string parameters with create_string before
  binding. Field access, tuple indexing, array/list string iteration, string
  literals, ordinary string builtins and string function returns similarly use
  GC-managed copies. Module literal exports use create_string. Checker and
  emitter placeholder callers use void/scalar values or create_string.
- Checked owned graph copies use unmanaged strings internally. Existing arena
  string roots remain borrowed, and their binding copies are independently
  owned. My low-level API does not infer ownership of arbitrary nested pointers
  presented by an external caller.
- Borrow formals do not acquire string ownership. I preserve this exception and
  do not turn a borrowed declaration into an owning copy.

Before owning publication, I propose identifying an unmanaged string pointer
that exactly equals an existing owning VAL_STRING binding root in this same
Environment. I make a checked snapshot copy for that known alias. I leave the
original binding owner intact, transfer only the new copy to the new binding,
and preserve the original input on allocation failure. Exact arena provenance
continues to trigger its existing copy. Fresh unmanaged strings with neither
known provenance still transfer without an extra copy. GC strings retain the
existing retain/release policy. No global registry, name-based inference or
recursive graph search is introduced.

I will audit handler argument publication against its actual Environment and
include forwarding expressions in corrected-only parsed tests. Direct controls
will distinguish fresh transfer from borrowed copy by pointer identity, retain
the original after alias replacement/scope cleanup, and cover checked failure
without modifying either live owner. No known faulty workload is run before
source review. Complete source/evaluator/bootstrap acceptance remains required;
this defensive repair is not a retrospective allocation trace.
