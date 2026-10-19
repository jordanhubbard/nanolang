# My native shared-record identity prerequisite

I discovered the missing `AGG_SET` implementation while qualifying #1001's array-field authority. My VM mutates a shared record allocation. My current ordinary native representation copies `nrec_t` values and snapshots them into retained owners. Adding a field assignment to a copied native value would lose alias visibility, so that is not a correct implementation.

I propose a separate native child after the public consumer baseline. I keep this contract distinct from implementation or qualification evidence.

## My bounded representation change

For an admitted ordinary array-field module that needs aggregate field replacement, I give each record constructor one canonical owned allocation. Native `nrec_t` values retain pointers to that allocation's field vectors and its canonical identity. Local, argument, result, global and nested copies preserve those pointers. The record width remains bounded by the existing checked native shape analysis; I use checked allocation arithmetic and preserve exact field storage tags.

I extend the existing `nrec_owned` pool and traced root walk rather than adding a disposal-only lifetime. A wrapper root marks its canonical record owner; canonical field tracing then marks child arrays, strings and records. Multiple wrappers may share one owner. Sweep frees an unreachable canonical owner exactly once. `nrec_snapshot` returns the canonical identity in this profile, instead of inventing a detached copy. No pointer survives collection unless reachable from the existing frame/global/result roots.

My other native profiles keep their existing representation until separately qualified. The new route has no captured/indirect or foreign callbacks, linked modules, resource/reference contracts, cyclic record declaration graph, or record-valued array elements. I must not infer that these excluded consumers acquire shared-record support.

## My lowering and failure obligations

I lower `AGG_SET` only after the complete original-module proof and native shape inference agree on receiver, field and stored representation. Both bytecode operands are evaluated once before mutation. I mutate the canonical field vectors and return the same record identity; aliases observe the update. Retaining a replacement child precedes retirement of its old edge, with collection only after all live temporary roots are represented. Wrong tags, missing fields, allocation failures and first-error cleanup preserve the existing runtime failure contract.

I compare VM and generated native execution at O0/O2 with ASan/UBSan for local aliases, cross-call aliases, globals, nested aliases, retained GET values, repeated updates, old-child retention, unreachable old-child collection, constructor allocation failure and error prefixes. I require live-owner/byte evidence before disposal, not just clean process exit. I retain ordinary nonmutating, owned-array, mixed, service and source-without-authority regressions. Only then may the bounded public admission stop refusing `AGG_SET`.

This child does not close #1001's remaining generic, source-publication, wider nominal and cross-platform obligations.
