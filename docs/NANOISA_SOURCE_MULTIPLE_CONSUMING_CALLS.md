# My multiple consuming source-parameter contract

I record `task_d54033e921f24e279e53e8ed4cf50d17` after runtime PR717 and
`task_6d8f539329294b24ad2746bec6a22862`. I pause implementation for the higher
priority affine example blocker `task_c4351c720aee424ea9b90187e51a08f2`.

I retain exactly main entry0 and helper1, with an int entry result and an
int/bool helper result. A consuming helper takes 1..8 mode-zero parameters,
each int, bool or an exact plain resource record, with at least one resource.
My source scalar boundary remains int/bool even though the runtime also
supports U8. Resource fields retain existing finite declaration-ordered trees,
int/bool leaves, depth and local-slot bounds. I keep borrowed-only CALL_REF
signatures separate; mixed references, owned returns, imports and deeper call
graphs remain outside this source slice.

I classify the whole signature before lowering a call. Each helper formal
gets its actual scalar tag or exact resource layout, a live mode-zero slot,
and its lexical name interval. Both ownership descriptors and declared
parameter tags retain positional identity. I do not emit all-STRUCT parameter
metadata when a scalar formal is present.

I prepare actuals once in source order. A resource actual must name a live,
unborrowed whole local with the exact nominal layout and no pending-disposal
flag. I emit OWN_MOVE_LOCAL and immediately mark that source moved. A scalar
actual uses my supported scalar expression forms and must match the formal's
exact tag. Materialized fields may be read before their owner moves; reading
one after an earlier argument moved it must fail. Duplicate owner actuals
therefore fail even when their record shapes match. I do not reorder or delay
moves to make an invalid argument list appear valid.

For this slice I refuse nested helper calls inside actual expressions. Named
scalar values, literals, supported operators and scalar field observations are
enough to qualify positional transfer without widening the call graph. I keep
constructed/projected/computed owner actuals refused. A call must denote the
exact unshadowed helper: static inspection found that the current raw producer
matches the helper name without checking a local binding of that name. I will
add that lexical refusal with the new call validation; checked source and raw
emission must agree on identity.

After every actual succeeds, I emit CALL1. A later lowering failure stops
emission and discards the unpublished module; it does not publish partial
metadata or restore a source owner by pretending the earlier move never
occurred. Runtime frame preflight and terminal cleanup remain PR717's contract.
Helper exits still require explicit complete consumption of every owner;
scalar arguments introduce no implicit resource disposal.

Both producers must preserve reaching-arm/loop owner states, lexical scopes,
local-name metadata and every selected shadow. Unsupported shadow graphs
refuse publication. I do not weaken the mandatory shadow gate.

My acceptance includes two distinct owners in both nominal orders, eight
interleaved int/bool/resource formals, nested resource values, scalar-before-
owner observation, helper-local observations, repeated calls and existing
branch/loop consumption. C-seed, Stage1 and Stage2 must retain exact metadata
and agree on VM/native results and selected-shadow execution. I check duplicate
moves, late wrong tags/layouts, observations after move, helper-name shadowing,
unconsumed parameters, unsupported arity and reference mixtures semantically,
with previous output retained. I convert the former valid two-owner refusal
fixture into positive acceptance rather than deleting it. Existing single-
owner and borrowed-only gates remain required.
