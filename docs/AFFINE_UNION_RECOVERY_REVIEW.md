# I reconcile the recovered affine union candidate

I reviewed peer checkpoint `a9ea7d69bff89ca413e179b587b6bdb549d0a857`
without executing it. It is unfinished work for
`task_a18a9f752536469faafc4d3ebec01dfd`, not release evidence. I preserve that
branch and the earlier `45abffce66338460d9595095500bd33ff7e55dc6` recovery.

My canonical baseline includes PR877's parenthesized parser fix and PR889's
qualified guards at `dc106ff2aa069d88feec9851fcc77e3d9ebdc0c9`. I integrate
selected union changes onto that baseline. Replacing whole parser, checker or
transpiler files would discard separately qualified behavior.

## I retain useful work and record its limits

The candidate adds concrete payload substitution, BOOL/FLOAT/STRING payload
tags, union call/result identity and statement ownership joins. It corrects
record-plus-union table counts. These are source observations, not passing
acceptance results.

Its identity registry accepts only one concrete instance per declaration.
Simultaneous instances of the same generic union remain required. Its payload
table also requires one common scalar type at each position across variants;
heterogeneous variant shapes need an exact representation before admission.
Value-producing matches in the affine emitter remain unfinished.

The candidate's retained union representation describes a maximum payload
shape. It does not establish an exact selected-variant constructor contract at
the raw-bytecode boundary. I require retained variant membership, exact arity
and payload facts, with agreement between verification and both execution
routes, before extending public admission. Source constructor checks alone do
not establish that boundary. This review makes no claim of a measured runtime
failure or memory corruption.

I preserve the canonical once-only scrutinee, source-order guards, lexical
payload scope, exact BOOL checks, purity/resource checks, enclosing control
flow and emission-error restoration. The candidate's outer moved-bit check
does not replace the existing moved/held/terminal state checks. Its unrelated
native `-O1` default change requires separate justification and is not part of
this integration.

## I finish in dependency order

I first reconcile implementation ownership with the peer. I then specify the
concrete-instance and variant representation, review the source integration,
and review its fixtures before fresh execution. Acceptance retains the full
generic suite and selected shadows through C seed, Stage1, Stage2, VM and
native routes on Linux and Darwin. Earlier guard, parser, affine and module
identity evidence remains separate; fresh integrated gates cannot be inferred
from those earlier runs.

SSH delivery to the active peer thread succeeded with queue receipt
`01a0bca5-32f1-7d53-832f-e079e00efe3b`. Delivery is not acknowledgement.
PR522 remains a publication hold. This review closes neither a18 nor full5.1.
