# My consuming source-call boundary

I admit `task_c5208ffb6e494d7691ba9089c6a8e208` only after the merged
[runtime prerequisite](NANOISA_CONSUMING_CALL.md), PR709. My full ownership
parents remain open.

I route declared plain resource parameters through my specialized ownership
producer in both C-seed and selfhost NanoISA. I retain exactly main entry0 and
one helper1. A consuming helper has one exact nominal mode-zero owned record
parameter and an int/bool result. Existing finite, declaration-ordered nested
resource layouts and int/bool scalar leaves remain the source boundary.

I initialize helper slot0 as a live owner, retain its lexical name and require
explicit complete consumption on each exit. A named whole owner actual must be
live, unborrowed, exact-layout and not a pending disposal holder. I emit
`OWN_MOVE_LOCAL` followed by `CALL 1`, then mark that source dead. The runtime
preflights frame storage, transfers once and activates fresh helper references.
I keep borrowed-only `CALL_REF` signatures distinct.

I preserve exact reaching-arm and loop ownership states, source scope, mandatory
selected shadows and prior-output retention on refusal. I do not silently omit
a shadow that cannot fit entry0/helper1. My first source slice refuses projected,
constructed or computed actuals, mixed signatures, owned results, deeper calls,
imports and the other existing specialized-profile exclusions. I do not change
runtime authority or ordinary non-resource calls.

I qualify leaf/nested owners, formal/local observations, repeated calls,
branch/loop transfers, helper consumption and selected shadow assertions through
C-seed and both selfhost stages. Canonical metadata, verifier decisions and
VM/native values must agree. Exact wrong-owner/use-after-transfer/unsupported
shape controls must fail semantically and preserve previous outputs. Existing
source-borrow and lifecycle gates remain acceptance requirements.
