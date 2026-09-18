# My bounded borrowed-source return paths

MAC `task_b077ad608868462da669b5a7a427567d`.

I track fallthrough separately for each if/else arm and while body. A return
terminates only its path. I emit joins and backedges only from paths which
continue; a loop's zero-iteration exit still continues. Both returning arms
satisfy the function return requirement. I do not infer termination from a
constant condition.

My first slice admits scalar early returns from borrowed helpers, whose caller
retains ownership, and from entry paths after explicit top-level consumption
of every owned local. I refuse an early return with a live owned local rather
than introduce an implicit drop. My [resource-path extension](NANOISA_SOURCE_RESOURCE_PATHS.md) separately
admits explicitly consumed branch/loop resources and exact reaching joins. I snapshot and restore incoming local liveness
between paths; the unchanged verifier checks exact ownership/reference/region
facts and definite scalar initialization. Existing expression lowering closes
borrow regions before returning its result.

I close lexical names at their body boundaries and preserve outer bindings.
I retain return refusal in every selected shadow, including nested bodies;
all selected shadows still execute or publication refuses. Broader ownership
movement, break/continue, deeper call graphs and path-dependent disposal remain
separate work.

My acceptance compares both producers and canonical Stage1/Stage2 code,
contracts, names and selected shadows. I execute one-return and both-return
branches, helper mutation before return, zero and entered loop paths, and
entry returns after explicit disposal in VM and sanitized native output.
Ordinary refusal controls retain live-owner, shadow-return, resource-move and
missing-return boundaries without changing verifier authority.

My explicit resource-path extension is MAC `task_d74d8a4fb4a048a786666195eaa4e8d5`; its separate contract preserves the remaining assignment and partial-move refusals.

My disposal-provenance prerequisite is MAC `task_6a55c8c1e40a4923804e20b450d32cdf`. Complete explicit leaf patterns may retain compiler holders until terminal cleanup; I distinguish these holders from unconsumed source owners with explicit slot metadata, never names.
