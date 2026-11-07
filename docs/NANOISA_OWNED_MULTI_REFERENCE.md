# I reconstruct one checked three-reference helper profile

I own this bounded #999 slice on `fix/5.1-owned-multi-reference` from `fc523280e5281b030cc9185f6eed40c1a345fe98`. My existing `source_borrow_multi.nano` fixture already passes both retained original producers, original verification and VM execution with result0; reconstruction currently refuses. I retain that source and its original assertions. Full required #999 scope remains open.

## I retain exact authority and simultaneous lifetimes

I retain the original two-function borrowed module: entry0 calls helper1, with exact scalar INT/BOOL result and exactly three scalar-leaf resource parameters in modes [exclusive, shared, exclusive]. Every layout remains completely validated and borrowed module depth stays at most2. I require three distinct live local owner identities and their exact positional nominal layouts. I admit one complete REGION_BEGIN, three ordered local borrows, CALL_REF, REGION_END sequence with consecutive exact reference slots starting at the CALL_REF base and full instruction coverage (REGION_BEGIN/END have no operands). No branch enters its interior. I preserve original verifier admission and reject path borrows, aliases, escaping references, mixed consuming calls, helper control flow and other arities. Caller owner identities survive the call; mutable bindings do not authorize moves or strengthen zero-iteration initialization.

I emit each scalar operand snapshot in its original order. REF_GET reads its exact parameter/field; REF_SET consumes exactly one matching scalar operand, once, and writes only an exclusive parameter. Wrapping INT ADD/SUB remain explicit staged statements using the established guarded formulas. Nano inlines these statements to preserve the original two-function admission; it does not add hidden helper calls. ASSERT remains an ordered always-active runtime operation, including native NDEBUG. A failed assertion is a trap, not successful owner cleanup.

## I select an actual positive witness without assuming every call succeeds

I derive witness field vectors from the actual entry prefix ending immediately before its first CALL_REF. I accept only a straight-line prefix whose referenced owners were constructed from exact literal INT/BOOL snapshots; scalar copies and owner transfers may preserve provenance, but computation, writes, unknown values or conditional provenance cannot manufacture it. Constructor operand evaluation and field ordering remain those of the original bytecode. I freeze each vector by value at this first call, not by a mutable source-local name. I refuse if any required vector is unavailable or an intervening operation invalidates it.

I admit helper ASSERT only when its BOOL operand has exact provenance from a field read of the shared parameter, propagated through scalar snapshots without transformation. The selected actual shared field must be true. Three distinct roots, no shared writes and no calls establish that exclusive writes cannot alter this asserted field. This narrow check proves only that the selected direct-call witness reaches successful return under the admitted total wrapping/scalar operations. It is not a general assertion solver or interpreter.

I preserve every original call and assertion. Later calls may supply a false shared field and must fail at the original assertion after the preceding writes; I never replace their values with the witness or assume all invocations succeed. A first call lacking a provable passing witness refuses reconstruction atomically even if some other possible input could succeed.

My generated helper shadow constructs two independent copies of the three captured owners, calls the helper directly once on each set, consumes all six owners and compares the two results and all final scalar fields. This is a deterministic consistency test. Original VM versus both reconstructed outputs supplies semantic evidence. Direct helper exercise is my explicit generated-shadow contract, not checker-proved coverage. I do not call main transitively, retain omitted original shadows, drop assertions or synthesize arbitrary true fields.

## I account for every emitted allocation before publication

I retain all function/instruction/field/layout/local limits and preflight generated Nano entry, helper and selected shadow independently against the retained128-local reconstruction cap. I count each parameter, scalar snapshot, both lexical inline arithmetic boundary declarations, constructor field staging, destructure owner/field copies, result capture and both complete shadow owner sets. Unknown statement forms refuse. I compare estimates to actual regenerated function.locals for the anchor, endpoints and near-boundary cases; conservative overcount may refuse but undercount is a defect. Previous output files survive every refusal. Existing single-helper output bytes and flat122/130 bounds remain unchanged.

- [x] I obtain contract review before production changes, then review symmetric facts/analysis admission and source rendering before coordinated tests.
- [x] I qualify the unchanged source, swapped exclusive arguments (13/17 then10/20), shared3/true, exact nominal identities, final consumption and wrapping endpoints through original VM, both reconstructed targets and sanitized O0/O2 natives.
- [x] I qualify a true first witness followed by a false shared argument, preserving ordered runtime assertion failure with NDEBUG, separately from a false/unknown first witness that refuses reconstruction.
- [x] I retain original-verifier versus defensive-facts distinctions for overlap, wrong mode/layout/slot, malformed region/count/interior targets, parent/path/reference escape and witness provenance failures.
- [x] I retain atomic output refusals, exact generated-local counts/boundaries, unchanged single-helper golden bytes and neighbor gates, source/provider hashes and every failed attempt before commit/push.

I run no bootstrap. The metadata peer owns the compiler slot until coordinated release. This contract does not claim general multi-reference, shared-alias, nested-path or arbitrary assertion support.

I retain [my qualification and review corrections](evidence/owned-multi-reference-20261019/README.md). Fresh aggregate qualification and full required scope remain open.
