# My explicit numeric-union storage contract

I track this bounded implementation as `task_87a7b44d10e240e999485d12aeecaca2`.
I introduce an explicit NUMERIC shape leaf for the set INT|FLOAT. I use it as
an OPTIONAL present-payload shape for proven numeric boxed storage. OPTIONAL
still describes the same tagged carrier and retains its existing absence
semantics; NUMERIC itself admits no void, bool, U8, string or heap payload.

Exact unification remains strict: INT, FLOAT and NUMERIC are distinct.
A directed storage conversion may inject INT or FLOAT into an explicitly
NUMERIC destination, preserving the source's exact kind. I do not infer a
union from conflicting exact constraints. Existing OPTIONAL(INT),
OPTIONAL(STRING), map payloads and array elements retain their constraints.

Checked numeric producers establish closed numeric provenance. Join plans,
local stores and actual call arguments may combine that provenance with exact
INT/FLOAT producers. Unknown or heap provenance does not authorize widening.
Existing void provenance on optional local storage preserves absence only;
it does not turn an unproved value into a numeric one.

I retain tags at mixed numeric branch/loop edges, locals and calls. Runtime
consumers still check a requested scalar tag before unboxing; I never implement
integer-to-float coercion merely by forgetting an integer tag. I require
conversion-order-independent solver tests, negative exact/payload/provenance
controls, paired VM/native GCC/Clang sanitizer execution and existing solver
and native gates. Integration with the independently held optional-storage
branch is a separate ordinary focused check, not compiler startup acceptance.
