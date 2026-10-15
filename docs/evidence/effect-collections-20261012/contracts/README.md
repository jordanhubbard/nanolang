# My exact array effect declarations

I add EFFECTS revision 2, which prefixes the existing operation rows with a
bounded element table. ARRAY descriptors index that table; nested ARRAY rows
refer only to earlier rows. Scalar leaves retain NO_INDEX, and nominal leaves
retain their exact complete STRUCT/UNION layout index. A table may hold up to
256 rows. Revision 1 retains its prior representation and rejection categories.

The new `nvm_ownership_effect_types` query validates the complete ownership
payload before copying any output. It returns an empty table for revision 1.
My tests cover nested arrays, distinct record identities, resource versus
ordinary record leaves, all truncated prefixes, malformed descriptors and
references, self/forward edges, capacity boundaries, unsupported revision/kind
combinations, and unchanged outputs. Binary and unverified assembly round-trips
preserve every ownership byte. These tests do not establish executable array
ownership or cover every source collection kind.

My affine perform and handler-state paths explicitly refuse ARRAY signatures
until their element/ownership flow is implemented. The positive transport
fixture has a matching zero-argument perform, so its execution refusal does not
rely on an arity mismatch. Ordinary assembly, VM verification and native emission
remain closed to this unsupported execution.

The contract harness and the instrumented reader/test harness pass 1,563 checks.
The latter uses ASan, UBSan and leak detection; other linked objects keep their
existing instrumentation. I retain exact compiler/link/run arguments. The
existing affine suite passes 1,747 ordinary and 2,555 fault/capacity checks.
The shared C-source effect suite and 26-fixture VM/native fault/cleanup suite
also pass. These are regression checks, not array-effect execution qualification.
No fresh Nano compiler product is claimed at this declaration checkpoint.

I investigated missing explicit Make prerequisites and found that generated `.d`
files already cover these includes. I removed the redundant proposed Make edits;
`dependency-existing.log` records the unchanged build's forced-newer response.

Next I must connect this exact type table to local declarations, array origin
and ownership analysis, handler entry/resume/lexical-return flow, producer
lowering and matched runtime cleanup. Resource payload layout integration,
other collection kinds, borrowed continuations and full 5.1 remain open.
