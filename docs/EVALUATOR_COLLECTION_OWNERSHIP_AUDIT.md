# My evaluator collection ownership audit

I retain this static audit under task_992713bde1494772b0cb0b58bc9ee3c3. It does not authorize executing an unchanged failing ownership path. My original495 direct-leak stacks identify seven24-byte fixed Array headers and two40-byte NLHashMapCore headers. The exact retained source is `/home/jkh/nanolang-qualification/candidate-eval-pop2-linux/sanitized/source`; I checked the same allocation/consumer paths in candidate53cc4f4e039f2a8765b6200c2d8fc3abfb2302c0.

## Actual owners and escapes

My evaluator creates fixed arrays in17 places: array_new, array_slice, map, filter, vector unary/binary operations, map_keys/map_values, and empty/nonempty literals. My low-level create_array API allocates a header and buffer without an Environment. Its external callers remain responsible for their own values. Registering every low-level call would silently change that contract.

My literal/new/slice/filter paths independently copy string slots. Record slots own cloned record/string storage. Nested array slots preserve shared identity. Callback map results can return early and explicitly discard a partially initialized array. A new registry must publish only completed storage or detach its entry before that rollback. Retaining only local bindings misses abandoned expression inputs and early results.

My map_new path allocates a raw NLHashMapCore and returns its pointer as VAL_INT. The allocator comment promises a later ARC wrapper, but this path constructs no wrapper. env_free_value intentionally skips raw integer pointers. My public GC_FEATURES contract says HashMap needs no manual free, so adding map_free to existing acceptance fixtures would evade the defect. Explicit map_free currently destroys the allocation immediately and must detach any future registry entry first.

My map_keys/map_values string projections currently put borrowed map-entry strings into a mutable fixed string array. That conflicts with static_array_remove and string stores, which release string slots as owned. I must normalize these projections to independent exact string copies before applying a common owned-slot destructor. This is a demonstrated static ownership conflict with a potential double release; I have not executed that route. Allocation failures must clean only completed copies and leave the map intact.

## Proposed bounded repair and prerequisite

I propose an Environment-owned registry of actual evaluator-created Array and NLHashMapCore allocations, keyed by pointer and kind. Aliased bindings do not acquire duplicate registry entries. Destruction releases each array's owned string/record slots and buffer/header, without recursively destroying borrowed nested arrays; each nested evaluator allocation has its own entry. Hash maps use their existing destructor. Public caller-created arrays and arbitrary integers remain outside this registry.

This is cumulative Environment-lifetime retention, not local ARC, tracing collection or an RSS bound. It depends on the separately owned completed-task/result lease and cross-environment admission correction: env_clone_value_snapshot currently copies reference leaves by borrowing identity. A task result or public snapshot must not retain such leaves beyond destruction of their source owner. I must qualify that dependency before freeing collection roots.

Required controls preserve alias identity and mutations across local scopes, nested arrays and returned record fields; cover all evaluator constructors and partial callbacks; distinguish explicit map_free from retained aliases; verify map projection strings remain independent under map/array mutation; and check allocation refusal/recovery before publication. Original full test-eval and whole owning sanitizer configurations remain required after integration. No fixture suppression, manual cleanup substitution or broad leak acceptance follows from this proposal.
