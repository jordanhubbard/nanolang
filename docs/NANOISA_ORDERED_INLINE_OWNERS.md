# I construct inline child owners in source order

I execute task_f4a34feabe954cc2921151c2819edc66 under product-affine e8d860 and ownership28f2. This contract starts from canonical `4fc20434c99181a9471a79c34ecd329c19f3e693` and precedes production. I preserve the original950f artifacts without execution. The two unchanged Bundle methods in `tests/test_owned_record_patterns.py` need inline Handle children, but their STRING/array fields require separate managed-owner contracts. This child does not complete those methods or the full parents.

## I preserve my admitted representation

My C `borrow_owner` and Nano `nb_owner` currently evaluate constructor fields in source order into hidden slots, then load/move those slots in declaration order for OWN_PACK. Nested fields currently accept only exact live named owners. I extend that expression boundary to recursively constructed exact child literals and already-admitted scalar-leaf owner factory calls. I retain named owners, their move semantics and exact nominal indexes.

I keep prior-child layouts, COMPLETE owned facts, existing INT/BOOL source leaves, depth32, field count256, physical local count256, and the acyclic eight-function/frame contract. Runtime U8 support is unchanged. I add no opcode, metadata schema, borrowed referent, managed field or result type. Nested-result ebdc86 is independently owned: this child neither removes its guards nor depends on widening results. After its merge, I preserve its qualified behavior without counting it as new inline-construction scope.

## I stage each child before evaluating the next field

1. I resolve the expected declared layout and reject spread, missing, duplicate, unknown or wrong-nominal fields. I keep full-source checking before publication.
2. I visit constructor field expressions in written source order. A named child moves exactly once. A literal child recursively follows this protocol with a strictly earlier expected layout. A direct factory uses the existing exact result-layout/call proof; it does not authorize a new signature.
3. I store each completed child immediately in a fresh hidden exact owner slot. Its temporary stays live while later expressions allocate, call or fail. Scalar fields retain their existing staged slots. No child value is evaluated a second time to arrange field order.
4. After all fields succeed, I move hidden owner slots and load scalar slots in declaration order, then OWN_PACK. I clear each moved slot's live state exactly once. The resulting shell has one owner; source names and hidden temporaries do not retain copies.
5. I enforce existing depth/local/function bounds before unsafe indexing or unchecked emission. Constructor recursion is bounded by expected prior layout identity; general expression/call restrictions remain. Exhaustion refuses publication rather than reusing physical slots or silently dropping fields.

I do not emit original names for compiler temporaries. I preserve lexical source bindings, outer-name restoration and all mandatory selected shadows. No AST mutation, source partition or shadow omission is part of this change.

## I preserve failure and transfer lifetimes

Existing VM OWN_PACK allocates before removing field operands. Existing native OWN_PACK allocates before clearing stack carriers; common cleanup drains live locals/stack. I audit these actual paths with nested temporaries, including a successful child followed by a failed sibling or parent allocation. A transferred child is rooted in its hidden local or operand stack until the new parent owns it. On a terminal failure, exactly the active root owns each descendant; cleanup neither leaks nor releases it twice. A helper failure retains the first status while caller temporary roots are drained.

Compile-time refusal may leave private emitter state incomplete; no module/output is published and normal compiler teardown frees that state. I do not promise rollback of private emission state or reuse it after an error.

I retain whole-owner moves, exact consuming arguments, destructive unpack and unavailable-after-move rules. Pattern field order can differ from declaration order; each declared field still reaches its exact destination once. I do not introduce implicit consumption of an unresolved source owner.

## I qualify the bounded source change

Before execution I send both producer changes and prepared fixtures for independent review. Fresh source/bootstrap tools then qualify C NanoVirt, Cseed/Stage1/Stage2-built emitters, canonical Stage1/Stage2 publication, complete selected shadows, VM execution and strict GCC/Clang sanitized native execution.

My positive controls include existing named children, inline scalar-leaf and deeper literal children, an existing scalar-leaf factory child, mixed named/literal/factory siblings, empty owned leaves and reordered named fields. Distinct scalar observations establish source-order once evaluation independently from declaration-order packing. Every owner is consumed, including children unpacked in reversed pattern order. Main and selected shadow calls exercise the same source without dropping PREFIX.

My refusal controls retain wrong nominal but same-shaped children, duplicate/missing fields, moved or held named owners, unconsumed siblings and existing unsupported managed/result boundaries. I verify previous output bytes remain unchanged. Bounds controls distinguish admitted depth/local budgets from conservative refusal without executing refused modules.

I qualify corrected ordinary modules under deterministic positive-capacity allocation failure at child, sibling and parent construction, with retained caller roots and exact cleanup/first-error assertions. Existing runtime allocation harnesses can exercise emitted instruction shapes; they do not replace actual source execution. I retain existing temporary-owner, transitive-wrapper, graph/result and source-borrow regression gates, including allocation ceilings. I do not replay any historical failed artifact.

I freeze source, fixtures and actual compiler/runtime identities around each gate, preserve every first terminal outcome, and qualify integration separately where source composition changes. The original two Bundle methods remain explicit managed-field dependents. This task closes only after reviewed source implementation, paired evidence and canonical integration.

## I stop private construction after allocation refusal

Before production I inspect both slot allocators: C checks count256 before touching its fixed tables; Nano checks length>=256 before appending parallel arrays. I retain these physical limits rather than relying on module verification. Static inspection also finds that Nano nb_owner leaves its field loop after an error but still enters declaration-order packing, where an unfinished slot can be -1. Under this same construction task I return immediately after recursive emission/slot failure and before packing; I perform no pre-fix fault execution and promise no emitter rollback. C likewise stops before using a failed staging slot.

## I prepare the first source checkpoint

My paired delta reuses each producer's existing owner-expression routine for a strictly earlier expected child layout, then stages its result immediately. Nano's reused identifier path explicitly retains the old child tag8 requirement. Both routines return before using a failed staging slot; Nano also returns before its final packing loop after any field error. Both physical allocators retain their preallocation256 checks. I add mandatory nested-construction and full-local-budget shadows without modifying source result guards.

My prepared separate Python suite selects only three new methods, retaining the existing full source suite independently. The ordinary fixture combines a named owner, an inline two-level literal, an existing leaf factory, reversed written fields and reversed destructive patterns. Exact outputNFL checks source order and once evaluation; numeric37/94 checks declaration identity. Another fixture transfers an empty inline child. Refusals retain exact prior output and include129 inline leaf children, whose258 minimum staging slots exceed256 before parent publication. A generated native allocation harness counts all owner allocations, fails each positive-capacity allocation individually, requires unchanged result output and zero live shells, then requires successful recovery with the same code. Existing runtime allocation gates remain separate VM cleanup evidence; this prepared source fixture adds no new VM fault injection claim. No prepared fixture has executed.
