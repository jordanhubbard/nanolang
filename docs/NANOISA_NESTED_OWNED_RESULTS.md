# I transfer exact nested owned record results

I execute task_ebdc86bd38af42cc844700f91fc90c95 under the open product-affine e8d860 and affine28f2 requirements. This contract precedes production and execution. My isolated branch starts from reviewed PR801 head80d1a42a; I preserve that worktree, tools and evidence. I integrate its canonical merge before freezing executable acceptance. I do not execute the retained refused factories or any historical failed compiler artifact.

## I distinguish the missing contract

PR738/taska279 explicitly qualified scalar-leaf owned results. Its phrase nested returns described nested calls, not nested fields. PR801 exposed the distinction: ordinary Connection locals/arguments pass, but an Outer{Inner{Leaf}} factory result meets deliberate guards in both producers and the shared descriptor query. Local construction cannot discharge this result obligation. I retain the original refusal and the later passing named-child local fixture separately.

My new scope is one exact owned STRUCT result whose complete declaration-ordered tree contains only INT, BOOL, U8 leaves and complete owned STRUCT children. Source remains its existing INT/BOOL subset. I preserve exact nominal indexes even for same-shaped declarations. I admit at most32 record levels and256 fields per reachable record in this result profile, preserving the existing source/runtime bounds elsewhere. Existing scalar-leaf results keep their old admission and allocation behavior. Empty explicitly owned leaf records retain acceptance. Ordinary nonowner children, FLOAT/STRING/collection/union/generic/unknown result fields remain refused in this child. I do not change their independent parameter/local or managed profiles.

Entry0 retains its existing scalar result ABI. Helper results remain zero for VOID or one scalar/owned value. I retain the acyclic graph's eight functions/frames and zero through eight exact mode-zero parameters. CALL_REF, borrowed results, recursion, external/indirect calls, reference/value graph mixing and implicit owner disposal do not expand.

## I trace admission before changing it

1. `ownership_contracts.c` validates exact descriptors, complete resource flags, prior-only resource-bearing layout edges and transitive scalar-tree facts. Its scalar-tree predicate includes FLOAT, so those flags alone are insufficient for my narrower result profile.
2. `nvm_affine_state_create` loads validated immutable layout/ownership facts. `nvm_affine_value_result` currently permits only scalar-leaf INT/BOOL/U8 results and leaves outputs unchanged on refusal. I extend this query, not the generic scalar/type predicates.
3. `owned_runtime_validate` in verifier.c consumes the query for each function and retains the entire module's opcode/layout/local/signature/graph restrictions. Inner query success alone never authorizes execution. Entry restrictions remain unchanged.
4. `affine_bytecode.c` validates each callee completely before consuming exact positional arguments and pushing one owned result with its exact layout. RET already requires exactly one non-observation owner, exact result identity, no active region and no remaining local-owned obligations. I preserve those rules.
5. VM owned RET calls the query while the returned value is still a stack root, validates non-null record/root layout/field count, reserves publication capacity, clears the finished reference activation, then removes callee roots and publishes the same owner. Native emission uses the query and already carries a whole `nown_value` through a private pending carrier, clears incoming argument carriers before possible overlapping result publication, drains locals/stack, publishes only on status0 and releases any remaining pending owner on failure.
6. C `borrow_codegen.inc` and Nano `nanoisa_borrows.nano` independently impose a scalar-leaf helper-result guard. Only after runtime qualification may these admit an exact already-collected owned layout. Their collector, named-child construction, argument moves, return moves, cleanup and exact result descriptors remain authoritative.

No ownership version, opcode, wire layout, nominal-ID policy or native calling convention changes are proposed. I require exact binary/text result descriptors and layout bytes after normal roundtrips, not inferred compatibility from the unchanged schema.

## I bound result validation without changing transport

I preserve scalar/VOID and existing scalar-leaf fast paths without adding allocations. For a genuinely nested result I validate the reachable result closure using one layout-indexed work/depth table and descending prior-index traversal. Each reachable layout is expanded at most once; repeated DAG edges update its maximum depth before expansion. I check prior indexes before indexing, complete/resource flags and STRUCT kind, exact leaf tags with no nested index, child ownership and field/depth bounds. Allocation or validation failure returns false without writing either output. I publish root type and root field count only after successful validation and free scratch storage on every path.

I do not recursively revisit declaration DAGs or traverse unrelated layouts merely to authorize the result. Full module validation still runs independently. I retain existing status mappings for an unavailable query; I report injected failure sites/statuses accurately rather than calling every refusal an allocation error. A changed error taxonomy would require separate review.

I audit existing VM/native child lifetimes with ordinary nested returns before deciding whether any transport correction is necessary. An exact root query is sufficient only with existing verified constructors/moves and authority analysis; I do not accept arbitrary externally fabricated aggregate trees. If the current transport cannot retain descendants or maintain failure atomicity, I record and review that precise correction before further execution.

## I qualify runtime before source admission

My first production checkpoint comprises the checked descriptor query/header and normative scope, with source guards still intact. Runtime fixtures use fresh ordinary assembled modules, not modifications of preserved refused artifacts. I cover:

- A named-child two-level factory, identity/relay, chained forwarding, conditional returns, distinct same-shaped nominal roots and an explicitly owned empty leaf. I destroy every returned child and observe exact scalar values.
- Repeated sibling calls, unrelated caller-owned roots before/after a returned allocation, scalar/VOID neighbors and separate normal/error cleanup. All four public VM APIs and strict GCC/Clang native execution must agree.
- Deterministic owner allocation failures at leaf/parent/result preparation, available query/preflight failure injection, unchanged caller holds and zero remaining owned allocations after termination. I name actual injectable sites; I do not claim exhaustive allocation coverage from a bounded budget sweep. Sanitizers cover native descendant cleanup.
- Static checked refusals for wrong nominal result, live unconsumed sibling, missing/extra results, ignored/duplicated returned owner, reference/observation escape, invalid result shape and beyond-bound declaration shapes. These are verification/query controls; I never execute refused modules.
- Unchanged scalar-leaf result allocation ceilings and existing owned graph/result/consuming/assertion/authority gates, including the current owned binary64 operand-stack intersection without widening FLOAT fields or signatures.

A second reviewed production checkpoint removes only the two source leaf-result restrictions for exact collected owned layouts. Its new ordinary fixture constructs named Leaf then Inner then Outer in a factory, returns that owner, relays it and consumes it in another helper. It retains source-order scalar observations, exact metadata, mandatory shadows, branch/loop ownership joins and use-after-transfer/output-preserving refusal controls. No inline nested child literal is needed or admitted. I statically walk all fixture forms against both producers before execution.

I require a fresh source bootstrap and C NanoVirt, Cseed/Stage1/Stage2-built raw emitters, canonical Stage1/Stage2 and selected shadows through exact metadata, VM and sanitized native observations. Frozen source/harness/tool inventories bracket each gate. First terminal failures remain preserved and require reviewed corrections. Prior PR801 success remains at its own pins, not attributed to these new tools.

I close only this nested-owned-result scope after canonical integration and complete paired evidence. Broader affine/product acceptance, managed/generic/union results, inline nested constructors and unrelated source/profile requirements remain separate. No checked-refusal boundary is silently recast as full-language acceptance.

## I freeze the runtime-only fixture scope

My prepared harness uses six fresh ordinary modules: both factory branch arms, a second same-shaped exact nominal root, an empty owned leaf, failure with a pending returned root and failure after caller receipt. Each normal entry holds an unrelated borrowed scalar owner across factory/relay/consume calls, consumes two returned nested trees and observes42. I exercise each of four VM APIs twice and repeat native entry plus deterministic native owner-allocation failures. Separate VM heap-allocation sweeps retain live-object/root checks after every terminal result. Linked VM objects use the ordinary build; generated native harnesses use strict GCC/Clang ASan/UBSan/leak checks. I do not imply the ordinary linked VM objects are fully sanitizer-instrumented.

My query-only DAG builder preserves valid metadata and is never executed. Shared subgraphs have a direct short edge plus a longer chain: depth32 passes and depth33 refuses after maximum-depth propagation. Width256 passes and257 refuses; FLOAT leaves and nonowned child declarations refuse. An instrumented calloc override affects affine_state only and is enabled after state creation: the one nested scratch failure leaves both caller outputs unchanged. Scalar/VOID/leaf queries succeed with zero allocation calls under the same override. Existing descriptor tests now qualify the previously refused exact nested layout and its roundtrip; the old runtime refusal6 still tests its independent exact caller/result mismatch, explicitly documented.

Both source result guards remain byte-identical to PR801. No source bootstrap or source factory acceptance is claimed at this runtime checkpoint. Fixture/output directories are retained explicitly through the test environment, including first failures; neither refused metadata controls nor historical source artifacts execute.
