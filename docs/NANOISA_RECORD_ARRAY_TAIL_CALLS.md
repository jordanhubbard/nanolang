# My bounded ordinary record-array tail transfer

I implement the reviewed bounded contract from `562fb5ebb` on the combined `67159d826` base. My concrete source gap was a generic forwarder that emitted TAIL_CALL: the same record identities already published verified v3 authority through ordinary CALL/RET. My [focused receipt](evidence/record-array-tail-20261018/README.md) now qualifies this acyclic tail slice; fresh compiler generations and exact Linux/Darwin release gates remain open.

## Boundary

I begin with direct, same-module tail edges that do not participate in a cycle of the combined CALL/TAIL_CALL graph. I do not newly reject existing CALL-only recursion. A tail edge whose target can reach its caller, including a self edge, remains UNRESOLVED until a separate recursive-tail qualification. This closes the demonstrated forwarding case without claiming constant native stack for mutually recursive C calls. I charge the bounded graph storage/traversal to the existing selected-profile budget; a limit remains LIMIT, allocation failure MEMORY, and malformed targets/signatures INVALID.

I retain existing complete ordinary declarations, exact flat field element bindings, closed-module restrictions and all current array/record lifetime rules. I do not use generic advisory strings, source spelling, opaque proof tokens or an opcode whitelist alone as authority. Other array-query modes keep their old instruction boundary.

## Proof transfer

My structural decoder already checks direct target ranges and matching caller/callee result signatures. I additionally require the reachable tail operand stack to contain exactly the callee arguments, without discarded extra operands. Typed argument/origin checks must be identical to a proved ordinary call; coarse STRUCT tags do not invent nominal indices where the metadata retains NO_INDEX.

I extend the global-write closure over both ordinary and tail call edges. At a tail instruction I seed the callee with the actual argument origins and current global state, wait for its proved return, incorporate its global exits/invalidation, and merge its result into the caller's result. I mark the caller returned using that updated global state. I never enqueue the instruction after the tail call. Existing field-origin propagation and constraints must cover records, arrays, shared aliases and global snapshots through this terminal transfer.

I add an explicit copied tail execution recipe with callee, argument pops, zero local pushes and no intraprocedural successors. Its obligations include staged operands, retain-before-release, exact runtime checks and first-error cleanup. Both the plan publisher and the independent VM recipe/image validator must agree, including all opcode-decision controls. The runtime independently verifies the original module on every admitted route.

## VM and native execution

My public route uses the ordinary scoped handler, while the independent private test route uses its copied recipe preflight. Both check cumulative staging headroom. My public supervisor retains immutable requested-root authorization and separately borrows an active-root identity on its synchronous call stack. Only a committed frame-zero tail replacement updates that identity; every supervisor exit restores the borrowed pointer. This permits assertion resumption after entry/initializer tail chains without authorizing arbitrary host entry into helpers.

My ordinary VM already retains a staged argument vector before dropping the old frame's locals/operands, then replaces that frame. I qualify that real handler through the public profile. Its preflight must check the replacement frame's local/maximum-stack capacity at the existing stack base, not require an additional frame. I check exact operand depth, declared argument/result tags and copied callee identity before any mutation.

I account for cumulative retain headroom when several arguments alias the same heap owner; checking each against UINT32_MAX independently is insufficient. The greater-than-16-argument path must exercise its real temporary allocation and failure cleanup. A failure must leave roots available to the existing unwind path and preserve the first error and prior published result. My private runner preserves initializer results below the entry stack base; my public supervisor keeps its existing scalar-initializer drain before entry. Module, return continuation and activation-floor identity remain unchanged by replacement.

My native lowerer already has a direct tail-call path and a separate self-restart path. This first slice uses the direct acyclic path only. I must qualify canonical record return identity, staged arguments and caller/callee roots through forced collection; unadmitted legacy native execution does not prove those selected canonical-owner paths. I do not alter the public ordinary representation or add a recursive trampoline in this slice.

## Qualification contract

I require actual public verifier, NanoVM, v2 transport, disassembly/reassembly and ASan/UBSan native execution for:

- Direct and multi-hop tail forwarding, entry/initializer tails, void/scalar/array/record results, all five flat element tags and forward record DAGs.
- Shared record/array arguments, mutation observed through aliases, new records returned by the tail callee, globals written in a callee and snapshots/refinements consumed after a caller returns.
- Tail calls nested under ordinary calls, 16 and 17 arguments, replacement frames with differing local/stack sizes, forced collections and constant VM frame count across an acyclic tail chain.
- Exact invalid target/result/argument/extra-stack controls, unproved origins, malformed metadata, self/mutual/mixed-edge tail-cycle refusals, unreachable unsupported instructions and charged graph/flow limits. Dead instructions remain subject to complete structural admission.
- Transient/persistent allocation prefixes for new proof/recipe/graph storage and real VM argument staging, with drained frame/root state, zero tracked owners after destruction, preserved prior results/output and an independent successful invocation after failure. I test cumulative retain overflow before argument/frame mutation.

Only after the public consumers and lifetime/failure gates pass do I change the paired source forwarding control from omission to expected v3 publication. I keep an explicit recursive-tail omission control, the existing foreign/resource/nested-array boundaries and fresh combined compiler/platform gates. This proposal does not close full #1001.
