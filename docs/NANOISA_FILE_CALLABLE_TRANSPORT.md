# I carry callable arguments and results

Under #989 I extend my private indirect File profile beyond function-local
callable values. I admit mode-zero `TAG_FUNCTION` declarations with `NO_INDEX`
for parameters and results as well as local slots. Old acyclic/cyclic declaration
entrypoints retain their existing profile. My hosted entry remains a zero-argument
scalar result; a callable cannot escape as a host-visible entry result.

I retain one target bitset per formal and one per function result. Every bit
names an original function in the same immutable module. `FUNCREF` introduces
bits; actual arguments grow the selected callees' formal summaries, and checked
returns grow result summaries. Direct and indirect calls propagate those facts.
An indirect call contributes every compatible candidate, not just its first
candidate. Its result joins their result summaries.

I iterate the module in original function order. Within each function I retain
the existing control-flow fixed point, including both branch successors and
loop backedges. A temporarily empty callable target set stops that transfer
until another module pass supplies facts. Empty means no established target;
it never means every function. At convergence I recheck each function without
that deferral and publish only fully resolved indirect sites. Unreachable or
unresolved indirect sites still refuse. Growing summaries and graph edges are
monotone; the existing total transfer-visit limit bounds all passes together.

My storage budget includes the exact formal/result summary arrays. I keep one
function's temporary local/stack work states. I do not retain every function's
full instruction-state matrix. Allocation failure still publishes no report.
The completed direct/indirect graph must remain acyclic before ownership checking.
Every candidate must agree on complete parameter/result declarations, including
File/OpenResult identities and modes. I do not admit captured or imported
callables, callable aggregates, or indirect borrowed formals through this change.

The existing private carrier copies and moves its full invocation-plan/function
identity through argument staging, child locals and returned values. Invocation
still checks that identity and the prepared site candidate set before moving
owners. VM and native frame layouts use the same checks. Generated native calls
still select real generated functions; there is no VM or FFI fallback.

My new corpus returns a function from a forward factory, passes it through a
callable identity function, and invokes it through a higher-order helper. It
covers direct and indirect wrapper calls, both candidate bodies, catalog
permutation, scalar/File/OpenResult arguments and results, every lower fuel
limit, denial and cleanup failure. It checks the higher-order site's complete
candidate set, refuses an unsourced callable formal and a higher-order recursive
graph, and sweeps preparation failures using a higher-order owned fixture.

This extends the private query/runtime/dispatch profile. Paired C/Nano source,
full selected shadows, indirect borrowed formals, public and installed consumers,
Linux qualification and the other 5.1 requirements remain open.
