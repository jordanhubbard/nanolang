# My affine example restoration contract

I track `task_c4351c720aee424ea9b90187e51a08f2`. The frozen product
`e9a5f55f` quick gate retains 243/244 eligible example compilations and stops
while compiling selected shadows for
`examples/language/nl_affine_resource_demo.nano`. I preserve
`/tmp/nanolang-product-quick-e9a5f55f/test-quick.log`; I do not replay the frozen
artifact, exclude the example, change its source or omit its shadows.

## My static finding

My example defines `open_file(string)->FileHandle`,
`close_file(FileHandle)->void`, and `main()->int`. The opener prints its path
and returns a newly constructed resource. The closer destructures that owner,
asserts its field and prints a literal. The main shadow calls main, whose body
calls both functions. The required shadow module therefore contains a
synthetic entry plus three ordinary functions and reaches three active frames.

Both source routes choose my specialized ownership producer because close_file
has a resource parameter. That producer currently permits exactly main and
one helper. My runtime also restricts owned execution to entry0/helper1,
scalar results and a fixed pair of reference activations. Its supported owned
instructions and native value carrier do not yet admit string parameters or
PRINT. These are independent limits; changing the source helper count alone
cannot restore the example.

I do not send this program through ordinary lowering without ownership
metadata. Such routing would remove the authority checks rather than satisfy
them. Multiple consuming source parameters under taskd540 remain separately
tracked and paused; they do not resolve this call topology.

## My prerequisite order

1. I qualify a bounded acyclic value-only owned call graph, including a
   zero-argument intermediate function and three active frames. Every frame
   retains its own local reference context and fresh generation. This is
   `task_4ce5cfc5b8034949852255d7307c9f91`, specified below.
2. I qualify exact owned results and void results in that graph. A returned
   resource is one transferred owner with exact nominal layout; every other
   local owner is explicitly consumed. Void calls produce no operand. Native
   status remains separate from value transport. I require return, error and
   caller-result cleanup before any source admission.
3. I qualify the string value and PRINT behavior needed by the unchanged
   example. I first audit VM print semantics and string lifetime. I do not
   treat STRING as an untracked integer or infer safe storage from a tag alone.
   The contract must name admitted producers, call transport, output bytes and
   terminal cleanup before implementation. Resource layouts keep their current
   scalar-leaf bounds unless a separate prerequisite expands them.
4. I connect both source producers to those qualified contracts. I preserve
   exact function identities, positional metadata, source-order evaluation,
   explicit resource construction/consumption and all selected shadows,
   including shadow main calling main. No hidden inlining or shadow omission
   substitutes for the accepted graph contract.
5. I compile and execute the unchanged example through C-seed and both
   selfhost stages, mandatory shadow supervision, verification, VM and native
   targets. I compare ordinary output and results, retain semantic refusals and
   prior-output preservation, then return the repaired example to the full
   installed-product quick gate. This document is not evidence that those
   later gates have passed.

## My first runtime prerequisite

I admit at most eight functions and eight active frames in an otherwise
standalone value-only owned module. Entry0 has no arguments and is never a call target. Other functions
have 0..8 mode-zero parameters, each INT/BOOL/U8 or an exact complete resource
STRUCT. Every function still returns exactly one INT/BOOL/U8. The module must
retain complete ownership/layout metadata and an actual explicit owner
transfer. A zero-argument intermediate function models a selected shadow
calling main; it does not change ordinary non-owned CALL.

I validate every direct call target and the complete graph before execution.
The graph is acyclic, including code outside the entry's reachable subset; I
bound graph analysis and retain exact arity/tag/layout checks. I analyze each
function with its declared mode-zero initial state. Ownership, stack, region
and reaching-edge facts remain exact except the already qualified scalar
initialization meet. Every normal exit still explicitly consumes local owners.

At a call I inspect the whole ordered argument list and reserve all required
frame/context storage before activation or transfer. My bounded reference
contexts are embedded in the VM state; no context allocation occurs after
transfer. Prepared owners remain
caller cleanup roots until activation. Each frame gets a fresh reference
generation; equal local/ref slot numbers in callers, children and siblings do
not identify the same owner. A caller's local-only hold may remain live while
a disjoint owner is passed by value, and must remain valid after the child
returns. No reference is passed across this new boundary.

I clear only the returning frame's context on normal return, and every active
context during terminal unwind through all four VM APIs. Native functions
must transfer/clear argument carriers before any overlapping scalar output,
propagate status, and clean all actual roots on failure. I retain generation
exhaustion and allocation preflight checks. Source-name metadata does not
supply authority.

My existing two-function borrowed-only CALL_REF profile remains separate and
unchanged. I do not admit mixed reference/value signatures, recursive or
indirect calls, imports/callbacks, owner or void returns, strings/PRINT, or
source programs in this first prerequisite.

My acceptance includes a four-function DAG with three active frames,
zero-argument wrapper, sibling and repeated calls, multiple exact nominal
owners and local reference observations at identical slot indices. I compare
all VM APIs and sanitized native execution of valid modules. I check graph,
authority and signature refusals without executing rejected modules; I qualify
preflight and terminal cleanup with corrected ordinary fixtures. Existing
single/multiple consuming and borrowed call gates remain required. The
first production checkpoint follows the reviewed contract; its new acceptance
gates remain pending.


## My bounded completion

I retain the preimplementation and first-failure statements above as history.
PR761 qualifies the unchanged example through C-seed/both selfhost stages,
original shadows, exact VM/native output and refusals. The later frozen cd72
installed-product run compiles all244 eligible examples, including this case,
then fails the next independent documentation gate. This satisfies my original
step5 example boundary; c435 is reconciled complete. My
[closure evidence](evidence/source-ledger-reconciliation.md) retains the exact
product evidence commit. Full product, ownership and release remain open.
