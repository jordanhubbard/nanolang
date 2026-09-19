# My multiple consuming-parameter runtime contract

I record `task_6d8f539329294b24ad2746bec6a22862` before implementation,
after the single-owner runtime and source gates in PR709 and PR714. This
contract does not admit new source syntax or expand my full ownership claim.

## My admitted shape

I retain standalone entry0 and helper1, zero entry arguments, and one
INT/BOOL/U8 result per function. My helper takes 1..8 parameters, using my
existing `NVM_AFFINE_MAX_PARAMETERS` bound. Every parameter has mode zero
and is either INT/BOOL/U8 or an exact complete resource STRUCT; at least one
is a resource. Each position retains its own tag and nominal layout in my
existing ownership metadata. I need no wire-format change.

I keep CALL_REF separate. I do not admit mixed borrowed/value signatures,
owned results, deeper graphs, callbacks, imports or additional functions.
My existing finite nested record layout and explicit consumption rules remain.

## My ordered transfer

I inspect every argument in source-order stack positions before changing my
affine stack. An owned argument must be a unique non-observation value with
the exact declared layout. A scalar argument must have its exact declared
tag and carry no owner. An already materialized scalar observation is a
value; a resource observation cannot become an owner. Existing OWN_MOVE_LOCAL
checks retain moved-owner and outstanding-hold refusals. Repeating one owner
cannot supply two arguments; equal layouts do not equate distinct declarations.

I initialize helper local p from argument p, preserving evaluation order and
once-only transfer. Source producers will later enforce this order explicitly;
this runtime prerequisite checks and executes the resulting ordered stack.

I validate the complete runtime contract and all values, then reserve the VM
frame before publishing the helper activation or advancing its generation.
Before activation, prepared arguments still belong to caller stack cleanup.
After activation, helper locals own them. Preflight is atomic at the frame
boundary; it does not undo argument evaluation or restore moved source locals.

My native helper moves and clears every incoming carrier before writing the
scalar result, whose destination may overlap the first argument carrier. I
preserve unrelated caller roots and fresh local-reference provenance. Every
normal helper exit explicitly consumes its resource parameters and locals;
terminal errors drain actual owner roots and clear both reference contexts.

## My implementation boundary

My current single-parameter assumptions are in
`nvm_affine_owned_parameter_type`, the affine CALL transfer, owned-module
signature validation, VM CALL preflight and `nvm2c_owned.h` stack/emission
logic. I replace those assumptions with a shared bounded positional contract;
I preserve the existing single-parameter API where callers still require it.
Ordinary non-owned CALL and borrowed CALL_REF behavior remain unchanged.

## My acceptance

I first check valid two- and eight-parameter modules with interleaved scalar
and owned arguments, distinct nominal layouts, nested owners and order-sensitive
results. I compare VM and native execution of the same verified modules.
Repeated calls and helper-local observations check generation and slot identity.
Entry/helper assertion failures check cleanup and successful subsequent use.

I retain static refusal controls for moved or borrowed owners, duplicated
transfers, wrong nominal/scalar types, mixed modes and unsupported graphs.
Rejected modules are not executed. I qualify heap allocation faults separately
from frame reservation and contract allocation failures, with exact measured
coverage. GCC/Clang sanitizer runs and existing affine, reference, consuming-call
and VM lifecycle gates must pass before source admission proceeds.
