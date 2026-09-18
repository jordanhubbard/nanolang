# My checked empty-record allocation boundary

I execute task_66983a5c5f9a4c869fc726824cf6a773 as a separate prerequisite
before private managed-record traversal. STRUCT_NEW allocates an empty record
with its existing definition index. I must check a failed allocation before
publishing TAG_STRUCT, matching the existing STRUCT_LITERAL/AGG_PACK contract.

I add only the missing NULL check and return VM_ERR_MEMORY through trap_error.
No operands are consumed by this instruction; no record is published on failure.
I preserve normal definition identity, zero-field semantics and all existing
execution admission. Managed nominal execution remains refused.

After the defensive check exists, I qualify fresh normal module invocation,
returned-record release, a bounded allocator refusal, unchanged heap counters,
no published result and successful later invocation. I reuse the existing
heap-local fault injector and run the full VM and sanitizer allocation gates.
I do not execute a pre-fix fault case or infer a historical incident cause.
