# My checked replacement allocation contract

I record `task_6b3d7306179f4a7786adbd29b9acbe6c` before changing VM replacement.
Static inspection found signed output-size multiplication and unchecked final
string publication in STR_REPLACE. I add a checked unsigned length helper,
scratch representability and result checks before managed replacement admission.

My helper accepts stored uint32 lengths and an occurrence count. For a
nonempty needle it requires count <= source_length / needle_length, subtracts
only that bounded product, then requires the added replacement bytes to fit
UINT32_MAX before multiplying. Empty needle permits only count0 and keeps the
source length. Invalid arguments leave the output unchanged. The caller checks
space for a scratch terminator before allocation; existing string allocation
checks its own header/payload bound. I do not change wire lengths.

I preserve exact stored-byte, left-to-right nonoverlapping replacement, including
embedded NUL, empty replacement and empty-needle identity. Three popped owners
are released once on type, sizing and allocation failure or success. Final and
empty-needle results are checked before publication. VM interning may reuse an
equal object; no physical identity or allocation-event guarantee is added.

I verify the size helper directly using finite scalar counts and boundaries,
without constructing oversized strings. Fresh ordinary byte fixtures exercise
shrinking/growing/no-match/empty-needle results, scratch-boundary lengths,
aliases and deterministic final allocation failure followed by recovery.
Focused sanitizers and full VM regression acceptance remain required. I do not
run a pre-fix failure or historical artifact. Managed replacement is separate;
STR_SPLIT still needs the aggregate runtime, and full51da/platform7ba/eval791a
remain open.
