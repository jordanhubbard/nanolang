# My compiler substring search

My reviewed4369 diagnostic measures78 completed transp_str_index_of calls at
1.998073 seconds wall and1.989835 seconds process CPU. The largest numeric
shadow interval spends1.995489 of2.054069 seconds in10 calls. These are
inclusive diagnostic intervals, not an unchanged-production timing claim.
The run stops at the original ten-second limit; unmarked final work remains
unattributed. I retain the complete graph, original assertions and deadline.

I preserve the internal helper signature and first byte offset contract.
My existing strings and search builtins use NUL-terminated byte strings; I do
not introduce Unicode character indexing or embedded-NUL semantics.

I first return0 for an empty needle, then-1 for a longer or absent needle.
For a present nonempty needle of length m in a haystack of length n, the
shortest prefix containing it lies in[m,n]. Prefix containment is monotone:
once a prefix contains the complete needle, every longer prefix does too.
I bisect this interval with low+(high-low)/2, testing the prefix through
existing str_substring and str_contains. True moves high to the midpoint;
false moves low to midpoint+1. The strictly shrinking interval ends at the
first occurrence's ending offset, so subtracting m yields its starting byte.
Overlap and repeated matches do not change this invariant.

All arithmetic stays within0..n: lengths are existing nonnegative string
lengths, high-low cannot overflow, midpoint+1 is used only when midpoint<high,
and the final subtraction is nonnegative. I do not emit a new runtime helper
or admit a new intrinsic. Both required operations already have exact Nano
checker/emitter mappings and are exercised by the unchanged compiler shadows.
The C/evaluator str_index_of helper cannot be substituted directly yet: its
missing Nano admission is separately task_9ec92cecd44b4b91b3124e0f07617f64.

I preserve every existing helper/global-generation assertion and add a short
reference scan for bounded differential shadows over empty, absent, repeated,
overlapping, late and multibyte byte patterns. Explicit expected offsets stay
independent of the reference. Full original bootstrap runs exercise interpreted
C-seed and actual Stage1/Stage2 native shadows; source/native18+8 remains a
separate required gate. Corrected source review precedes any execution.
