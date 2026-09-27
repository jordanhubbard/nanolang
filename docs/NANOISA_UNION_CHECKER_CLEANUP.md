# My union checker cleanup boundary

I retain the 85a ordinary passes and both sanitizer failures. Linux GCC reports
1,786 bytes in 136 allocations after 11.542 seconds; Darwin Homebrew reports the
same allocation total after 8.711 seconds. These are leak reports, not a claim
that the full bootstrap or original 18+8 source/native corpus passed.

I correct ownership without changing accepted types or constructor evaluation:

- My module enum collector allocates variant_values twice. I retain the first
  checked allocation and fill it once.
- My parameter parser separately owns its spelling and parse_type auxiliary
  name. Before replacing the auxiliary name with the retained spelling, I free
  only the former; complete TypeInfo owns its independent copy.
- My module checker creates a record placeholder, then env_define_var copies
  the graph. I discard the caller-owned placeholder after publication, keeping
  the symbol's independent copy. Borrow parameters retain their established
  behavior; I do not change the binding API into a transfer API.
- My inferred binding metadata can replace a copied struct_type_name. I free
  the old owned string before replacing it, matching existing explicit routes.
- My failed parse_block closing-brace path frees the statement vector but drops
  its already owned children. I destroy those children before the vector.

I audit paired root/module and parser callers before applying these changes.
The existing 72 parsed scalar combinations, nested declaration controls and
three malformed constructors remain required. I add direct structural checks
where needed; the actual sanitizer run establishes cleanup across real APIs.
No known failing revision is replayed. New source is reviewed before execution.

My task is task_ebfdc333d6a8495b8c903ab8142c88b6. First reports remain under
`tuple-85a-linux-focused` and persistent Puck `tuple-85a-focused`; I copy the
remote reports and immutable artifacts locally before any further work.

My paired audit finds the root enum collector already allocates once and is
unchanged. Both root and module parameter metadata writers can replace an
inherited owned name, so both release that prior string. Only the module
collector constructs a record placeholder; the root collector uses VOID and
needs no graph discard. I add two same-named enum parameters to check retained
metadata and copied placeholder storage through real module registration.

My corrected09e ordinary run passes; its sanitizer reports one remaining
120-byte spread AST allocation. The actual anonymous-literal parser is the
only production writer: parse_expression creates its separate owned child,
then publication transfers that child into spread_source. I destroy it with
its literal, or before returning from failed closing-brace parsing. There is
no borrowed registration or separate owner to retain. I add real parsed spread
controls and retain the original synthetic refusal/unchanged-node assertion.
