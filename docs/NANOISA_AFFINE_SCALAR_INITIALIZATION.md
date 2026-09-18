# My definite scalar initialization at owned joins

MAC `task_714b48782ed04de19976c790820227b4`.

I add a definite-initialization meet only for declared mode-zero scalar locals
(int, bool, u8, float) in owned bytecode analysis. Owners, non-scalar locals,
declaration identity, stack provenance, reference origins/holds and regions
retain exact equality. A local is definitely initialized at a join only when
every incoming path initializes it; LOAD_LOCAL remains checked. I introduce no
dummy stores, runtime changes, wire changes or source admission.

My worklist rechecks successors after initialization facts decrease. A
deduplicated bounded queue and an explicit instruction-count times
(local-count plus one) visit ceiling ensure bounded convergence; limits and
allocation failures conservatively refuse. Existing exact-state equality
remains available unchanged. Tests cover predecessor orders, branch-local
unused scalars, both-arm initialization, zero-iteration loops, loop-carried
initialized locals, reprocessed descendants, exact authority mismatch
refusals, allocation failure and visit bounds. Existing affine state/bytecode,
owned runtime/reference/assertion gates remain required. Dependent source
lexical declarations and path-sensitive fallthrough/early-exit work stay
separate.
