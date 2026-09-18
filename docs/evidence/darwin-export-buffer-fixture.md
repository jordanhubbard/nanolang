# My Darwin export-buffer fixture qualification

I qualify the four-line test-only correction `b59c6aeb` against pre-code contract
`fb1b385b`, from canonical `f761af4c`. I undefine the active SDK macro immediately
before my fault interception, after my wrapper's ordinary platform formatter
call. I preserve every success, first/second format failure, initial/growth
allocation failure and overflow assertion. Production is unchanged.

On a fresh detached Darwin arm64 checkout, my strict target passes in 2.950s
with Homebrew Clang23.1.1. My O1 ASan/UBSan fixture compiles and passes in 0.623s
with LeakSanitizer enabled. On Linux arm64, the strict target and GCC13.3/Clang18.1
O1 ASan/UBSan/leak fixture executions pass. All commands retain `-Werror`;
I do not suppress macro diagnostics or disable fortification globally. Common
objects are ordinary builds; these sanitizer claims concern the included
exporter and fixture. My initial ordinary Linux target also passes.

I retain [the exact commands, logs and hashes](darwin-export-buffer-fixture/reports.sha256.json).
Both platform inventories contain 1,594 source/test/build inputs unchanged
before/after and matching this final tree. The actual selected compiler binaries
(two Linux, one Darwin) retain their before/after hashes; I hash the selected
Homebrew compiler rather than Apple's `cc` dispatcher. My source and assertions
remain frozen throughout qualification.

The original SDK macro diagnostic is retained verbatim from the Darwin owner's
rollout line157221, under `original-diagnostic.txt`. Its compiler error identifies
`secure/_stdio.h:124` and the fixture's original line33. I read that historical
log without rerunning its source or executable. The original report remains at
`docs/evidence/shared-match-short-paths.md` in `d9917dee`.

This supplies the bounded acceptance for task_43dea95525b24546b4b3e259a3148205,
pending canonical merge. The adjacent reference-evaluator leak, shared-match
production, full NanoCore acceptance and release remain separate. I do not claim
a compiler bootstrap or full product gate from this fixture check.
