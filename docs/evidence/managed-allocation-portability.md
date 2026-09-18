# My managed allocation portability qualification

I retain exact source pins and unchanged source/tool hashes for every terminal
run in [the evidence directory](managed-allocation-portability/).

At494f21a3993fbfbb10d538e3c6080893836b1a6c, all71 Linux methods passed in
106.553 seconds. Darwin built in5.180 seconds and stopped after17 passing
methods: the eighteenth attempted method still used GNU malloc wrapping.
I filed0b732 before migrating that missed binary64-format wrapper interface.

Atf73601e539c7def5f93571eb4f04c78634c7f566, all71 Linux methods passed in
102.728 seconds. Darwin built in3.708 seconds, passed18 methods including the
corrected allocation-recovery control, then failed the nineteenth method at
negative-NaN reference index28. Its suite ended in51.832 seconds. Neither run
timed out; before/after source and tool hashes match. I independently verified
all copied Darwin report bytes against remote SHA256 hashes.

My fresh six-value native diagnostic shows Linux libc prints negative quiet
and signaling NaNs as `-nan`, while Darwin libc prints both as `nan`, including
its hexadecimal literal formatter. The managed core retains the sign and thus
cannot equal both native reference policies. The failing harness returns before
cleanup, producing the retained sanitizer leak report; I have not demonstrated
an independent runtime leak. I track the formatting-contract discrepancy as
e92a45b66a104e9ba3854cd5f994df8b, without changing expected bytes or skipping NaNs.

I retain all original allocation budgets and assertions. An AST comparison
confirms the additional binary64-format harness bodies are unchanged. My shared
helper redirects only module malloc symbols before the existing sanitizer pass;
all original generated LLVM remains unchanged. There are no remaining GNU malloc
wrap call sites in tests or scripts at this pin.

This PR remains a draft: full Darwin managed qualification is failed, and52
methods after the stopping failure have not run there. Linux qualification is
not a Darwin or full-release result. Product PR522 remains held.
