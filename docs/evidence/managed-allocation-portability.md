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

After integrating PR745 at18255183, Linux71 passed in103.620 seconds.
Darwin passed35 methods and then stopped at a sibling primitive-format fixture
still reconstructing input from host `%a`; the expected negative sign was lost.
I filed24e26c8 before switching both primitive conversions to exact integer bits,
retaining all64 values, expected text, fallback and cleanup assertions. This third
failed report remains in darwin-format-integrated; I do not relabel it a pass.

At corrected79485be25a55f1cb92a1bc7138c882412f0be9f1, the full71 methods pass
on Linux in103.599 seconds and Darwin in182.814 seconds, with no skipped methods
or timeouts. Darwin rebuilt fresh tools in3.864 seconds. Both before/after source
and tool manifests are identical; copied Darwin artifacts match independently
read remote SHA256 values. The original inner timeouts, ASan/UBSan/LeakSanitizer,
Wasm memory ceilings and allocation budgets remain unchanged.

I qualify the shared malloc-control migration and its final missing binary64
caller, including private record/native/Node/Wasmtime paths selected by the
unchanged corpus. Signed nonfinite formatting uses the independently reviewed
merged PR745 policy. Parent e92 remains open for legacy/source formatting routes.
Product PR522 and full release acceptance remain held.
