# Full recovered Darwin source-snapshot corpus

I run the unchanged complete corpus at clean `fdca4ee9b53b816232074b0d539893168715baf2`.
The command exits zero after 5,324.065 seconds; unittest reports 128 methods,
19 platform skips and no failures in 5,317.818 seconds. Source, HEAD and the
probe hash remain unchanged. The manifest retains the original PATH, compiler,
free-space precondition, command, terminal log hash and before/after inventories.
Earlier failed terminals remain retained separately. This passes the Darwin
corpus; it does not qualify Linux or a later final release candidate.
