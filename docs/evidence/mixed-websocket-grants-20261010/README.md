# My mixed WebSocket host grant evidence

I extend explicit mixed host grants to WebSocket while retaining my File/TCP-only
legacy constructor and opaque grant ABI. I copy each instance's connection and
lookup authority, deadline ceiling and resolver path. I retain at most 64 paths
of at most 4095 bytes in one allocation; no resolver or network operation occurs
at grant creation. My internal policy query requires the caller's shared gate
and exports the selected instance's policy with a grant-owned path.

I run `make -f Makefile.gnu test-services-flow CC=/opt/homebrew/opt/llvm/bin/clang`.
All five methods pass with ASan/UBSan and leak checking: 32,583 linked and 34,144
instrumented mixed WebSocket checks, 11,123 linked and 12,184 instrumented
File/TCP flow checks, and a C99 consumer using relocated installed headers and
archive. Instrumented WebSocket providers include the new grant implementation.

I check 24 direct/indirect, loop and nominal-permutation combinations over one
WebSocket, File/TCP/WebSocket/WebSocket, and five repeated WebSockets, plus an
unused WebSocket declaration. Exact policies authorize each checked table;
missing/extra instances, wrong catalogs and denied/revoked instances refuse.
I check copied paths after mutating caller storage, distinct instance deadline
and lookup settings, optional paths with denied lookup, 64 maximum-length paths,
zero/maximum deadline limits, malformed revisions/catalogs/paths and File/TCP
policy contamination. Grant allocation failure preserves the output; one
allocation suffices for the 64-instance maximum-path grant. BUSY refuses invalid
pointers before inspection and is shared with the File gate. Repeated revocation
and destruction preserve the established lifecycle contract.

My 179 flow and 611 hosted allocation-prefix checks still pass. Ordinary
File/TCP grants cannot authorize a WebSocket table. Checked mixed runtime
creation, native emission, product policy parsing and public execution still
refuse WebSocket; I retain those refusal tests until the runtime and source
integration are implemented. Grant-table authorization is not evidence of
checked service execution or live-network qualification. I do not close #990.
