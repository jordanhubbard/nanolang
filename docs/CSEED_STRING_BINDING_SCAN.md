# My managed STRING binding hint

My unchanged C interpreter scans all symbols before binding a STRING that may
already belong to another binding. My source-pinned counter diagnostic records
746,387,154 declaration probes: 90,589 of 91,472 scans have no alias. Reversing
the scan cannot remove this work.

I reserve one existing GC header flag for a monotonic fact: this managed STRING
has been published in an environment binding. My allocator already clears flags
on every allocation. A never-published managed STRING cannot match a live STRING
binding when every publication records that fact. I may skip its alias scan.
Once published, I retain the existing exact scan, including its current
same-binding exclusion and single retain. The flag stays set after reassignment,
scope exit and environment destruction while the object remains alive. It is
not a liveness, owner-count, environment-membership or allocation-success claim.

I mark regular declarations, replacements, borrowed/identity assignments,
legacy imported STRING literals and direct static/dynamic/list loop values.
Checkpoint and REPL rollback restore already-marked bindings. Direct internal
Symbol value publications must call my marker. The marker checks the registry
before touching a header; unmanaged STRING pointers remain outside this hint.
I retain every other flag bit and change no header layout or allocation path.

My gate must check fresh and already-bound values, all alias positions,
multiple aliases, old/global values, assignment, borrowed/control-flow values,
scopes, rollback and allocator address reuse. My existing ownership, imported
literal, string-loop, source-only/shadow and native bootstrap gates remain
required. My implementation and local semantic/sanitizer regressions now pass.
My hardware diagnostic completes all1139 shadows at the original10-second
limit; a preceding original-limit observation still fails and remains retained.
Fresh hosted original bootstrap/native/emitter/cache and full release
qualification remain pending. I track the reproduced prior temporary-array
argument leak separately in #1266.
