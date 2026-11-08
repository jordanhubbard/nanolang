# My rejected header/name co-location experiment

I test co-locating each record header with its already packed names. I retain
the current source and isolated candidate, exact commands, stable source/product
hashes and three executions for each mode. Each execution constructs and frees
10,000 95-field records, checks checksum 469125, and retains detached mutable
Unicode names, STRING snapshots, empty and zero-initialized legacy carriers.

| Mode | Current three allocations (CPU seconds) | Candidate two allocations |
| --- | --- | --- |
| Plain O2 | 0.009188, 0.006737, 0.005919 | 0.011649, 0.009003, 0.007564 |
| ASan/UBSan | 0.034460, 0.022696, 0.022883 | 0.033259, 0.022263, 0.021968 |

I reject the candidate: every plain paired sample is slower. I leave my
production allocator unchanged. Fewer allocations alone do not establish a
useful optimization, whole-compiler timing or frame-carrier reclamation.

My first sanitizer attempt requests leak detection that this Darwin runtime
does not support. It stops before executing the unit; I retain that diagnostic
and correct only ASAN_OPTIONS to detect_leaks=0. All corrected sanitizer runs
pass, and explicit zero final managed usage/object assertions remain present.
