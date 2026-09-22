# My measured interpreter search cost

I ran one full-graph diagnostic at4369b6097 against independently copied and
verified46b inputs/providers. Fresh main/eval and the Make-selected unchanged
wrapper generator rebuilt before linking. The original ten-second child bound
stopped execution; the120-second outer bound did not expire. The command
returned1 after50.668 seconds.

My [analysis](list-search-timing/analysis.json) retains2,198 numeric markers,
1,066 completed shadows and one unfinished interval. Across completed marker
deltas,78 transp_str_index_of invocations consume1.998073 seconds wall and
1.989835 seconds process CPU. The largest numeric shadow interval consumes
2.054069 seconds, including1.995489 seconds in10 measured invocations.
No clock failure, counter overflow or record truncation is observed.

These are inclusive diagnostic measurements. I do not add overlapping parent
and child timings or attribute the final unmarked interval. An attempted old
source-order label map did not cover the current module graph; I retain numeric
identifiers and make no name attribution from that map. Source-level inspection
and exact registered-function selection establish the measured helper.

[Build status](list-search-timing/fresh-observer-build-status.json),
[command status](list-search-timing/timed-full-driver-status.json), and
[closure inventory](list-search-timing/inherited-diagnostic-inputs.json) retain
provenance. Full raw maps/logs/CAS remain under the persistent qualification
root named there. This is a bounded measurement report, not final archive
sealing or compiler acceptance. Production optimization requires review and
all original gates with the same deadline.
