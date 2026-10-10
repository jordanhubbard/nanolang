# My TCP source-checking checkpoint

I check TCP source bodies and affine ownership in C and Nano before wire
admission. I retain Endpoint construction, nominal Result and field checks,
Conn/ConnectResult moves, branches, loops, exclusive borrows and callables.

- `paired-file-tcp.log`: all 15 methods pass, including the unchanged File
  adjacency corpus and new interleaved-shadow regression (186.622 seconds).
- `callables.log`: the additional callable method passes its three cases
  (37.804 seconds); it was added after the combined run started.
- `mixed-namespace.log`: the mixed File/TCP fixture passes through the C
  producer and installed Stage2, with VM/native execution (152.070 seconds).
- `lowering-boundary.log`: C and Nano lowerers refuse TCP before publishing
  wire output (45.423 seconds). The driver corpus preserves prior output.
- `c-sanitizers.log` and `c-sanitizers.py`: five C-only corpus methods pass;
  one driver-publication method is intentionally skipped because the paired
  corpus covers it. I instrument the checker/parser fixtures with ASan/UBSan
  and leak detection; linked ordinary compiler objects are not all instrumented.
- `gcc-warnings.log`: both changed C checkers compile with GCC16 and strict
  warnings. This is compiler compatibility evidence, not Linux execution.

I retain the original 30 exact ownership-fact comparison failures. C visited
interleaved shadows in source order while Nano visited functions first, shifting
binding IDs. I align C with Nano without weakening exact fact comparisons.
I also retain the initial callable fixture parser failure: `use` is reserved;
I renamed the helper to `exercise` and reran the original callable assertions.

I retain source hashes in `sources.sha256`. My latest full bootstrap predates
this checkpoint. These component checks do not establish a new compiler fixed
point, TCP wire/runtime execution, selected network shadows, or platform release
qualification. Issues #989/#990 remain open; network restrictions prevented
publishing this checkpoint or updating their remote state during this session.
