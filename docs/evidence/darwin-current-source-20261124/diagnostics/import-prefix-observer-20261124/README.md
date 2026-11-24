# I retain three bounded Darwin diagnostics

I use exact e6a7c3bf634e15b5eb13a2e98feeb93797ef6f27 and the original qualified normal compiler. My [parent receipt](../../README.md) retains the original 600.073-second failure. I change no production source or original failure bytes here.

| Observation | Wall seconds | Native result | Shadows |
| --- | ---: | --- | --- |
| Direct compiler_support import | 16.622 | compile and native run pass | three selected shadows pass |
| Same import through external CC observer | 6.455 | compile and native run pass | three selected shadows pass |
| Original compiler source through observer, deliberate 120-second prefix | 120.170 | -9, timed out | zero starts/completions |

My observer delegates literal arguments to `/usr/bin/cc`. Version and invalid-option controls preserve exact output and exit status. Selecting this observer changes compiler identity/context, so these times establish no gain or generic tool admission. The observed import completes 24 case tool calls (5.140 child wall / 4.500 child CPU seconds). The source prefix completes 23 (28.675 / 27.877 seconds), with one in flight at cutoff. I do not infer duplicate module compilation from a shared directory cache key.

I retain all three roots, 3,520 selected Git source guards each, original tools and compiler images, 494 original provider copies for the prefix, timestamps, control streams, observer source/image, command/resource records and generated C still present at termination. Controller exit zero records successful retention; the prefix native process failed its diagnostic deadline. My ordinary 600-second and private compiler-source acceptance remain open under #1247.

Run `python3 read_archive.py`. I verify 14,445 file records, every compressed/raw chunk, Git source identity, parent manifest binding, source/tool/provider guards, output equality, shadow/native outcomes and the unmatched final observer event. Reused chunks refer to the parent archive; 89 chunks are new. My manifest SHA256 is `e85889d7e431fdcdf7245cb4d517245e7059af83cbbce8ac10583805cdebc294`.

I retain the original launcher metadata with its literal trailing `\n`; it is distinct from native output. I claim no historical timeout cause, performance gain, full qualification or release completion.
