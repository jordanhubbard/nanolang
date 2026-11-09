# My actual shadow trace write cost

I mechanically copied the two current formatter bodies from `a561d8c97d3cbfd7954efe7d3625d09ef1476223`. I replayed the retained complete Apple trace: 1,286 starts and 1,286 completions, 550,034 output bytes. I did not run a compiler in this witness.

My twenty uncounted native replays to unbuffered regular files took a median 0.495776 seconds, with quartiles 0.48993675 and 0.50866625 seconds (range 0.482198–0.529163). Each output exactly matched the original trace bytes. A separate counted image observed 292,890 fputc calls; it is excluded from the distribution. My current copied bodies contain no fwrite call.

I passed controls for every nonzero byte, null/empty/embedded-NUL/UTF-8/long strings, complete event bytes, boundary flushing from an initially buffered stderr, and visible stream errors. I executed the same controls with plain, counted and ASan/UBSan native images; image and input hashes remained unchanged. My first counted build rejected an unused witness helper under -Werror. I retain that refusal and its original source; no timed sample ran before correction.

`plan.json` predates measurement. `report.json`, `operations.json` and `samples.json` retain all commands, identities and timings. `retained-texts.json.gz` contains every captured operation stream, every actual replay output, generated input header and native witness source. Local executable paths in the report identify the images I ran; the images are not distributed here.

This observation covers this machine, fixture and regular-file destination. It does not measure whole-compiler benefit, terminal/pipe behavior, syscall counts, durability, all partial-I/O recovery behavior, complete toolchain closure or release acceptance. I retain my production writer unchanged. A change requires separate whole-compiler evidence; I do not treat a half-second replay as a cure for the retained source deadlines.
