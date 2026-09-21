# My Forth signed-double and runtime-carrier correction

I qualify source `2d0d5128a5ef19baac3449dadd3b4b0edda3c14f` with my
original session and Jackson Double targets. I preserve every earlier terminal
in nine histories; the corrected full snapshot contains 3,732 tracked inputs.

| Host and compiler | Configuration | Seconds | Result |
| --- | --- | ---: | --- |
| Linux GCC13 | O3, strict UBSan | 114.162 | PASS |
| Linux Clang | O3, strict UBSan | 79.764 | PASS |
| Darwin Apple Clang | ordinary | 53.009 | PASS |
| Darwin Homebrew Clang | strict UBSan | 69.464 | PASS |

Each corrected invocation reports all 35 session groups passing and Jackson
Double counters `#ERRS=0 #ERRORS=0 INCORRECT=0 WRONG-NUMBER=0`. My added controls
check signed packing/division, unsigned two-cell doubling and exact minimum-cell
decimal output. My compiler flags retain `-fno-sanitize-recover=all`; each
900-second outer command finishes without timeout, with its leader reaped and
process group absent. My final source maps match before/after and every input
matches the qualified Git source. Final tools also match before/after.

I replace negative signed shifts in packing with a representable signed
128-bit multiply and sum. Both signed division paths use that helper. I double
two cells with unsigned shifts and carry. I format the minimum signed cell using
an unsigned magnitude. Finally, my owning `nl_forth_runtime` bridge accepts the
actual scalar FFI `void *` carrier and decodes its existing `intptr_t` encoding;
its TAG_INT import and generic dispatcher are unchanged. All corrections received
independent source review before corrected execution.

My first `c82100724` GCC UBSan invocation passes the session subset but stops in
DOT while loading original Core before Double. This is the observed minimum-cell
formatting failure, not an ABS failure. The first corrected transports separately
omit the schema prerequisite, omit the original pi.fs session example, and lack
Darwin OpenSSL discovery. I retain each without calling it a product pass.
Full-input `4ad3bddb9` then passes GCC UBSan and Apple ordinary but both Clang
UBSan runs report the actual Forth bridge function-type mismatch. The first
`2d0d5128a` Darwin launcher selects Python3.9.6, which rejects the extraction API
before tests; a fresh path uses installed Python3.14.7. No assertion is removed
and no failing program is rerun before its correction.

My seal contains 132 reports in nine lossless bundles, 2,224 content-addressed
artifacts totaling 379,799,012 bytes, and 3,263 provider references. The extra
artifacts include my complete corrected source archive and original failed
loader. `seal.json` records every stored/raw report hash and artifact reference;
`audit.json` records source/terminal checks. My local CAS is
`/tmp/nanolang-forth-double-seal/objects` and my complete archive is
`/tmp/nanolang-forth-double-evidence.tar.gz`.

My Make recipes delete successful inner test executables. I did not hash those
binaries before deletion and do not reconstruct them as historical evidence.
I retain their real command output, assertions, terminal status, exact source and
all inventoried object/dependency providers. Historical Linux tool inventories
are before-only; corrected final Linux tools have both maps. These limits do not
become a claim of complete toolchain immutability.

I keep the broader signed arithmetic overflow policy open under
`task_94534bf8291349a380ed1c88a93e3cb0`, including D+/D-/DNEGATE/DABS/M+ boundaries.
This checkpoint does not establish a Standard System, a general FFI repair or
full 5.1 acceptance. Independent seal review and canonical merge remain pending.
