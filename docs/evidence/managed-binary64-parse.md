# My portable binary64 parsing evidence

I implement parser task `task_4d2f69a19d754ac88876f93a0913d1fb` and canonical
NaN policy task `task_89fd02e9aa4d4a4faca33ba5fbc2b703` under my open managed
parent51da. Contract `6de7b170` preceded implementation `c9bd0d47`.

My allocation-free parser uses checked fixed-capacity integer arithmetic for
nearest-even binary64 rounding. I retain 800 decimal digits or 32 hexadecimal
digits plus a sticky tail. My contract proves every binary64 midpoint has at
most 770 significant decimal digits; a discarded nonzero tail can change an
exact lower-endpoint tie but cannot cross an interior rounding boundary.

My corpus has 1,131 ordinary inputs: decimal/hexadecimal syntax and suffixes,
embedded NUL, huge exponents and long mantissa/exponent cancellation, signed
zero, finite/subnormal/overflow boundaries, infinities and NaN payloads. Exact
midpoints and plus/minus 10^-1400 controls exercise retained-prefix ties,
including the subnormal/normal boundary. Deterministic binary64 samples use
hexadecimal, 17-digit and long decimal spellings. My reference uses native
strtod under the closed C-locale/default-rounding contract.

All 1,131 expected bit patterns pass through the actual VM, both direct and
boxed generated-C conversion helpers, native ASan/UBSan managed core, and
import-free Node/Wasmtime core. I also run 75 selected values through emitted
LLVM/Wasm with dynamic concatenation, global aliases, helper calls, repeated
entry and final disposal. Parsing preserves two retained owners while new
allocation is disabled. Existing cleanup/failure acceptance remains separate
and included in the managed suite.

I explicitly normalize overflowing numeric NaN payloads to the existing Linux
unsigned64 saturation policy. My original reference logs are
`/tmp/nanolang-managed-parser-linux.txt` and
`/tmp/nanolang-managed-parser-darwin.txt`: the ordinary decimal overflow case
produced `7fffffffffffffff` on Linux and `7ff8000000000000` on Darwin. This is an
intentional Darwin semantic change, not accidental libc parity. A fresh pure
parser O0 harness on puck matches all 1,131 portable expected bits; its log is
`/tmp/nanolang-managed-parser-darwin-progress.log`.

My separate Darwin managed-core ASan/UBSan run did not finish within the bounded
check and was terminated after about 110 seconds of CPU execution. I retain
`/tmp/nanolang-managed-parser-darwin-core.log` and the original remote artifacts
in `/tmp/nanolang-binary64-parse-4d2f`. I establish no cause and claim no Darwin
managed sanitizer completion. Required platform follow-up
`task_7ba59bf363f7454ba50bf3bbd965db8e` remains open.

The generated-C test initially inherited an embedded-NUL string pool that the
existing native subset refuses. I corrected the ordinary module fixture and
kept NUL semantics in the direct parser corpus. My first combined Make gate
also exposed a missing new check target, then an outdated expected manifest
source list; both harness changes are corrected. These earlier logs remain
`/tmp/nanolang-managed-parser-focused.log` and
`/tmp/nanolang-managed-parser-integrated{,-r2}.log`.

My raw parser header is hashed into the managed package manifest. I check its
embedded native source for exact drift, and explicit Make dependencies rebuild
the VM/native translator when the relevant header changes. CLOSED_SCALAR and
CLOSED_LITERAL_STRINGS retain their previous refusal contracts; only the
managed profile admits string CAST_FLOAT. Legacy AST conversion/endptr policy
remains required task `task_9e93c1badb1a4da093a737b3a2c15ef7`; I do not claim it
or full managed-runtime/release acceptance here.

My frozen Linux ARM64 gates pass: 11 scalar-global methods, nine literal
methods, two package methods, three core methods, 22 managed methods, shared
profile decisions with output preservation, 1,269 shape checks and 2,422 native
checks. Logs are `/tmp/nanolang-managed-parser-integrated.log` (globals/literals)
and `/tmp/nanolang-managed-parser-integrated-r3.log` (remaining corrected gates).
The focused parser rerun also passes all three methods in 2.188 seconds.
