# My nested record-array storage views

My clean full compiler-product gate at 9049b58e6 ran 109 methods and failed
the two compiler translation routes. I preserve its full log and terminal
receipt (Make exit 2, 429.744 seconds, unchanged clean source).
The optional/string conflict reproduces on my retained bootstrap seed module;
the preceding translator handles that same module successfully.

I trace the conflict to parser records passed into the type-name rewriter.
A copied record still contains shared record-array handles. I propagate the
checked-view mode when conversion traverses those array elements, retaining
payload compatibility and runtime projection guards. Ordinary exact scalar
conversions remain strict. My graph controls test that distinction and reject
incompatible optional payloads. They pass 2,635 checks; the full native suite
passes 2,431 and callable constraints pass 379.

The corrected translator emits my retained seed compiler. Strict C11 compilation
with Homebrew LLVM succeeds. That native compiler and my C-seed producer each
pass the ten-product diagnostic/alias mutation matrix under ASan/UBSan/LSan.
The source check selects `/private/tmp/nanolang-nested-view-compiler` through
NANOLANG_SELFHOST_COMPILER and `/private/tmp/nanolang-diag-tools/bin/nvm2c`
through NANOLANG_TEST_NVM2C; it runs `tests.test_native_mutable_record_arrays`.
My clean complete compiler-product gate at `763d786cf` now passes all 109
methods in 828.725 seconds (Make exit 0, 830.232 seconds). Both generated
compiler routes include the capture and mutable-record-array controls. My
terminal receipt records unchanged HEAD and a clean tree before and after;
I retain the runner, full log and receipt under `clean-763d786cf-*`.

My additional `holder-regression.py` remains a failing reproducer: wrapping a
shared array in a record before helper mutation loses the reverse write facts
through both source producers. I preserve that failure rather than equating
read views with writes. A broader reverse-flow experiment passes this fixture
but violates an existing function-target assertion (2,634 shape checks pass,
one fails). I withdraw it and retain its patch/log. Its seed translation also
takes much longer; it eventually exits 0. My guarded stop attempt found no
matching live process and sends no signal. No release or full alias-flow
completion is claimed.
