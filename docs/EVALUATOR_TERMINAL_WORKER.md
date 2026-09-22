# I isolate my terminal-match control from earlier test allocations

Under task992713 I retain the exact495e full evaluator first sanitizer terminals.
The existing fork inherits every allocation from the preceding tests, then my
production fatal match path calls exit(EXIT_FAILURE). LSan reports inherited
allocations as well as the child's unfreed fixture inputs. This is separate from
real effect/AST/callable leaks that the full parent suite must still expose.

I replace only that control's fork with posix_spawn of the same executable and a
private worker argument dispatched before all other tests. I preserve the exact
incomplete Nano source, explicit checker bypass, production call_function and
fatal exit. Fixture-owned Environment/AST/token state is registered with atexit
before construction, so both setup refusal and the production exit free those
inputs normally. I do not use _Exit, suppressions or alternate leak settings.
Unexpected return keeps status91; setup refusal keeps90. Parent checks actual
ordinary EXIT_FAILURE plus the exact fatal diagnostic, with bounded retained
stderr. The existing outer test supervisor remains authoritative.

I retain all other tests and test order. Fresh process isolation does not repair
or hide parent leaks: full ordinary and ASan/UBSan/LSan test-eval remains required
after root's independent production ownership repair. Source review precedes
execution of this corrected fixture.
