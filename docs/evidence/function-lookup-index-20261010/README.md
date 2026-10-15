# Function lookup and compiler shadow cost

I retain the reproduced ten-second compiler-suite timeout at `fc7d94c3f` in
my adjacent compiler-product evidence. I measure one candidate cost before
changing lookup: a temporary compiler instruments `env_get_function` with
monotonic timestamps, reset/reported per shadow (`profile.patch`). My parser
import harness completes 451,431 lookups, spending 0.459568 of 1.966027 shadow
seconds inside lookup. Instrumentation adds clock overhead; this is a cost
attribution, not a production benchmark. I retain the build script and log.

I add an environment-owned optional function-name index. Hash buckets store
slot numbers rather than pointers or cached ownership. I select the earliest
matching declaration and read module/body data from current slots. Qualified
namespace visibility, builtin precedence and the local array_push exception
remain unchanged. Append/reallocation updates the index; allocation failure
uses the complete linear scan. Environment destruction releases the index.

My independent declaration-order oracle checks 513 declarations with duplicate
names and three owners, every append/reallocation, qualified aliases,
builtin precedence, live ownership/body data and all three index allocation
failures. The index tests pass normally and with Homebrew Clang ASan/UBSan
(`-fsanitize=address,undefined -fno-sanitize-recover=all -O1`). I instrument
the env.c test translation unit; other linked compiler objects are ordinary.
Environment checks pass 60 assertions and ten lexical-scope methods; evaluator
units also pass (`units.log`).

My five Python suites run 43 methods: capture, mutable records, lexical scope,
imported globals and service namespaces (`regressions.log`). Forty-two pass.
The mixed service control expects an old unconnected-TCP diagnostic, but both
my retained pre-index instrumented compiler and current compiler instead
require the missing TCP capability grant and preserve existing output
(`service-baseline.log`). I retain this preexisting discrepancy as open; I do
not weaken its assertion or claim the complete service suite passed.

My unchanged full compiler command is:

```
NANO_SHADOW_TRACE=1 bin/nanoc_c src_nano/nanoc_v06.nano -o /private/tmp/nl51-function-index-compiler
```

It exits 1 at the ten-second deadline after 831 completed shadows, all without
assertion failures, during nisa_emit_array_slice. The previous trace completed
678 shadows. Those same 678 completed shadows take 7.221598 seconds here versus
9.553542 previously. This run overlapped adjacent regression execution; these
are observed timings, not a controlled performance guarantee. I preserve the
full failure (`indexed-compiler-trace.log`): the correction reduces lookup
cost but does not close the compiler-product deadline or qualify 5.1.
