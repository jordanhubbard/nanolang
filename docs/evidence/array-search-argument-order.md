# Array search argument order

I track this correction as `task_1460c2ef56414a7a98e9ce2bcd6b2890`.

I emitted `array_index_of` as an ordinary C call with the array expression and
needle expression as arguments. C leaves their evaluation order unspecified.
On ARM64 my fixture passed, but on x86_64 GCC evaluated the needle first and
failed my `trace == 12` assertion at line 29.

I now bind the array and needle to separate temporaries in source order before
calling either `array_index_of` or `array_contains`. Each input executes once.
I preserve the existing trace 12/121, typed reverse, empty-array, string, and
alias assertions, and add a trace assertion for `array_contains`.

I verified with a fresh coverage compiler build on Linux ARM64, using the CI
`-fprofile-arcs -ftest-coverage` flags. Native execution and NanoVM execution
pass. I also instrumented the emitted native C and runtime at `-O0` with those
coverage flags; the fixture passes.

On madmax (Linux x86_64, GCC 15.2.0), I compiled the preceding and corrected C
outputs with the same runtime sources and `-O0 -fprofile-arcs -ftest-coverage`.
The preceding output exits 1 with:

```text
Contract violation at line 29: (== trace 12)
```

The corrected output exits 0. I also checked both outputs at `-O2` without
coverage instrumentation: the preceding output fails the same assertion and
the corrected output passes. Coverage exposed a platform-dependent ordering
bug; it did not cause the bug.

This change covers the two array search builtins. It does not claim a complete
audit of evaluation order for every native function call.
