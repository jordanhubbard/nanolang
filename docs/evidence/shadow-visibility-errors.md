# My shadow visibility errors stop publication

My unqualified function-access helper printed an error and returned false,
but did not record a typechecking error. Its caller returned `TYPE_UNKNOWN`.
Because my checker accepts unknown types in several contexts, a private call
could pass shadow checking and compilation could publish an executable.

I now report failed function visibility/import access through my existing
counted diagnostic path. Private calls use `E009`, matching qualified private
calls. I do not change unknown-type compatibility as part of this repair.

## Evidence

The new `test_private_shadow_calls_fail_before_publication` regression in
`tests/test_cseed_import_shadows.py` reproduced four incorrectly successful
compilations before the fix: root and dependency shadows, each with a bare
private call or an integer local initialized by that call.

With the fix, all four fail before shadow execution, return structured `E009`
private-access diagnostics, and preserve an existing output artifact. Making
the function public then compiles and runs successfully for each fixture.
The default dependency-shadow selection is unchanged.

`make test-cseed-import-shadows` rebuilds my C seed and passes all ten methods.
The final suite run took 29.151 seconds; its log is
`/tmp/nanolang-shadow-visibility-final.log` on this macOS host. The failed
baseline is `/tmp/nanolang-shadow-visibility-before.log`.

The same serial command ran `test-typechecker`: both self-hosted stages rebuilt,
their smoke tests and the installed no-C-seed check passed, and all C
typechecker unit tests passed. Native bootstrap binary equality is not
established by these checks.

MAC task: `task_22bb4774aeb145f3bcd15d60c532a1b1`.
This is not a claim that every legacy printed diagnostic increments an error
count, that unknown-type compatibility is sound, or that all module visibility
paths are complete.
